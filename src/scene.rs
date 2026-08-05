use bus::Bus;
use core::f32;
use regex::Regex;
use std::{
    cmp::Ordering,
    collections::HashMap,
    io::{BufRead, BufReader, Cursor, Read, Seek},
    sync::{Arc, LazyLock, Mutex},
};
//use wasm_thread as thread;

use crate::log;
use crate::utils::*;

const MAX_PLY_HEADER_LINES: usize = 1024;
const SH_C0: f32 = 0.28209479177387814;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalarType {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PlyProperty {
    offset: usize,
    scalar_type: PlyScalarType,
}

#[derive(Debug)]
struct PlyHeader {
    data_offset: usize,
    splat_count: usize,
    vertex_stride: usize,
    properties: HashMap<String, PlyProperty>,
}

static CONSTRUCTOR_TILE_FILENAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^tile(\d+)_lod(\d+)\.(?:ply|splat)$").unwrap());
static LEGACY_TILE_FILENAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^lod(\d+)_tile_(\d+)\.(?:ply|splat)$").unwrap());

fn parse_tile_filename(filename: &str) -> Option<(usize, usize)> {
    if let Some(captures) = CONSTRUCTOR_TILE_FILENAME.captures(filename) {
        let tile_id = captures.get(1)?.as_str().parse().ok()?;
        let lod_id = captures.get(2)?.as_str().parse().ok()?;
        return Some((lod_id, tile_id));
    }

    let captures = LEGACY_TILE_FILENAME.captures(filename)?;
    let lod_id = captures.get(1)?.as_str().parse().ok()?;
    let tile_id = captures.get(2)?.as_str().parse().ok()?;
    Some((lod_id, tile_id))
}

/// A point cloud of Gaussian splats
pub struct Scene {
    pub splat_count: usize,
    pub(crate) buffer: Vec<u8>,
    pub(crate) tex_data: Vec<u32>,
    pub(crate) tex_width: usize,
    pub(crate) tex_height: usize,
    prev_vp: Mutex<Vec<f32>>,
}
impl Scene {
    pub fn new() -> Self {
        Self {
            splat_count: 0,
            buffer: Vec::<u8>::new(),
            tex_data: Vec::<u32>::new(),
            tex_width: 0,
            tex_height: 0,
            prev_vp: Mutex::new(Vec::<f32>::new()),
        }
    }

    fn parse_ply_header(bytes: &[u8]) -> Result<PlyHeader, String> {
        let mut reader = BufReader::new(Cursor::new(bytes));
        let mut line = String::new();
        let mut line_number = 0_usize;
        let mut saw_ply = false;
        let mut saw_binary_little_endian = false;
        let mut current_element = String::new();
        let mut splat_count = None;
        let mut vertex_stride = 0_usize;
        let mut properties = HashMap::new();

        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .map_err(|error| format!("could not read PLY header: {error}"))?;
            if bytes_read == 0 {
                return Err("PLY header ended before end_header".to_string());
            }

            line_number += 1;
            if line_number > MAX_PLY_HEADER_LINES {
                return Err(format!(
                    "PLY header exceeds the supported limit of {MAX_PLY_HEADER_LINES} lines"
                ));
            }

            let header_line = line.trim_end_matches(['\r', '\n']);
            if line_number == 1 {
                if header_line != "ply" {
                    return Err("file does not start with a PLY header".to_string());
                }
                saw_ply = true;
                continue;
            }

            if header_line == "end_header" {
                break;
            }

            let fields = header_line.split_whitespace().collect::<Vec<_>>();
            match fields.as_slice() {
                ["format", "binary_little_endian", "1.0"] => {
                    saw_binary_little_endian = true;
                }
                ["format", format, version] => {
                    return Err(format!(
                        "unsupported PLY format '{format} {version}'; expected binary_little_endian 1.0"
                    ));
                }
                ["element", name, count] => {
                    current_element = (*name).to_string();
                    if *name == "vertex" {
                        splat_count = Some(count.parse::<usize>().map_err(|_| {
                            format!("invalid vertex count '{count}' in PLY header")
                        })?);
                    }
                }
                ["property", "list", ..] if current_element == "vertex" => {
                    return Err("list-valued vertex properties are not supported".to_string());
                }
                ["property", scalar_type, name] if current_element == "vertex" => {
                    let scalar_type = PlyScalarType::parse(scalar_type).ok_or_else(|| {
                        format!("unsupported PLY scalar type '{scalar_type}' for property '{name}'")
                    })?;
                    if properties
                        .insert(
                            (*name).to_string(),
                            PlyProperty {
                                offset: vertex_stride,
                                scalar_type,
                            },
                        )
                        .is_some()
                    {
                        return Err(format!("duplicate PLY vertex property '{name}'"));
                    }
                    vertex_stride = vertex_stride
                        .checked_add(scalar_type.size())
                        .ok_or_else(|| "PLY vertex stride overflowed usize".to_string())?;
                }
                _ => {}
            }
        }

        if !saw_ply {
            return Err("missing PLY signature".to_string());
        }
        if !saw_binary_little_endian {
            return Err("PLY must use binary_little_endian 1.0".to_string());
        }
        let splat_count = splat_count.ok_or_else(|| "missing PLY vertex element".to_string())?;
        let data_offset = reader
            .stream_position()
            .map_err(|error| format!("could not locate PLY vertex data: {error}"))?
            as usize;

        Ok(PlyHeader {
            data_offset,
            splat_count,
            vertex_stride,
            properties,
        })
    }

    /// Loads binary little-endian Gaussian PLY data by property name.
    /// Optional properties such as normals and higher-order SH coefficients are ignored.
    pub fn from_ply_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let header = Self::parse_ply_header(&bytes)?;
        let property_offset = |name: &str| -> Result<usize, String> {
            let property = header
                .properties
                .get(name)
                .ok_or_else(|| format!("missing required PLY vertex property '{name}'"))?;
            if property.scalar_type != PlyScalarType::F32 {
                return Err(format!(
                    "PLY vertex property '{name}' must be float32, found {:?}",
                    property.scalar_type
                ));
            }
            Ok(property.offset)
        };

        let position_offsets = [
            property_offset("x")?,
            property_offset("y")?,
            property_offset("z")?,
        ];
        let color_offsets = [
            property_offset("f_dc_0")?,
            property_offset("f_dc_1")?,
            property_offset("f_dc_2")?,
        ];
        let opacity_offset = property_offset("opacity")?;
        let scale_offsets = [
            property_offset("scale_0")?,
            property_offset("scale_1")?,
            property_offset("scale_2")?,
        ];
        let rotation_offsets = [
            property_offset("rot_0")?,
            property_offset("rot_1")?,
            property_offset("rot_2")?,
            property_offset("rot_3")?,
        ];

        let vertex_bytes = header
            .splat_count
            .checked_mul(header.vertex_stride)
            .ok_or_else(|| "PLY vertex data size overflowed usize".to_string())?;
        let expected_length = header
            .data_offset
            .checked_add(vertex_bytes)
            .ok_or_else(|| "PLY file size overflowed usize".to_string())?;
        if bytes.len() < expected_length {
            return Err(format!(
                "truncated PLY vertex data: expected at least {expected_length} bytes, found {}",
                bytes.len()
            ));
        }

        let read_f32 = |record: &[u8], offset: usize| -> f32 {
            f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap())
        };
        let record_at = |index: usize| -> &[u8] {
            let start = header.data_offset + index * header.vertex_stride;
            &bytes[start..start + header.vertex_stride]
        };

        let importance = (0..header.splat_count)
            .map(|index| {
                let record = record_at(index);
                let scale = scale_offsets
                    .iter()
                    .map(|&offset| read_f32(record, offset).exp())
                    .product::<f32>();
                let opacity = 1.0 / (1.0 + (-read_f32(record, opacity_offset)).exp());
                scale * opacity
            })
            .collect::<Vec<_>>();
        let mut sorted_indices = (0..header.splat_count).collect::<Vec<_>>();
        sorted_indices.sort_by(|&a, &b| {
            importance[b]
                .partial_cmp(&importance[a])
                .unwrap_or(Ordering::Equal)
        });

        const ROW_LENGTH: usize = 32;
        let mut buffer = vec![0_u8; ROW_LENGTH * header.splat_count];
        for (output_index, source_index) in sorted_indices.into_iter().enumerate() {
            let record = record_at(source_index);
            let output = &mut buffer[output_index * ROW_LENGTH..(output_index + 1) * ROW_LENGTH];

            for (component, &offset) in position_offsets.iter().enumerate() {
                let start = component * 4;
                output[start..start + 4].copy_from_slice(&read_f32(record, offset).to_le_bytes());
            }
            for (component, &offset) in scale_offsets.iter().enumerate() {
                let start = 12 + component * 4;
                output[start..start + 4]
                    .copy_from_slice(&read_f32(record, offset).exp().to_le_bytes());
            }
            for (component, &offset) in color_offsets.iter().enumerate() {
                output[24 + component] = ((0.5 + SH_C0 * read_f32(record, offset)) * 255.0) as u8;
            }
            output[27] = ((1.0 / (1.0 + (-read_f32(record, opacity_offset)).exp())) * 255.0) as u8;

            let rotation = rotation_offsets.map(|offset| read_f32(record, offset));
            let rotation_length = rotation
                .iter()
                .map(|value| value.powi(2))
                .sum::<f32>()
                .sqrt();
            if !rotation_length.is_finite() || rotation_length == 0.0 {
                return Err(format!(
                    "PLY vertex {source_index} has an invalid zero or non-finite rotation"
                ));
            }
            for (component, value) in rotation.into_iter().enumerate() {
                output[28 + component] = (((value / rotation_length) + 1.0) * 0.5 * 255.0) as u8;
            }
        }

        let mut scene = Self::new();
        scene.splat_count = header.splat_count;
        scene.buffer = buffer;
        Ok(scene)
    }

    /// Generates a 2D texture from the splats
    pub fn generate_texture(&mut self) {
        // TODO: parallelize
        if self.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());
        let u_buffer: &[u8] = transmute_slice::<_, u8>(self.buffer.as_slice());

        let texwidth = 1024 * 2 as usize;
        let texheight = ((2 * self.splat_count) as f64 / texwidth as f64).ceil() as usize;
        let len_texdata = texwidth * texheight * 4 as usize; // 4 components per pixel (RGBA)
        log!(
            "Scene::generate_texture(): texheight={}, len_texdata={}",
            texheight,
            len_texdata
        );
        let mut texdata = vec![0_u32; len_texdata];
        // texdata structure: 32B (2 pixels) per gs, 1024 gs (2048 pixels) per row
        // |                          32B                          |
        // |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |
        // | posx | posy | posz | none |  ab  |  cd  |  ef  | rgba |

        {
            let texdata_f = transmute_slice_mut::<_, f32>(texdata.as_mut_slice());
            for i in 0..self.splat_count {
                // x, y, z components of the i-th splat in f_buffer
                let index_f: usize = 8 * i;
                texdata_f[index_f + 0] = f_buffer[index_f + 0];
                texdata_f[index_f + 1] = f_buffer[index_f + 1];
                texdata_f[index_f + 2] = f_buffer[index_f + 2];
            }
        }

        {
            let texdata_c = transmute_slice_mut::<_, u8>(texdata.as_mut_slice());
            for i in 0..self.splat_count {
                // r, g, b, a components of the i-th splat in u_buffer
                let index_c: usize = 4 * (8 * i + 7);
                let index_u: usize = 32 * i + 3 * 4 + 3 * 4;
                texdata_c[index_c + 0] = u_buffer[index_u + 0];
                texdata_c[index_c + 1] = u_buffer[index_u + 1];
                texdata_c[index_c + 2] = u_buffer[index_u + 2];
                texdata_c[index_c + 3] = u_buffer[index_u + 3];
            }
        }

        for i in 0..self.splat_count {
            let index_f: usize = 8 * i;
            let scale = [
                f_buffer[index_f + 3],
                f_buffer[index_f + 4],
                f_buffer[index_f + 5],
            ];

            let index_u: usize = 32 * i + 3 * 4 + 3 * 4 + 4;
            let rot = [
                // [0, 255] -> [-1, 1]
                ((u_buffer[index_u + 0] as f32) / 255.0) * 2.0 - 1.0, // qw
                ((u_buffer[index_u + 1] as f32) / 255.0) * 2.0 - 1.0, // qx
                ((u_buffer[index_u + 2] as f32) / 255.0) * 2.0 - 1.0, // qy
                ((u_buffer[index_u + 3] as f32) / 255.0) * 2.0 - 1.0, // qz
            ];

            let r = Mat3::new(
                // column-major
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            );

            let s = Mat3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]);

            let m = r * s;
            let m = &[
                // column-major: [col][row]
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            ];

            // M * M^T = R * S * S^T * R^T
            let sigma = [
                m[0] * m[0] + m[3] * m[3] + m[6] * m[6],
                m[0] * m[1] + m[3] * m[4] + m[6] * m[7],
                m[0] * m[2] + m[3] * m[5] + m[6] * m[8],
                m[1] * m[1] + m[4] * m[4] + m[7] * m[7],
                m[1] * m[2] + m[4] * m[5] + m[7] * m[8],
                m[2] * m[2] + m[5] * m[5] + m[8] * m[8],
            ];

            // JavaScript typically uses the host system's endianness
            // (x86-64 and Apple CPUs are little-endian).
            // WASM's linear memory is always little-endian.
            texdata[index_f + 4] = pack_half_2x16(4.0 * sigma[0], 4.0 * sigma[1]); // a, b
            texdata[index_f + 5] = pack_half_2x16(4.0 * sigma[2], 4.0 * sigma[3]); // c, d
            texdata[index_f + 6] = pack_half_2x16(4.0 * sigma[4], 4.0 * sigma[5]); // e, f
        }

        self.tex_data = texdata;
        self.tex_width = texwidth;
        self.tex_height = texheight;
    }

    /// Sorts the splats based on their depth using 16-bit single-pass counting sort
    pub fn sort(scene: &Arc<Self>, view_proj: &[f32], bus: &mut Bus<Vec<u32>>, n_threads: usize) {
        if scene.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(scene.buffer.as_slice());

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            if (*mutex).is_empty() {
                (*mutex).push(view_proj[2]);
                (*mutex).push(view_proj[6]);
                (*mutex).push(view_proj[10]);
            } else {
                let dot = (*mutex)[0] * view_proj[2]
                    + (*mutex)[1] * view_proj[6]
                    + (*mutex)[2] * view_proj[10];
                if (dot - 1.0).abs() < 0.01 {
                    return;
                }
            }
        }

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; scene.splat_count];
        for i in 0..scene.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..scene.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..scene.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; scene.splat_count];
        for i in 0..scene.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        //////////////////////////////////
        // no cloning is happening for the single-consumer case
        let _ = bus.try_broadcast(depth_index);
        //////////////////////////////////

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            (*mutex)[0] = view_proj[2];
            (*mutex)[1] = view_proj[6];
            (*mutex)[2] = view_proj[10];
        }
    }

    pub fn sort_self(&self, view_proj: &[f32]) -> (Vec<u32>, Vec<i32>) {
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; self.splat_count];
        for i in 0..self.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..self.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let raw_depth = size_list.clone();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..self.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; self.splat_count];
        for i in 0..self.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        (depth_index, raw_depth)
    }

    pub fn sort_merged(
        view_proj_z: Vec3,
        scene_vec: Vec<&Self>,
        scene_offset: Vec<Vec3>,
    ) -> Vec<(usize, usize)> {
        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        let mut full_splat_count: usize = 0;
        let mut size_list: Vec<i32> = Vec::new();
        let mut splat_displ: Vec<usize> = vec![0];
        for scene_id in 0..scene_vec.len() {
            let f_buffer: &[f32] = transmute_slice::<_, f32>(scene_vec[scene_id].buffer.as_slice());
            let mut local_size_list: Vec<i32> = (0..scene_vec[scene_id].splat_count)
                .map(|i| {
                    let index_f = 8 * i as usize;
                    let depth = ((view_proj_z.x
                        * (f_buffer[index_f + 0] + scene_offset[scene_id].x)
                        + view_proj_z.y * (f_buffer[index_f + 1] + scene_offset[scene_id].y)
                        + view_proj_z.z * (f_buffer[index_f + 2] + scene_offset[scene_id].z))
                        * 4096.0) as i32;
                    if depth > max_depth {
                        max_depth = depth;
                    }
                    if depth < min_depth {
                        min_depth = depth;
                    }
                    depth
                })
                .collect();
            size_list.append(&mut local_size_list);
            full_splat_count += scene_vec[scene_id].splat_count;
            splat_displ.push(full_splat_count);
        }
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..full_splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index: Vec<(usize, usize)> = vec![(0, 0); full_splat_count];
        for scene_id in 0..scene_vec.len() {
            for i in splat_displ[scene_id]..splat_displ[scene_id + 1] {
                let depth = size_list[i] as usize;
                let j = starts0[depth] as usize;
                // depth_index[j] = (i - splat_displ[scene_id]) as u32 + scene_index_offset[scene_id];
                depth_index[j] = (scene_id, i - splat_displ[scene_id]);
                starts0[depth] += 1;
            }
        }
        depth_index.reverse(); // FIXME

        depth_index
    }

    pub fn sort_raw_depth_vec(raw_depth_vec: Vec<&Vec<i32>>) -> Vec<(usize, usize)> {
        let mut full_splat_count: usize = 0;
        let mut size_list: Vec<i32> = Vec::new();
        let mut splat_displ: Vec<usize> = vec![0];
        for scene_id in 0..raw_depth_vec.len() {
            size_list.extend(raw_depth_vec[scene_id]);
            full_splat_count += raw_depth_vec[scene_id].len();
            splat_displ.push(full_splat_count);
        }
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);
        let min_depth = *size_list.iter().min().unwrap();
        let max_depth = *size_list.iter().max().unwrap();

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..full_splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index: Vec<(usize, usize)> = vec![(0, 0); full_splat_count];
        for scene_id in 0..raw_depth_vec.len() {
            for i in splat_displ[scene_id]..splat_displ[scene_id + 1] {
                let depth = size_list[i] as usize;
                let j = starts0[depth] as usize;
                // depth_index[j] = (i - splat_displ[scene_id]) as u32 + scene_index_offset[scene_id];
                depth_index[j] = (scene_id, i - splat_displ[scene_id]);
                starts0[depth] += 1;
            }
        }
        depth_index.reverse(); // FIXME

        depth_index
    }

    /// Sorts the splats based on their depth using 16-bit single-pass counting sort
    pub fn sort2(scene: &Self, view_proj: &[f32], bus: &mut Bus<Vec<u32>>, n_threads: usize) {
        if scene.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(scene.buffer.as_slice());

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            if (*mutex).is_empty() {
                (*mutex).push(view_proj[2]);
                (*mutex).push(view_proj[6]);
                (*mutex).push(view_proj[10]);
            } else {
                let dot = (*mutex)[0] * view_proj[2]
                    + (*mutex)[1] * view_proj[6]
                    + (*mutex)[2] * view_proj[10];
                if (dot - 1.0).abs() < 0.01 {
                    return;
                }
            }
        }

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; scene.splat_count];
        for i in 0..scene.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..scene.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..scene.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; scene.splat_count];
        for i in 0..scene.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        //////////////////////////////////
        // no cloning is happening for the single-consumer case
        let _ = bus.try_broadcast(depth_index);
        //////////////////////////////////

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            (*mutex)[0] = view_proj[2];
            (*mutex)[1] = view_proj[6];
            (*mutex)[2] = view_proj[10];
        }
    }

    pub fn merge(&mut self, scene: &Scene) {
        self.buffer.extend(scene.buffer.clone());
        self.splat_count = self.splat_count + scene.splat_count
    }

    pub fn translate(&mut self, offset: Vec3) {
        let row_length = 3 * 4 + 3 * 4 + 4 + 4; // 32bytes
        for i in 0..self.splat_count {
            let start = i * row_length;
            let end = start + 3 * 4;
            {
                // read 3x f32
                let position: &mut [f32] =
                    transmute_slice_mut::<_, f32>(&mut self.buffer[start..end]);
                position[0] += offset.x;
                position[1] += offset.y;
                position[2] += offset.z;
            }
        }
    }

    pub fn copy_from(&mut self, scene: &Scene) {
        self.splat_count = scene.splat_count;
        self.buffer = scene.buffer.clone();
        self.tex_data = scene.tex_data.clone();
        self.tex_width = scene.tex_width;
        self.tex_height = scene.tex_height;
    }

    pub fn compute_aabb_and_center(&self) -> ((Vec3, Vec3), Vec3) {
        let mut aabb: Option<(Vec3, Vec3)> = None;
        let mut avg_center = Vec3::zero();
        let row_length = 3 * 4 + 3 * 4 + 4 + 4; // 32bytes
        for i in 0..self.splat_count {
            let start = i * row_length;
            let end = start + 3 * 4;
            {
                // read 3x f32
                let position: &[f32] = transmute_slice::<_, f32>(&self.buffer[start..end]);
                let position = vec3(position[0], position[1], position[2]);
                avg_center += position;
                if let Some(aabb_ref) = aabb.as_mut() {
                    aabb_ref.0 = vec3(
                        aabb_ref.0.x.min(position.x),
                        aabb_ref.0.y.min(position.y),
                        aabb_ref.0.z.min(position.z),
                    );
                    aabb_ref.1 = vec3(
                        aabb_ref.1.x.max(position.x),
                        aabb_ref.1.y.max(position.y),
                        aabb_ref.1.z.max(position.z),
                    );
                } else {
                    aabb = Some((position, position));
                }
            }
        }
        avg_center /= self.splat_count as f32;

        (aabb.unwrap(), avg_center)
    }

    pub fn compute_scale_sum(&self) -> f32 {
        let mut scale_sum: f32 = 0.0;
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());
        for i in 0..self.splat_count {
            scale_sum += f_buffer[8 * i + 3];
            scale_sum += f_buffer[8 * i + 4];
            scale_sum += f_buffer[8 * i + 5];
        }

        scale_sum
    }
}
impl Clone for Scene {
    fn clone(&self) -> Self {
        Self {
            splat_count: self.splat_count,
            buffer: self.buffer.clone(),
            tex_data: self.tex_data.clone(),
            tex_width: self.tex_width,
            tex_height: self.tex_height,
            prev_vp: Mutex::new(Vec::<f32>::new()),
        }
    }
}

/// Loads a .ply or .splat file and returns a [Scene]
pub async fn load_scene() -> Scene {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */
    let mut scene = Scene::new();

    let file = rfd::AsyncFileDialog::new()
        .add_filter("3DGS model", &["ply", "splat", "spz"])
        .pick_file()
        .await;
    if let Some(f) = file.as_ref() {
        if f.file_name().contains(".ply") {
            let bytes = f.read().await;
            scene = match Scene::from_ply_bytes(bytes) {
                Ok(scene) => scene,
                Err(e) => {
                    log!("load_scene(): ERROR: {}", e);
                    unreachable!();
                }
            };
        } else if f.file_name().contains(".splat") {
            scene.buffer = f.read().await;
            scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
        } else {
            unreachable!();
        }
    }

    scene.generate_texture();

    log!("load_scene(): scene.splat_count={}", scene.splat_count);

    scene
}

/// Loads multiple .ply or .splat file and returns a [Vec<Vec<Scene>>] with shape [n_lod, n_tile]
pub async fn load_scene_vec() -> Vec<Vec<Scene>> {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */

    let file_vec = rfd::AsyncFileDialog::new()
        .set_title("Upload Tiles")
        .add_filter("3DGS model", &["ply", "splat", "spz"])
        .pick_files()
        .await;

    if file_vec.is_none() {
        return Vec::new();
    }

    let mut file_vec = file_vec.unwrap();
    file_vec.sort_by_key(|s| {
        let filename = s.file_name();
        parse_tile_filename(filename.as_str()).unwrap()
    });
    let first_filename = file_vec.first().unwrap().file_name();
    let first_nums = parse_tile_filename(first_filename.as_str()).unwrap();
    let last_filename = file_vec.last().unwrap().file_name();
    let last_nums = parse_tile_filename(last_filename.as_str()).unwrap();

    let n_lod = last_nums.0 as usize - first_nums.0 as usize + 1;
    let n_tile = last_nums.1 as usize + 1;

    let mut scene_vec: Vec<Vec<Scene>> = Vec::new();

    for i in 0..n_lod {
        let mut lod_vec: Vec<Scene> = Vec::new();
        for j in 0..n_tile {
            let f = &file_vec[i * n_tile + j];
            let mut scene = Scene::new();

            if f.file_name().contains(".ply") {
                let bytes = f.read().await;
                scene = match Scene::from_ply_bytes(bytes) {
                    Ok(scene) => scene,
                    Err(e) => {
                        log!("load_scene(): ERROR: {}", e);
                        unreachable!();
                    }
                };
            } else if f.file_name().contains(".splat") {
                scene.buffer = f.read().await;
                scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
            } else {
                unreachable!();
            }

            scene.generate_texture();

            log!("load_scene(): {}", f.file_name());
            log!("load_scene(): scene.splat_count={}", scene.splat_count);

            lod_vec.push(scene);
        }
        scene_vec.push(lod_vec);
    }

    scene_vec
}

pub async fn load_scene_zip() -> Vec<Vec<Scene>> {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */

    let file_zip = rfd::AsyncFileDialog::new()
        .set_title("Upload Tiles (.zip)")
        .add_filter("Tiles", &["zip"])
        .pick_file()
        .await;

    if file_zip.is_none() {
        return Vec::new();
    }
    let file_zip = file_zip.unwrap().read().await;
    let file_cursor = Cursor::new(file_zip);
    let mut archive = zip::ZipArchive::new(file_cursor).unwrap();

    // Extract zip
    struct SceneFileEntry {
        index: usize,
        filename: String,
        lod_id: usize,
        tile_id: usize,
    }
    let mut file_vec: Vec<SceneFileEntry> = Vec::new();
    for i in 0..archive.len() {
        let file = archive.by_index(i).unwrap();
        let filename = file
            .enclosed_name()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        if let Some((lod_id, tile_id)) = parse_tile_filename(filename.as_str()) {
            let entry = SceneFileEntry {
                index: i,
                filename,
                lod_id,
                tile_id,
            };
            file_vec.push(entry);
        }
    }

    file_vec.sort_by_key(|e| (e.lod_id, e.tile_id));
    let first_entry = file_vec.first().unwrap();
    let last_entry = file_vec.last().unwrap();

    let n_lod = last_entry.lod_id - first_entry.lod_id + 1;
    let n_tile = last_entry.tile_id as usize + 1;

    let mut scene_vec: Vec<Vec<Scene>> = Vec::with_capacity(n_lod);

    for i in 0..n_lod {
        let mut lod_vec: Vec<Scene> = Vec::with_capacity(n_tile);
        for j in 0..n_tile {
            let file_entry = &file_vec[i * n_tile + j];
            let mut scene = Scene::new();

            if file_entry.filename.contains(".ply") {
                let mut file = archive.by_index(file_entry.index).unwrap();
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read_exact(&mut bytes.as_mut_slice())
                    .expect(format!("Error loading file: {}", file_entry.filename).as_str());
                scene = match Scene::from_ply_bytes(bytes) {
                    Ok(scene) => scene,
                    Err(e) => {
                        log!("load_scene(): ERROR: {}", e);
                        unreachable!();
                    }
                };
            } else if file_entry.filename.contains(".splat") {
                let mut file = archive.by_index(file_entry.index).unwrap();
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read(&mut bytes.as_mut_slice())
                    .expect(format!("Error loading file: {}", file_entry.filename).as_str());
                scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
            } else {
                unreachable!();
            }

            // scene.generate_texture();

            log!("load_scene(): {}", file_entry.filename);
            log!("load_scene(): scene.splat_count={}", scene.splat_count);

            lod_vec.push(scene);
        }
        scene_vec.push(lod_vec);
    }

    scene_vec
}

/// Merges a vec of scenes into one
pub fn merge_scene(scene_vec: &Vec<Scene>) -> Scene {
    let mut new_scene = Scene::new();

    for scene in scene_vec {
        new_scene.merge(scene);
    }

    new_scene.generate_texture();

    log!(
        "merge_scene(): new_scene.splat_count={}",
        new_scene.splat_count
    );

    new_scene
}

pub fn translate_scene(scene: &Scene, offset: Vec3, gen_tex: bool) -> Scene {
    let mut new_scene = scene.clone();

    new_scene.translate(offset);

    if gen_tex {
        new_scene.generate_texture();
    }

    new_scene
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binary_ply(properties: &[&str], values: &[f32]) -> Vec<u8> {
        assert_eq!(properties.len(), values.len());

        let mut bytes = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n{}end_header\n",
            properties
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        )
        .into_bytes();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn constructor_properties() -> [&'static str; 14] {
        [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
    }

    fn constructor_values() -> [f32; 14] {
        [
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn parses_constructor_tile_filename() {
        assert_eq!(parse_tile_filename("tile15_lod5.ply"), Some((5, 15)));
    }

    #[test]
    fn preserves_legacy_tile_filename_support() {
        assert_eq!(parse_tile_filename("lod5_tile_15.ply"), Some((5, 15)));
    }

    #[test]
    fn rejects_partial_tile_filename_matches() {
        assert_eq!(parse_tile_filename("backup_tile15_lod5.ply.tmp"), None);
    }

    #[test]
    fn loads_constructor_ply_without_normals() {
        let scene = Scene::from_ply_bytes(make_binary_ply(
            &constructor_properties(),
            &constructor_values(),
        ))
        .unwrap();

        assert_eq!(scene.splat_count, 1);
        assert_eq!(read_f32(&scene.buffer, 0), 1.0);
        assert_eq!(read_f32(&scene.buffer, 4), 2.0);
        assert_eq!(read_f32(&scene.buffer, 8), 3.0);
        assert_eq!(read_f32(&scene.buffer, 12), 1.0);
        assert_eq!(read_f32(&scene.buffer, 16), 1.0);
        assert_eq!(read_f32(&scene.buffer, 20), 1.0);
        assert_eq!(&scene.buffer[24..28], &[127, 127, 127, 127]);
        assert_eq!(&scene.buffer[28..32], &[255, 127, 127, 127]);
    }

    #[test]
    fn ignores_optional_normals_and_spherical_harmonics() {
        let properties = [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let values = [
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0,
        ];

        let scene = Scene::from_ply_bytes(make_binary_ply(&properties, &values)).unwrap();

        assert_eq!(scene.splat_count, 1);
        assert_eq!(read_f32(&scene.buffer, 0), 1.0);
        assert_eq!(read_f32(&scene.buffer, 12), 1.0);
        assert_eq!(&scene.buffer[24..28], &[127, 127, 127, 127]);
        assert_eq!(&scene.buffer[28..32], &[255, 127, 127, 127]);
    }

    #[test]
    fn rejects_ply_missing_required_property() {
        let properties = constructor_properties();
        let values = constructor_values();
        let opacity_index = properties
            .iter()
            .position(|name| *name == "opacity")
            .unwrap();
        let properties = properties
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| (index != opacity_index).then_some(name))
            .collect::<Vec<_>>();
        let values = values
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| (index != opacity_index).then_some(value))
            .collect::<Vec<_>>();

        let error = match Scene::from_ply_bytes(make_binary_ply(&properties, &values)) {
            Ok(_) => panic!("expected missing opacity to fail"),
            Err(error) => error,
        };

        assert!(error.contains("opacity"), "{error}");
    }
}
