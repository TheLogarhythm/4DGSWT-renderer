//! Equiangular cubed sphere with consistently oriented rotated-edge Wang tiles.
use crate::utils::*;

pub(crate) const DEFAULT_FACE_TILES: usize = 8;
pub(crate) const MAX_FACE_TILES: usize = 128;

/// Face axes already include the toy experiment's [0,270,0,270,180,270]
/// degree rotations. Asset coordinates and W/N/E/S labels need no further rotation.
pub(crate) fn basis(face: usize) -> (Vec3, Vec3, Vec3) {
    let x = Vec3::unit_x();
    let y = Vec3::unit_y();
    let z = Vec3::unit_z();
    match face {
        0 => (x, y, z),
        1 => (-x, -z, -y),
        2 => (y, -x, z),
        3 => (-y, -z, x),
        4 => (z, -x, -y),
        5 => (-z, y, x),
        _ => unreachable!("cubed sphere has six faces"),
    }
}

pub(crate) fn neighbor(coord: Vector2<usize>, side: usize, n: usize) -> (Vector2<usize>, usize) {
    let face = coord.x / n;
    let i = coord.x % n;
    let j = coord.y;
    match side {
        0 if i > 0 => return (coord - vec2(1, 0), 2),
        1 if j + 1 < n => return (coord + vec2(0, 1), 3),
        2 if i + 1 < n => return (coord + vec2(1, 0), 0),
        3 if j > 0 => return (coord - vec2(0, 1), 1),
        _ => (),
    }
    let (normal, u, v) = basis(face);
    let next_normal = [-u, v, u, -v][side];
    let next_face = (0..6).find(|&f| basis(f).0 == next_normal).unwrap();
    let (_, next_u, next_v) = basis(next_face);
    // Exact integer-valued midpoint coordinates; no angular rounding at seams.
    let edge_xy = [
        vec2(0., 2. * j as f32 + 1.),
        vec2(2. * i as f32 + 1., 2. * n as f32),
        vec2(2. * n as f32, 2. * j as f32 + 1.),
        vec2(2. * i as f32 + 1., 0.),
    ][side];
    let nf = n as f32;
    let midpoint = normal * nf + u * (edge_xy.x - nf) + v * (edge_xy.y - nf);
    let x = midpoint.dot(next_u);
    let y = midpoint.dot(next_v);
    let next_side = if x == -nf {
        0
    } else if y == nf {
        1
    } else if x == nf {
        2
    } else {
        3
    };
    let ni = (((x + nf) * 0.5) as usize).min(n - 1);
    let nj = (((y + nf) * 0.5) as usize).min(n - 1);
    (vec2(next_face * n + ni, nj), next_side)
}

pub(crate) fn map(face: usize, position: Vec3, n: usize, width: f32, radius: f32) -> (Vec3, Mat3) {
    let rate = std::f32::consts::FRAC_PI_2 / (n as f32 * width);
    let project = |s: f32| {
        let angle = s * rate - std::f32::consts::FRAC_PI_4;
        // C1 linear continuation outside a face avoids tan poles for animated
        // splats whose centers/support extend beyond their owning tile.
        let limited = angle.clamp(-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);
        let tangent = limited.tan();
        let slope = 1. + tangent * tangent;
        (tangent + (angle - limited) * slope, slope * rate)
    };
    let (a, da) = project(position.x);
    let (b, db) = project(position.y);
    let (normal, u, v) = basis(face);
    let q = normal + u * a + v * b;
    let inv_length = 1. / q.magnitude();
    let radial = q * inv_length;
    let r = radius + position.z;
    let dx = (u - radial * radial.dot(u)) * (r * inv_length * da);
    let dy = (v - radial * radial.dot(v)) * (r * inv_length * db);
    (radial * r, Mat3::from_cols(dx, dy, radial))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Increasing source coordinate: W/E go up, N/S go left, as in the
    // constructor's rotate_tile=true placement (not simply local edge order).
    fn edge_position(coord: Vector2<usize>, side: usize, q: f32, n: usize) -> Vec3 {
        let (x, y) = match side {
            0 => (0., q),
            1 => (1. - q, 1.),
            2 => (1., q),
            3 => (1. - q, 0.),
            _ => unreachable!(),
        };
        vec3((coord.x % n) as f32 + x, coord.y as f32 + y, 0.7)
    }

    #[test]
    fn cubed_sphere_neighbors_close_for_single_odd_and_even_grids() {
        for n in [1, 3, 8] {
            let mut edges = HashSet::new();
            let mut seam_edges = HashSet::new();
            for x in 0..6 * n {
                for y in 0..n {
                    for side in 0..4 {
                        let coord = vec2(x, y);
                        let (other, other_side) = neighbor(coord, side, n);
                        assert!(other.x < 6 * n && other.y < n);
                        assert_eq!(neighbor(other, other_side, n), (coord, side));
                        let a = (x * n + y, side);
                        let b = (other.x * n + other.y, other_side);
                        let key = if a < b { (a, b) } else { (b, a) };
                        edges.insert(key);
                        if x / n != other.x / n {
                            seam_edges.insert(key);
                        }
                        for q in [0., 0.13, 0.5, 0.79, 1.] {
                            let (pa, _) = map(x / n, edge_position(coord, side, q, n), n, 1., 17.);
                            let (pb, _) = map(
                                other.x / n,
                                edge_position(other, other_side, q, n),
                                n,
                                1.,
                                17.,
                            );
                            assert!(
                                (pa - pb).magnitude() < 5e-5,
                                "N={n}, {coord:?}:{side} -> {other:?}:{other_side}, q={q}"
                            );
                        }
                    }
                }
            }
            assert_eq!(edges.len(), 12 * n * n);
            assert_eq!(seam_edges.len(), 12 * n);
        }
    }

    #[test]
    fn cubed_sphere_jacobian_matches_height_and_tangential_motion() {
        // Double-precision finite differences avoid f32 cancellation and the
        // O(step) error where equiangular mapping meets its C1 continuation.
        let reference = |face: usize, p: Vec3, axis: Vec3, step: f64, n: usize| {
            let (normal, u, v) = basis(face);
            let point = [
                p.x as f64 + axis.x as f64 * step,
                p.y as f64 + axis.y as f64 * step,
                p.z as f64 + axis.z as f64 * step,
            ];
            let project = |s: f64| {
                let a =
                    s * std::f64::consts::FRAC_PI_2 / (n as f64 * 4.) - std::f64::consts::FRAC_PI_4;
                let b = a.clamp(-std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_4);
                b.tan() + (a - b) * (1. + b.tan().powi(2))
            };
            let q = normal.cast::<f64>().unwrap()
                + u.cast::<f64>().unwrap() * project(point[0])
                + v.cast::<f64>().unwrap() * project(point[1]);
            q.normalize() * (20. + point[2])
        };
        for face in 0..6 {
            for n in [1, 3, 8] {
                for uv in [vec2(0., 0.), vec2(0.3, 0.6), vec2(1., 1.)] {
                    let p = vec3(uv.x * n as f32 * 4., uv.y * n as f32 * 4., 1.3);
                    let (mapped, j) = map(face, p, n, 4., 20.);
                    assert!((mapped.magnitude() - 21.3).abs() < 1e-5);
                    assert!(j.determinant() > 0.);
                    for axis in [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()] {
                        let delta = 1e-5;
                        let derivative = ((reference(face, p, axis, delta, n)
                            - reference(face, p, axis, -delta, n))
                            / (2. * delta))
                            .cast::<f32>()
                            .unwrap();
                        assert!(
                            (derivative - j * axis).magnitude() < 1e-4,
                            "face={face} N={n} p={p:?} numeric={derivative:?} analytic={:?}",
                            j * axis
                        );
                    }
                }
            }
        }
    }
}
