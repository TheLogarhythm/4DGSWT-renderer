use winit::dpi::PhysicalSize;

use crate::utils::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRay {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl WorldRay {
    pub fn intersect_z_plane(&self, z: f32) -> Option<Vec3> {
        if !z.is_finite() || self.direction.z.abs() <= 1e-6 {
            return None;
        }
        let distance = (z - self.origin.z) / self.direction.z;
        if !distance.is_finite() || distance < 0.0 {
            return None;
        }
        Some(self.origin + self.direction * distance)
    }
}

// From three_d
pub struct Camera {
    viewport: PhysicalSize<u32>,
    position: Vec3,
    target: Vec3,
    up: Vec3,
    fovy: Radians,
    z_near: f32,
    z_far: f32,

    view: Mat4,
    projection: Mat4,
}
impl Camera {
    pub fn new(viewport: PhysicalSize<u32>) -> Self {
        Self {
            viewport,
            position: Vec3::zero(),
            target: Vec3::zero(),
            up: Vec3::zero(),
            fovy: radians(0.0),
            z_near: 0.0,
            z_far: 0.0,

            view: Mat4::zero(),
            projection: Mat4::zero(),
        }
    }

    pub fn new_perspective(
        viewport: PhysicalSize<u32>,
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fovy: impl Into<Radians>,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let mut cam = Self::new(viewport);

        cam.set_view(position, target, up);
        cam.set_perspective_projection(fovy, z_near, z_far);
        cam
    }

    pub fn viewport(&self) -> &PhysicalSize<u32> {
        &self.viewport
    }

    pub fn position(&self) -> &Vec3 {
        &self.position
    }

    pub fn target(&self) -> &Vec3 {
        &self.target
    }

    pub fn up(&self) -> &Vec3 {
        &self.up
    }

    pub fn fovy(&self) -> Radians {
        self.fovy
    }

    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    pub fn right_direction(&self) -> Vec3 {
        self.view_direction().cross(self.up)
    }

    pub fn view(&self) -> &Mat4 {
        &self.view
    }

    pub fn projection(&self) -> &Mat4 {
        &self.projection
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection * self.view
    }

    pub fn world_ray(&self, pixel: [f32; 2]) -> Result<WorldRay, String> {
        if self.viewport.width == 0 || self.viewport.height == 0 {
            return Err("camera viewport must be nonzero".to_string());
        }
        if pixel.into_iter().any(|coordinate| !coordinate.is_finite()) {
            return Err("camera ray pixel must be finite".to_string());
        }
        let ndc = vec4(
            2.0 * pixel[0] / self.viewport.width as f32 - 1.0,
            1.0 - 2.0 * pixel[1] / self.viewport.height as f32,
            1.0,
            1.0,
        );
        let inverse = self
            .view_proj()
            .invert()
            .ok_or_else(|| "camera view-projection matrix is singular".to_string())?;
        let world_h = inverse * ndc;
        if !world_h.w.is_finite() || world_h.w.abs() <= f32::EPSILON {
            return Err("camera ray unprojection produced an invalid point".to_string());
        }
        let world = world_h.truncate() / world_h.w;
        let delta = world - self.position;
        if !delta.magnitude2().is_finite() || delta.magnitude2() <= f32::EPSILON {
            return Err("camera ray direction is degenerate".to_string());
        }
        Ok(WorldRay {
            origin: self.position,
            direction: delta.normalize(),
        })
    }

    pub fn set_view(&mut self, position: Vec3, target: Vec3, up: Vec3) {
        self.position = position;
        self.target = target;
        self.up = up;
        self.view = Mat4::look_at_rh(
            Point3::from_vec(self.position),
            Point3::from_vec(self.target),
            self.up,
        );
    }

    pub fn set_perspective_projection(
        &mut self,
        fovy: impl Into<Radians>,
        z_near: f32,
        z_far: f32,
    ) {
        assert!(
            z_near >= 0.0 || z_near < z_far,
            "Wrong perspective camera parameters"
        );
        self.fovy = fovy.into();
        self.z_near = z_near;
        self.z_far = z_far;

        self.projection = cgmath::perspective(
            self.fovy,
            self.viewport.width as f32 / self.viewport.height as f32,
            z_near,
            z_far,
        );
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = PhysicalSize { width, height };
        self.projection = cgmath::perspective(
            self.fovy,
            width as f32 / height as f32,
            self.z_near,
            self.z_far,
        );
    }

    pub fn translate(&mut self, change: &Vec3) {
        self.set_view(self.position + change, self.target + change, self.up);
    }

    pub fn pitch(&mut self, delta: impl Into<Radians>) {
        let target = (self.view.invert().unwrap()
            * Mat4::from_angle_x(delta)
            * self.view
            * self.target.extend(1.0))
        .truncate();
        if (target - self.position).normalize().dot(self.up).abs() < 0.999 {
            self.set_view(self.position, target, self.up);
        }
    }

    pub fn yaw(&mut self, delta: impl Into<Radians>) {
        let target = (self.view.invert().unwrap()
            * Mat4::from_angle_y(delta)
            * self.view
            * self.target.extend(1.0))
        .truncate();
        self.set_view(self.position, target, self.up);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub focal: [f32; 2],
    pub viewport: [f32; 2],
    pub htan_fov: [f32; 4],
    pub cam_pos: [f32; 4],
}
impl CameraUniforms {
    pub fn from_camera(cam: &Camera) -> Self {
        let view_matrix: &Mat4 = cam.view();
        let projection_matrix: &Mat4 = cam.projection();
        let w = cam.viewport().width as f32;
        let h = cam.viewport().height as f32;
        let cam_pos = cam.position();
        let fx = 0.5 * projection_matrix[0][0] * w;
        let fy = -0.5 * projection_matrix[1][1] * h;
        let htany = (cam.fovy() / 2.0).tan() as f32;
        let htanx = (htany / h) * w;

        Self {
            projection: (*projection_matrix).into(),
            view: (*view_matrix).into(),
            focal: [fx.abs(), fy.abs()],
            viewport: [w, h],
            htan_fov: [htanx, htany, 0.0, 0.0],
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_center_ray_points_at_the_camera_target() {
        let camera = Camera::new_perspective(
            PhysicalSize::new(100, 100),
            vec3(0.0, 0.0, 10.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            degrees(90.0),
            0.1,
            100.0,
        );

        let ray = camera.world_ray([50.0, 50.0]).unwrap();
        assert!((ray.origin - vec3(0.0, 0.0, 10.0)).magnitude() < 1e-5);
        assert!((ray.direction - vec3(0.0, 0.0, -1.0)).magnitude() < 1e-5);
        let hit = ray.intersect_z_plane(0.0).unwrap();
        assert!(hit.magnitude() < 1e-4);
    }

    #[test]
    fn invalid_pixels_and_parallel_ground_hits_are_rejected() {
        let camera = Camera::new_perspective(
            PhysicalSize::new(100, 100),
            vec3(0.0, 0.0, 10.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            degrees(60.0),
            0.1,
            100.0,
        );
        assert!(camera.world_ray([f32::NAN, 5.0]).is_err());
        let parallel = WorldRay {
            origin: vec3(0.0, 0.0, 1.0),
            direction: vec3(1.0, 0.0, 0.0),
        };
        assert!(parallel.intersect_z_plane(0.0).is_none());
    }
}
