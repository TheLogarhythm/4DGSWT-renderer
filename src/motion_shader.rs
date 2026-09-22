//! Shared WGSL is composed once at pipeline creation, never per frame.

pub fn source(body: &str) -> String {
    [include_str!("motion_math.wgsl"), "\n", body].concat()
}
