// Must match cubed_sphere.rs. Oriented face frames preserve the source-pattern
// direction of rotate_tile=true Wang assets across all twelve cube seams.
fn cube_basis(face: u32) -> mat3x3<f32> {
    let x=vec3(1.0,0.0,0.0); let y=vec3(0.0,1.0,0.0); let z=vec3(0.0,0.0,1.0);
    switch face {
        case 0u: { return mat3x3(x,y,z); }
        case 1u: { return mat3x3(-x,-z,-y); }
        case 2u: { return mat3x3(y,-x,z); }
        case 3u: { return mat3x3(-y,-z,x); }
        case 4u: { return mat3x3(z,-x,-y); }
        default: { return mat3x3(-z,y,x); }
    }
}

fn cube_project(s: f32, rate: f32) -> vec2<f32> {
    let angle=s*rate-0.7853981633974483;
    let limited=clamp(angle,-0.7853981633974483,0.7853981633974483);
    let tangent=tan(limited);
    let slope=1.0+tangent*tangent;
    return vec2(tangent+(angle-limited)*slope,slope*rate);
}

struct CubeMapping {
    position: vec3<f32>,
    jacobian: mat3x3<f32>,
}

fn cube_map(face: u32, position: vec3<f32>, n: u32, width: f32, radius: f32) -> CubeMapping {
    let rate=1.5707963267948966/(f32(n)*width);
    let a=cube_project(position.x,rate); let b=cube_project(position.y,rate);
    let basis=cube_basis(face);
    let q=basis[0]+basis[1]*a.x+basis[2]*b.x;
    let inv_length=inverseSqrt(dot(q,q)); let radial=q*inv_length;
    let r=radius+position.z;
    let dx=(basis[1]-radial*dot(radial,basis[1]))*(r*inv_length*a.y);
    let dy=(basis[2]-radial*dot(radial,basis[2]))*(r*inv_length*b.y);
    return CubeMapping(radial*r,mat3x3(dx,dy,radial));
}
