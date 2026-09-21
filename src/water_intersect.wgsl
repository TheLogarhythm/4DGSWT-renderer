@group(1) @binding(0) var hits_out: texture_storage_2d<rg32float,write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(hits_out)) { return; }
    let hit = water_surface_hit(vec2<f32>(id.xy)+vec2(0.5),water.camera,water.level.x,water.bounds,water.waves,water.phases);
    // Zero view depth marks a miss. Every pixel is overwritten, including misses.
    textureStore(hits_out,vec2<i32>(id.xy),vec4(hit.x,select(0.0,hit.w,hit.y>0.5),0.0,0.0));
}
