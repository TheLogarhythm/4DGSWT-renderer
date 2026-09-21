// Pixel-addressed, full precision; never filter across water/land or miss boundaries.
@group(WATER_HIT_GROUP) @binding(0) var water_hits: texture_2d<f32>;
fn load_water_hit(pixel: vec2<f32>) -> vec4<f32> {
    let hit = textureLoad(water_hits,vec2<i32>(pixel),0).xy;
    return vec4(hit.x,select(0.0,1.0,hit.y>0.0),0.0,hit.y);
}
