// High-quality 4-tap supersample downsample blit from 2x offscreen target into the final window target.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blit(@builtin(vertex_index) idx: u32) -> VsOut {
    let x = f32(i32(idx & 1u) * 4 - 1);
    let y = f32(i32((idx >> 1u) & 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_blit(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 0.5 / max(dims, vec2<f32>(1.0));
    // 4-tap rotated-grid box filter over the 2x supersampled source
    let c0 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-texel.x, -texel.y));
    let c1 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>( texel.x, -texel.y));
    let c2 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-texel.x,  texel.y));
    let c3 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>( texel.x,  texel.y));
    return (c0 + c1 + c2 + c3) * 0.25;
}
