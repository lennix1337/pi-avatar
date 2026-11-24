struct FrameUniforms {
    view_proj: mat4x4<f32>,
    srgb_encode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MaterialUniforms {
    base_color: vec4<f32>,
    shade_color: vec4<f32>,
    emissive: vec4<f32>,
    shading_toony: f32,
    shading_shift: f32,
    has_emissive_tex: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> frame_u: FrameUniforms;
@group(0) @binding(1) var<storage, read> joints: array<mat4x4<f32>>;

@group(1) @binding(0) var<uniform> mat_u: MaterialUniforms;
@group(1) @binding(1) var base_tex: texture_2d<f32>;
@group(1) @binding(2) var emissive_tex: texture_2d<f32>;
@group(1) @binding(3) var tex_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let skin_mat =
        joints[v.joints.x] * v.weights.x +
        joints[v.joints.y] * v.weights.y +
        joints[v.joints.z] * v.weights.z +
        joints[v.joints.w] * v.weights.w;

    let world_pos = skin_mat * vec4<f32>(v.position, 1.0);
    let n_mat = mat3x3<f32>(skin_mat[0].xyz, skin_mat[1].xyz, skin_mat[2].xyz);
    let world_n = normalize(n_mat * v.normal);

    var out: VsOut;
    out.clip_pos = frame_u.view_proj * world_pos;
    out.world_normal = world_n;
    out.uv = v.uv;
    return out;
}

fn linear_to_srgb_channel(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return x * 12.92;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_channel(rgb.r),
        linear_to_srgb_channel(rgb.g),
        linear_to_srgb_channel(rgb.b),
    );
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if (!front_facing) {
        n = -n;
    }

    let base_sample = textureSample(base_tex, tex_sampler, in.uv);
    let lit_color = base_sample.rgb * mat_u.base_color.rgb;
    let shade_color = lit_color * mat_u.shade_color.rgb;

    // Match studio lighting from labs/avatar/avatar.js (key + fill + ambient)
    let key_dir = normalize(vec3<f32>(1.2, 2.2, 2.0));
    let fill_dir = normalize(vec3<f32>(-1.5, 1.2, 1.0));

    let ndl_key = dot(n, key_dir);
    let feather = max(0.001, 1.0 - mat_u.shading_toony);
    let toon_key = smoothstep(
        mat_u.shading_shift - feather,
        mat_u.shading_shift + feather,
        ndl_key,
    );

    let ndl_fill = clamp(dot(n, fill_dir) * 0.5 + 0.5, 0.0, 1.0);

    // MToon-style blend between shade and lit with gentle key/fill/ambient illumination
    let direct_color = mix(shade_color, lit_color, toon_key);
    let light_energy =
        vec3<f32>(0.78, 0.78, 0.80) +
        vec3<f32>(0.24, 0.24, 0.24) * toon_key +
        vec3<f32>(0.08, 0.10, 0.13) * ndl_fill;

    var color = direct_color * light_energy;

    if (mat_u.has_emissive_tex != 0u) {
        let em_sample = textureSample(emissive_tex, tex_sampler, in.uv);
        color = color + em_sample.rgb * mat_u.emissive.rgb;
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    if (frame_u.srgb_encode != 0u) {
        color = linear_to_srgb(color);
    }
    return vec4<f32>(color, 1.0);
}
