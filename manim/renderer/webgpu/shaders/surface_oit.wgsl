// WebGPU OIT accumulation shader — Weighted Blended Order-Independent Transparency.
//
// McGuire & Bavoil 2013.  Renders transparent surface fragments into two
// accumulation targets instead of the main framebuffer:
//
//   location 0  accum   rgba16float   weighted colour + alpha sum
//   location 1  reveal  rgba16float   per-channel transmittance product
//
// The main pipeline blend modes for these targets are set to:
//   accum:   {src: one,  dst: one}            — additive accumulation
//   reveal:  {src: zero, dst: one-minus-src-alpha} — transmittance multiplication
//
// A subsequent full-screen composition pass reads both textures and composites
// the result onto the opaque framebuffer.
//
// Lighting model and uniform layout are identical to surface.wgsl so that
// opaque and transparent surfaces are lit consistently.
//
// Uniform layout (group 0, binding 0):
//   offset   0 — projection        mat4x4<f32>   (64 bytes)
//   offset  64 — view              mat4x4<f32>   (64 bytes)
//   offset 128 — light_pos         vec3<f32>     (12 bytes)
//   offset 140 — light_intensity   f32           ( 4 bytes)
//   offset 144 — light_color       vec3<f32>     (12 bytes)
//   offset 156 — ambient_intensity f32           ( 4 bytes)
//   offset 160 — ambient_color     vec3<f32>     (12 bytes)
//   offset 172 — _pad              f32           ( 4 bytes)
//   total: 176 bytes

struct Uniforms {
    projection        : mat4x4<f32>,
    view              : mat4x4<f32>,
    light_pos         : vec3<f32>,
    light_intensity   : f32,
    light_color       : vec3<f32>,
    ambient_intensity : f32,
    ambient_color     : vec3<f32>,
    _pad              : f32,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) in_vert   : vec3<f32>,
    @location(1) in_normal : vec3<f32>,
    @location(2) in_color  : vec4<f32>,
};

struct VertexOutput {
    @builtin(position)  clip_position : vec4<f32>,
    @location(0)        v_color       : vec4<f32>,
    @location(1)        v_view_normal : vec3<f32>,
    @location(2)        v_view_pos    : vec3<f32>,
    @location(3)        v_view_light  : vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let view_pos          = u.view * vec4<f32>(in.in_vert, 1.0);
    out.clip_position     = u.projection * view_pos;
    out.v_view_pos        = view_pos.xyz;
    let view3             = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.v_view_normal     = view3 * in.in_normal;
    out.v_view_light      = (u.view * vec4<f32>(u.light_pos, 1.0)).xyz;
    out.v_color           = in.in_color;
    return out;
}

struct FragOutput {
    @location(0) accum  : vec4<f32>,  // weighted colour sum   → rgba16float
    @location(1) reveal : vec4<f32>,  // transmittance product → rgba16float
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragOutput {
    // Per-material diffuse and specular strengths — identical to surface.wgsl.
    // Will be replaced by per-surface gloss/shadow when LightSource system lands.
    let diffuse_strength  = 0.9;
    let specular_strength = 0.8;
    let specular_exp      = 16.0;

    // Two-sided lighting: flip the normal for back-facing fragments so that
    // both sides of open surfaces are correctly lit from either direction.
    // Transparent surfaces (cull_mode="none") commonly show both sides — a
    // semi-transparent sphere's inner hemisphere is visible through the front.
    let raw_normal = select(-in.v_view_normal, in.v_view_normal, front_facing);
    let norm       = normalize(raw_normal);
    let light_dir_vec    = in.v_view_light - in.v_view_pos;
    let light_distance2  = dot(light_dir_vec, light_dir_vec);
    let light_dir        = normalize(light_dir_vec);
    let view_dir         = normalize(-in.v_view_pos);

    let diff     = clamp(dot(norm, light_dir), 0.0, 1.0);
    let half_vec = normalize(light_dir + view_dir);
    let spec     = pow(max(dot(norm, half_vec), 0.0), specular_exp);

    // Identical lighting formula to surface.wgsl.
    let attenuation = u.light_intensity / light_distance2;

    let ambient_rgb  = in.v_color.rgb * u.ambient_color * u.ambient_intensity;
    let diffuse_rgb  = in.v_color.rgb * u.light_color * (diffuse_strength  * diff * attenuation);
    let specular_rgb = u.light_color              * (specular_strength * spec * attenuation);

    let rgb   = clamp(ambient_rgb + diffuse_rgb + specular_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let alpha = in.v_color.a;

    // Depth-based weight that balances contributions from front and back layers.
    let z = in.v_view_pos.z;
    let w = clamp(
        pow(alpha, 3.0) / (1e-5 + pow(abs(z) / 5.0, 4.0)),
        1e-2, 3e3
    );

    var out: FragOutput;
    // accum: additive blend (pipeline: src=one, dst=one)
    out.accum  = vec4<f32>(rgb * alpha * w, alpha * w);
    // reveal: multiplicative blend (pipeline: src=zero, dst=one-minus-src-alpha)
    out.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
    return out;
}
