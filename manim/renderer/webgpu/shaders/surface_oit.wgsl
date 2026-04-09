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
// Vertex layout matches surface_combined.wgsl (stride 80 bytes):
//   location 0 — in_vert           float32x3  offset  0
//   location 1 — in_normal         float32x3  offset 12
//   location 2 — in_fill_color     float32x4  offset 24
//   location 3 — in_stroke_color   float32x4  offset 40
//   location 4 — in_bary           float32x3  offset 56
//   location 5 — stroke_half_px    float32    offset 68
//   location 6 — diffuse_strength  float32    offset 72
//   location 7 — specular_strength float32    offset 76

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
    @location(0) in_vert           : vec3<f32>,
    @location(1) in_normal         : vec3<f32>,
    @location(2) in_fill_color     : vec4<f32>,
    @location(3) in_stroke_color   : vec4<f32>,
    @location(4) in_bary           : vec3<f32>,
    @location(5) stroke_half_px    : f32,
    @location(6) diffuse_strength  : f32,
    @location(7) specular_strength : f32,
};

struct VertexOutput {
    @builtin(position)              clip_position  : vec4<f32>,
    @location(0)                    v_fill_color   : vec4<f32>,
    @location(1)                    v_stroke_color : vec4<f32>,
    @location(2)                    v_view_normal  : vec3<f32>,
    @location(3)                    v_view_pos     : vec3<f32>,
    @location(4)                    v_view_light   : vec3<f32>,
    @location(5)                    v_bary         : vec3<f32>,
    @location(6) @interpolate(flat) v_stroke_half  : f32,
    @location(7) @interpolate(flat) v_diffuse      : f32,
    @location(8) @interpolate(flat) v_specular     : f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let view_pos       = u.view * vec4<f32>(in.in_vert, 1.0);
    out.clip_position  = u.projection * view_pos;
    out.v_view_pos     = view_pos.xyz;
    let view3          = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.v_view_normal  = view3 * in.in_normal;
    out.v_view_light   = (u.view * vec4<f32>(u.light_pos, 1.0)).xyz;
    out.v_fill_color   = in.in_fill_color;
    out.v_stroke_color = in.in_stroke_color;
    out.v_bary         = in.in_bary;
    out.v_stroke_half  = in.stroke_half_px;
    out.v_diffuse      = in.diffuse_strength;
    out.v_specular     = in.specular_strength;
    return out;
}

struct FragOutput {
    @location(0) accum  : vec4<f32>,  // weighted colour sum   → rgba16float
    @location(1) reveal : vec4<f32>,  // transmittance product → rgba16float
};

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> FragOutput {
    let diffuse_strength  = in.v_diffuse;
    let specular_strength = in.v_specular;
    let specular_exp      = 16.0;

    // Two-sided lighting: flip normal for back-facing fragments.
    let raw_normal  = select(-in.v_view_normal, in.v_view_normal, front_facing);
    let norm        = normalize(raw_normal);
    let light_dir_v = in.v_view_light - in.v_view_pos;
    let light_dir   = normalize(light_dir_v);
    let view_dir    = normalize(-in.v_view_pos);
    let half_vec    = normalize(light_dir + view_dir);

    let diff        = clamp(dot(norm, light_dir), 0.0, 1.0);
    let spec        = pow(max(dot(norm, half_vec), 0.0), specular_exp);
    let attenuation = u.light_intensity / dot(light_dir_v, light_dir_v);

    let ambient_rgb  = in.v_fill_color.rgb * u.ambient_color  * u.ambient_intensity;
    let diffuse_rgb  = in.v_fill_color.rgb * u.light_color * (diffuse_strength  * diff * attenuation);
    let specular_rgb = u.light_color                       * (specular_strength * spec * attenuation);
    let lit_rgb      = clamp(ambient_rgb + diffuse_rgb + specular_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let fill_a       = in.v_fill_color.a;

    // ── Barycentric wireframe ──────────────────────────────────────────────
    let edge_dist_px = in.v_bary.x / max(fwidth(in.v_bary.x), 1e-6);
    let stroke_cov   = clamp(in.v_stroke_half + 0.5 - edge_dist_px, 0.0, 1.0);
    let stroke_a     = in.v_stroke_color.a * stroke_cov;

    // ── Porter-Duff "over": stroke on top of fill ─────────────────────────
    let total_a = stroke_a + fill_a * (1.0 - stroke_a);
    if total_a <= 0.001 { discard; }
    let out_rgb = (stroke_a * in.v_stroke_color.rgb + fill_a * (1.0 - stroke_a) * lit_rgb) / total_a;

    // ── Weighted Blended OIT ───────────────────────────────────────────────
    let z = in.v_view_pos.z;
    let w = clamp(
        pow(total_a, 3.0) / (1e-5 + pow(abs(z) / 5.0, 4.0)),
        1e-2, 3e3
    );

    var out: FragOutput;
    out.accum  = vec4<f32>(out_rgb * total_a * w, total_a * w);
    out.reveal = vec4<f32>(total_a, total_a, total_a, total_a);
    return out;
}
