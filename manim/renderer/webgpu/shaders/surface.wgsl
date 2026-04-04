// WebGPU surface shader for Manim — Phase 3.
//
// Renders flat-shaded triangulated Surface faces (shade_in_3d=True VMobjects).
// Each vertex carries its world-space position, the face normal, and the fill
// colour.  A Phong diffuse + ambient lighting model is applied in the fragment
// shader using the light_pos uniform.
//
// Depth test is enabled so that 3-D surfaces occlude each other correctly.
//
// Uniform layout (group 0, binding 0) — shared with fill/stroke:
//   offset   0 — projection  mat4x4<f32>   (64 bytes)
//   offset  64 — view        mat4x4<f32>   (64 bytes)
//   offset 128 — light_pos   vec3<f32>     (12 bytes, padded to 16)
//
// Vertex attributes:
//   location 0 — in_vert     vec3<f32>   world-space position
//   location 1 — in_normal   vec3<f32>   world-space face normal
//   location 2 — in_color    vec4<f32>   RGBA fill colour

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    light_pos  : vec3<f32>,
    _pad       : f32,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) in_vert   : vec3<f32>,
    @location(1) in_normal : vec3<f32>,
    @location(2) in_color  : vec4<f32>,
};

struct VertexOutput {
    @builtin(position)              clip_position : vec4<f32>,
    @location(0)                    v_color       : vec4<f32>,
    @location(1)                    v_normal      : vec3<f32>,
    @location(2)                    v_world_pos   : vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos      = vec4<f32>(in.in_vert, 1.0);
    out.clip_position  = u.projection * u.view * world_pos;
    out.v_world_pos    = in.in_vert;
    out.v_normal       = in.in_normal;
    out.v_color        = in.in_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient_strength  = 0.3;
    let diffuse_strength  = 0.7;

    let norm       = normalize(in.v_normal);
    let light_dir  = normalize(u.light_pos - in.v_world_pos);

    // Two-sided lighting: use abs so back-faces aren't fully dark.
    let diff       = abs(dot(norm, light_dir));

    let lighting   = ambient_strength + diffuse_strength * diff;
    let lit_rgb    = clamp(in.v_color.rgb * lighting, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(lit_rgb, in.v_color.a);
}
