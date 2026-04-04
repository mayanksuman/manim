// WebGPU fill shader for VMobject — Phase 2.
//
// Uses the Loop-Blinn quadratic bezier test for smooth, anti-aliased fill
// boundaries.  The CPU produces three kinds of triangles:
//
//   texture_mode = +1  concave bezier region — fill where u²−v ≥ 0
//   texture_mode = −1  convex  bezier region — fill where u²−v ≤ 0
//   texture_mode =  0  flat interior         — always fill
//
// Uniform layout (group 0, binding 0) — 144 bytes total:
//   offset   0 — projection  mat4x4<f32>   (64 bytes)
//   offset  64 — view        mat4x4<f32>   (64 bytes)
//   offset 128 — light_pos   vec3<f32>     (12 bytes, padded to 16)
//
// Vertex attributes:
//   location 0 — in_vert        vec3<f32>   world-space position
//   location 1 — in_color       vec4<f32>   RGBA fill colour
//   location 2 — texture_coords vec2<f32>   Loop-Blinn UV (u, v)
//   location 3 — texture_mode   f32         0 / +1 / −1 (stored as float)

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    light_pos  : vec3<f32>,
    _pad       : f32,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) in_vert        : vec3<f32>,
    @location(1) in_color       : vec4<f32>,
    @location(2) texture_coords : vec2<f32>,
    @location(3) texture_mode   : f32,
};

struct VertexOutput {
    @builtin(position)              clip_position    : vec4<f32>,
    @location(0)                    v_color          : vec4<f32>,
    @location(1)                    v_texture_coords : vec2<f32>,
    @location(2) @interpolate(flat) v_texture_mode   : i32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position    = u.projection * u.view * vec4<f32>(in.in_vert, 1.0);
    out.v_color          = in.in_color;
    out.v_texture_coords = in.texture_coords;
    out.v_texture_mode   = i32(in.texture_mode);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv         = in.v_texture_coords;
    let curve_func = uv.x * uv.x - uv.y;
    // texture_mode == 0  → always keep (interior)
    // sign(texture_mode) * curve_func >= 0  → keep (bezier edge region)
    if (f32(in.v_texture_mode) * curve_func >= 0.0) {
        return in.v_color;
    }
    discard;
}
