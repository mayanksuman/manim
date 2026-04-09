// WebGPU TrueDot shader — screen-aligned sphere dot rendering.
//
// Each dot is expanded CPU-side into 2 triangles (6 vertices) forming a
// screen-aligned quad.  UV coords span (-1,-1) → (1,1) across the quad.
// The fragment shader treats the quad as a sphere projected onto the screen:
//   • Pixels outside the unit disc are discarded (anti-aliased edge).
//   • The sphere normal is reconstructed from the UV position.
//   • Cairo-style lighting (gloss/shadow) is applied to the colour.
//
// The same camera Uniforms struct and bind group layout as the surface
// shaders are reused (binding 0, group 0).
//
// Vertex layout (stride 48 bytes) — must match _TRUE_DOT_DTYPE:
//   location 0 — center  float32x3  offset  0   (12 B)
//   location 1 — color   float32x4  offset 12   (16 B)
//   location 2 — uv      float32x2  offset 28   ( 8 B)
//   location 3 — radius  float32    offset 36   ( 4 B)
//   location 4 — gloss   float32    offset 40   ( 4 B)
//   location 5 — shadow  float32    offset 44   ( 4 B)

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
    @location(0) center : vec3<f32>,
    @location(1) color  : vec4<f32>,
    @location(2) uv     : vec2<f32>,
    @location(3) radius : f32,
    @location(4) gloss  : f32,
    @location(5) shadow : f32,
};

struct VertexOutput {
    @builtin(position)              clip_position  : vec4<f32>,
    @location(0)                    v_color        : vec4<f32>,
    @location(1)                    v_uv           : vec2<f32>,
    @location(2) @interpolate(flat) v_gloss        : f32,
    @location(3) @interpolate(flat) v_shadow       : f32,
    @location(4)                    v_center_view  : vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Project dot center into view space.
    let cv = u.view * vec4<f32>(in.center, 1.0);

    // Expand quad in view space: move corner by radius × UV along x/y.
    // This replicates the OpenGL geometry shader expansion and naturally
    // applies the correct perspective foreshortening (larger expansion near
    // the camera, smaller far away).
    let expanded = cv + vec4<f32>(in.uv.x * in.radius, in.uv.y * in.radius, 0.0, 0.0);

    var out: VertexOutput;
    out.clip_position = u.projection * expanded;
    out.v_color       = in.color;
    out.v_uv          = in.uv;
    out.v_gloss       = in.gloss;
    out.v_shadow      = in.shadow;
    out.v_center_view = cv.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.v_uv);

    // Anti-aliased disc: smoothstep over one pixel width around d == 1.
    let fw = fwidth(d);
    let alpha_mult = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, d);
    if alpha_mult <= 0.001 { discard; }

    // Reconstruct sphere surface normal in view space from UV position.
    let z2 = max(0.0, 1.0 - d * d);
    let sphere_normal = normalize(vec3<f32>(in.v_uv.x, in.v_uv.y, sqrt(z2)));

    // Light and camera directions in view space.
    // Camera sits at the origin in view space, so to_camera = -in.v_center_view.
    let light_view = (u.view * vec4<f32>(u.light_pos, 1.0)).xyz;
    let to_light   = normalize(light_view - in.v_center_view);
    let to_camera  = normalize(-in.v_center_view);

    // Cairo-style lighting (finalize_color.glsl → add_light):
    //   shine     = gloss  * exp(-3 * (1 - dot(reflect(-L, N), V))^2)
    //   darkening = mix(1, max(dot(L, N), 0), shadow)
    //   out_rgb   = darkening * mix(color, WHITE, shine)
    let light_reflection = reflect(-to_light, sphere_normal);
    let dot_rv  = clamp(dot(light_reflection, to_camera), 0.0, 1.0);
    let shine   = in.v_gloss * exp(-3.0 * pow(1.0 - dot_rv, 2.0));
    let dp2     = dot(to_light, sphere_normal);
    let darkening = mix(1.0, max(dp2, 0.0), in.v_shadow);

    let lit_rgb = darkening * mix(in.v_color.rgb, vec3<f32>(1.0), shine);
    return vec4<f32>(lit_rgb, in.v_color.a * alpha_mult);
}
