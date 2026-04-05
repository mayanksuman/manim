// WebGPU stroke shader for VMobject — Phase 2/3 (true cubic bezier).
//
// All four cubic bezier control points (b0, h0, h1, b3) are passed from the
// CPU without approximation.  The vertex shader builds a tight bounding quad
// from the cubic AABB, and the fragment shader computes the exact unsigned
// distance to the cubic bezier via Newton's-method minimisation, discarding
// fragments outside the stroke half-width.
//
// Uniform layout (group 0, binding 0):
//   offset   0 — projection  mat4x4<f32>   (64 bytes)
//   offset  64 — view        mat4x4<f32>   (64 bytes)
//   offset 128 — light_pos   vec3<f32>     (12 bytes, padded to 16)
//
// Vertex attributes:
//   location 0 — current_curve_0  vec3<f32>  start anchor  (b0)
//   location 1 — current_curve_1  vec3<f32>  first handle  (h0)
//   location 2 — current_curve_2  vec3<f32>  second handle (h1)
//   location 3 — current_curve_3  vec3<f32>  end anchor    (b3)
//   location 4 — tile_coordinate  vec2<f32>  quad corner ∈ [0,1]
//   location 5 — in_color         vec4<f32>  RGBA stroke colour
//   location 6 — in_width         f32        stroke width (Manim units)

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    light_pos  : vec3<f32>,
    _pad       : f32,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

// ---- Cubic bezier helpers ------------------------------------------------

fn cubic_eval(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32
) -> vec2<f32> {
    let s = 1.0 - t;
    return s*s*s*p0 + 3.0*s*s*t*p1 + 3.0*s*t*t*p2 + t*t*t*p3;
}

fn cubic_deriv1(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32
) -> vec2<f32> {
    let s = 1.0 - t;
    return 3.0 * (s*s*(p1 - p0) + 2.0*s*t*(p2 - p1) + t*t*(p3 - p2));
}

fn cubic_deriv2(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32
) -> vec2<f32> {
    return 6.0 * ((1.0 - t)*(p2 - 2.0*p1 + p0) + t*(p3 - 2.0*p2 + p1));
}

// Unsigned distance from pos to the cubic bezier.  Uses coarse sampling to
// seed Newton's-method minimisation of f(t) = |B(t) − pos|².
fn ud_cubic_bezier(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>,
    pos: vec2<f32>,
) -> f32 {
    // Coarse sampling: 9 equally-spaced t values.
    var best_t : f32 = 0.0;
    var best_d2 : f32 = 1e18;
    for (var i = 0u; i <= 8u; i = i + 1u) {
        let t  = f32(i) * (1.0 / 8.0);
        let pt = cubic_eval(p0, p1, p2, p3, t);
        let d2 = dot(pt - pos, pt - pos);
        if (d2 < best_d2) { best_d2 = d2; best_t = t; }
    }
    // Newton refinement.
    for (var k = 0u; k < 4u; k = k + 1u) {
        let t    = clamp(best_t, 0.0, 1.0);
        let pt   = cubic_eval(p0, p1, p2, p3, t);
        let dp   = cubic_deriv1(p0, p1, p2, p3, t);
        let d2p  = cubic_deriv2(p0, p1, p2, p3, t);
        let diff = pt - pos;
        let denom = dot(dp, dp) + dot(diff, d2p);
        if (abs(denom) > 1e-10) { best_t = t - dot(diff, dp) / denom; }
    }
    let closest = cubic_eval(p0, p1, p2, p3, clamp(best_t, 0.0, 1.0));
    return length(closest - pos);
}

// ---- Bounding-box helpers ------------------------------------------------

fn cubic_eval_1d(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let s = 1.0 - t;
    return s*s*s*p0 + 3.0*s*s*t*p1 + 3.0*s*t*t*p2 + t*t*t*p3;
}

// Axis-aligned bounding box of a 2-D cubic bezier.  Returns vec4(min_xy, max_xy).
// Roots of the quadratic derivative give potential extremes between the endpoints.
fn bbox_cubic(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>
) -> vec4<f32> {
    var mi = min(p0, p3);
    var ma = max(p0, p3);

    // B'(t)/3 = a(1-t)^2 + 2b(1-t)t + ct^2  →  At^2 + Bt + C = 0
    // where A = a-2b+c,  B = 2(b-a),  C = a,  a=(p1-p0), b=(p2-p1), c=(p3-p2)
    let a_v = p1 - p0;
    let b_v = p2 - p1;
    let c_v = p3 - p2;
    let A   = a_v - 2.0*b_v + c_v;
    let B   = 2.0 * (b_v - a_v);
    let C   = a_v;

    // x component
    if (abs(A.x) > 1e-8) {
        let disc = B.x*B.x - 4.0*A.x*C.x;
        if (disc >= 0.0) {
            let sq = sqrt(disc);
            let t1 = (-B.x + sq) / (2.0*A.x);
            let t2 = (-B.x - sq) / (2.0*A.x);
            if (t1 > 0.0 && t1 < 1.0) {
                let v = cubic_eval_1d(p0.x, p1.x, p2.x, p3.x, t1);
                mi.x = min(mi.x, v); ma.x = max(ma.x, v);
            }
            if (t2 > 0.0 && t2 < 1.0) {
                let v = cubic_eval_1d(p0.x, p1.x, p2.x, p3.x, t2);
                mi.x = min(mi.x, v); ma.x = max(ma.x, v);
            }
        }
    } else if (abs(B.x) > 1e-8) {
        let t = -C.x / B.x;
        if (t > 0.0 && t < 1.0) {
            let v = cubic_eval_1d(p0.x, p1.x, p2.x, p3.x, t);
            mi.x = min(mi.x, v); ma.x = max(ma.x, v);
        }
    }

    // y component
    if (abs(A.y) > 1e-8) {
        let disc = B.y*B.y - 4.0*A.y*C.y;
        if (disc >= 0.0) {
            let sq = sqrt(disc);
            let t1 = (-B.y + sq) / (2.0*A.y);
            let t2 = (-B.y - sq) / (2.0*A.y);
            if (t1 > 0.0 && t1 < 1.0) {
                let v = cubic_eval_1d(p0.y, p1.y, p2.y, p3.y, t1);
                mi.y = min(mi.y, v); ma.y = max(ma.y, v);
            }
            if (t2 > 0.0 && t2 < 1.0) {
                let v = cubic_eval_1d(p0.y, p1.y, p2.y, p3.y, t2);
                mi.y = min(mi.y, v); ma.y = max(ma.y, v);
            }
        }
    } else if (abs(B.y) > 1e-8) {
        let t = -C.y / B.y;
        if (t > 0.0 && t < 1.0) {
            let v = cubic_eval_1d(p0.y, p1.y, p2.y, p3.y, t);
            mi.y = min(mi.y, v); ma.y = max(ma.y, v);
        }
    }

    return vec4<f32>(mi, ma);
}

fn to_uv(x_unit: vec3<f32>, y_unit: vec3<f32>, point: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(dot(point, x_unit), dot(point, y_unit));
}

fn from_uv(
    translation: vec3<f32>, x_unit: vec3<f32>, y_unit: vec3<f32>, p: vec2<f32>
) -> vec3<f32> {
    return p.x * x_unit + p.y * y_unit + translation;
}

// ---- Vertex I/O -----------------------------------------------------------

struct VertexInput {
    @location(0) current_curve_0 : vec3<f32>,
    @location(1) current_curve_1 : vec3<f32>,
    @location(2) current_curve_2 : vec3<f32>,
    @location(3) current_curve_3 : vec3<f32>,
    @location(4) tile_coordinate  : vec2<f32>,
    @location(5) in_color         : vec4<f32>,
    @location(6) in_width         : f32,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) v_thickness : f32,
    @location(1) uv_point    : vec2<f32>,
    @location(2) uv_curve_0  : vec2<f32>,
    @location(3) uv_curve_1  : vec2<f32>,
    @location(4) uv_curve_2  : vec2<f32>,
    @location(5) uv_curve_3  : vec2<f32>,
    @location(6) v_color     : vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let thickness_multiplier = 0.004;
    var out: VertexOutput;
    out.v_color     = in.in_color;
    out.v_thickness = thickness_multiplier * in.in_width;

    // For 2-D scenes the scene-normal is always +Z.
    let manim_unit_normal = vec3<f32>(0.0, 0.0, 1.0);

    // Tile x-axis follows the chord (b3 − b0); fall back to h0 direction.
    let chord     = in.current_curve_3 - in.current_curve_0;
    let chord_len = length(chord.xy);

    var tile_x_unit : vec3<f32>;
    if (chord_len > 1e-6) {
        tile_x_unit = vec3<f32>(normalize(chord.xy), 0.0);
    } else {
        let alt     = in.current_curve_1 - in.current_curve_0;
        let alt_len = length(alt.xy);
        if (alt_len > 1e-6) {
            tile_x_unit = vec3<f32>(normalize(alt.xy), 0.0);
        } else {
            // Truly degenerate — collapse to a point, nothing to draw.
            out.clip_position = u.projection * u.view * vec4<f32>(in.current_curve_0, 1.0);
            out.uv_point   = vec2<f32>(0.0);
            out.uv_curve_0 = vec2<f32>(0.0);
            out.uv_curve_1 = vec2<f32>(0.0);
            out.uv_curve_2 = vec2<f32>(0.0);
            out.uv_curve_3 = vec2<f32>(0.0);
            return out;
        }
    }
    let tile_y_unit = cross(manim_unit_normal, tile_x_unit);

    // Project all four cubic control points into the local tile UV space.
    let uv0 = to_uv(tile_x_unit, tile_y_unit, in.current_curve_0);
    let uv1 = to_uv(tile_x_unit, tile_y_unit, in.current_curve_1);
    let uv2 = to_uv(tile_x_unit, tile_y_unit, in.current_curve_2);
    let uv3 = to_uv(tile_x_unit, tile_y_unit, in.current_curve_3);
    out.uv_curve_0 = uv0;
    out.uv_curve_1 = uv1;
    out.uv_curve_2 = uv2;
    out.uv_curve_3 = uv3;

    // Tight bounding quad: cubic AABB padded by the stroke thickness plus a
    // small extra margin for the anti-aliasing fringe (≈1 pixel in UV space).
    // thickness_multiplier * 1.0 pixel ≈ aa_pad estimate; we use a fixed
    // conservative constant that matches the thickness_multiplier scale.
    let t      = out.v_thickness;
    let aa_pad = thickness_multiplier * 2.0; // ~1–2 px feather margin in UV
    let uv_bb  = bbox_cubic(uv0, uv1, uv2, uv3);
    let uv_min = uv_bb.xy - vec2<f32>(t + aa_pad);
    let uv_max = uv_bb.zw + vec2<f32>(t + aa_pad);

    // tile_coordinate ∈ [0,1]² → lerp within [uv_min, uv_max].
    let uv_tile = mix(uv_min, uv_max, in.tile_coordinate);

    let tile_translation = manim_unit_normal * dot(in.current_curve_0, manim_unit_normal);
    let tile_point       = from_uv(tile_translation, tile_x_unit, tile_y_unit, uv_tile);

    out.clip_position = u.projection * u.view * vec4<f32>(tile_point, 1.0);
    out.uv_point      = uv_tile;
    return out;
}

// ---- Fragment shader -------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = ud_cubic_bezier(
        in.uv_curve_0, in.uv_curve_1, in.uv_curve_2, in.uv_curve_3,
        in.uv_point,
    );

    // ── Analytic SDF anti-aliasing ──────────────────────────────────────────
    //
    // fwidthFine(dist) returns |∂dist/∂x| + |∂dist/∂y|, which approximates
    // the change in UV-space distance over one screen pixel.  We use half of
    // that as the half-width of the smooth transition band:
    //
    //   coverage = 1  when dist ≤ thickness − half_px   (fully inside)
    //   coverage = 0  when dist ≥ thickness + half_px   (fully outside)
    //
    // smoothstep interpolates smoothly between those limits.
    let px        = fwidthFine(dist);          // ≈ 1 px in UV units
    let half_px   = 0.5 * px;
    let edge_low  = in.v_thickness - half_px;
    let edge_high = in.v_thickness + half_px;
    let coverage  = 1.0 - smoothstep(edge_low, edge_high, dist);

    // Fully outside the anti-aliased fringe — discard to avoid touching the
    // depth/stencil buffer unnecessarily.
    if (coverage <= 0.0) { discard; }

    // Multiply the stored alpha by the smooth SDF coverage so transparent
    // strokes composite correctly.
    return vec4<f32>(in.v_color.rgb, in.v_color.a * coverage);
}
