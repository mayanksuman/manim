// WebGPU stroke shader for VMobject — 2-D and 3-D curves.
//
// The vertex shader transforms control points to VIEW space first (camera
// looks along −Z, so view-space XY is the screen plane), then builds the
// bounding tile in that 2-D screen space.  This correctly handles 3-D curves
// such as the world Z-axis whose world-XY chord collapses to zero length.
//
// Uniform layout (group 0, binding 0):
//   offset   0 — projection  mat4x4<f32>   (64 bytes)
//   offset  64 — view        mat4x4<f32>   (64 bytes)
//   offset 128 — light_pos   vec3<f32>     (12 bytes, padded to 16)
//
// Vertex attributes:
//   location 0-3 — current_curve_{0-3}  vec3<f32>  cubic bezier control points
//   location 4   — tile_coordinate      vec2<f32>  quad corner ∈ [0,1]
//   location 5   — in_color             vec4<f32>  RGBA stroke colour
//   location 6   — in_width             f32        stroke width (Manim units)

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

fn ud_cubic_bezier(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>,
    pos: vec2<f32>,
) -> f32 {
    var best_t : f32 = 0.0;
    var best_d2 : f32 = 1e18;
    for (var i = 0u; i <= 8u; i = i + 1u) {
        let t  = f32(i) * (1.0 / 8.0);
        let pt = cubic_eval(p0, p1, p2, p3, t);
        let d2 = dot(pt - pos, pt - pos);
        if (d2 < best_d2) { best_d2 = d2; best_t = t; }
    }
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

fn bbox_cubic(
    p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>
) -> vec4<f32> {
    var mi = min(p0, p3);
    var ma = max(p0, p3);

    let a_v = p1 - p0;
    let b_v = p2 - p1;
    let c_v = p3 - p2;
    let A   = a_v - 2.0*b_v + c_v;
    let B   = 2.0 * (b_v - a_v);
    let C   = a_v;

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

// ---- Vertex shader ---------------------------------------------------------
//
// Strategy: transform control points to VIEW space first.  In view space the
// camera looks along −Z, so the XY plane is the screen plane.  The chord XY
// always has the correct 2-D screen-space direction regardless of the 3-D
// curve orientation.  The bounding tile is built in view-space XY, then the
// tile corners are projected to clip space using the projection matrix.
// Using avg-Z for all tile corners is a valid approximation for curves that
// are short relative to the viewing distance.

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let thickness_multiplier = 0.004;
    var out: VertexOutput;
    out.v_color     = in.in_color;
    out.v_thickness = thickness_multiplier * in.in_width;

    // Transform all 4 control points to view space.
    let vs0 = (u.view * vec4<f32>(in.current_curve_0, 1.0)).xyz;
    let vs1 = (u.view * vec4<f32>(in.current_curve_1, 1.0)).xyz;
    let vs2 = (u.view * vec4<f32>(in.current_curve_2, 1.0)).xyz;
    let vs3 = (u.view * vec4<f32>(in.current_curve_3, 1.0)).xyz;

    // Tile x-axis: chord direction in view-space XY (= screen plane).
    let chord_xy  = vs3.xy - vs0.xy;
    let chord_len = length(chord_xy);

    var tile_x : vec2<f32>;
    if (chord_len > 1e-6) {
        tile_x = chord_xy / chord_len;
    } else {
        let alt     = vs1.xy - vs0.xy;
        let alt_len = length(alt);
        if (alt_len > 1e-6) {
            tile_x = alt / alt_len;
        } else {
            // Curve points directly toward / away from camera — collapse quad.
            out.clip_position = u.projection * vec4<f32>(vs0, 1.0);
            out.uv_point   = vec2<f32>(0.0);
            out.uv_curve_0 = vec2<f32>(0.0);
            out.uv_curve_1 = vec2<f32>(0.0);
            out.uv_curve_2 = vec2<f32>(0.0);
            out.uv_curve_3 = vec2<f32>(0.0);
            return out;
        }
    }
    let tile_y = vec2<f32>(-tile_x.y, tile_x.x);  // 90° CCW in screen plane

    // Project control points into the 2-D tile UV space (view-space XY).
    // The view matrix is a rigid-body transform, so UV distances == world distances
    // and the thickness comparison in the fragment shader remains correct.
    let uv0 = vec2<f32>(dot(vs0.xy, tile_x), dot(vs0.xy, tile_y));
    let uv1 = vec2<f32>(dot(vs1.xy, tile_x), dot(vs1.xy, tile_y));
    let uv2 = vec2<f32>(dot(vs2.xy, tile_x), dot(vs2.xy, tile_y));
    let uv3 = vec2<f32>(dot(vs3.xy, tile_x), dot(vs3.xy, tile_y));
    out.uv_curve_0 = uv0;
    out.uv_curve_1 = uv1;
    out.uv_curve_2 = uv2;
    out.uv_curve_3 = uv3;

    let t      = out.v_thickness;
    let aa_pad = thickness_multiplier * 2.0;
    let uv_bb  = bbox_cubic(uv0, uv1, uv2, uv3);
    let uv_min = uv_bb.xy - vec2<f32>(t + aa_pad);
    let uv_max = uv_bb.zw + vec2<f32>(t + aa_pad);

    let uv_tile  = mix(uv_min, uv_max, in.tile_coordinate);
    out.uv_point = uv_tile;

    // Reconstruct the 3-D view-space position: XY from tile UV, Z from
    // the average depth of the 4 control points.
    // Reconstruct the 3-D view-space position: XY from tile UV.
    // For Z, interpolate strictly along tile_x to maintain proper 3-D depth
    // for perspective projection and correct occlusion with 3-D surfaces.
    var vs_z : f32;
    if (chord_len > 1e-6) {
        let t = (uv_tile.x - uv0.x) / chord_len;
        vs_z = mix(vs0.z, vs3.z, t);
    } else {
        vs_z = 0.5 * (vs0.z + vs3.z);
    }

    let vs_tile = vec3<f32>(
        uv_tile.x * tile_x.x + uv_tile.y * tile_y.x,
        uv_tile.x * tile_x.y + uv_tile.y * tile_y.y,
        vs_z,
    );
    out.clip_position = u.projection * vec4<f32>(vs_tile, 1.0);
    return out;
}

// ---- Fragment shader -------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = ud_cubic_bezier(
        in.uv_curve_0, in.uv_curve_1, in.uv_curve_2, in.uv_curve_3,
        in.uv_point,
    );

    let px        = fwidthFine(dist);
    let half_px   = 0.5 * px;
    let edge_low  = in.v_thickness - half_px;
    let edge_high = in.v_thickness + half_px;
    let coverage  = 1.0 - smoothstep(edge_low, edge_high, dist);

    if (coverage <= 0.0) { discard; }

    return vec4<f32>(in.v_color.rgb, in.v_color.a * coverage);
}
