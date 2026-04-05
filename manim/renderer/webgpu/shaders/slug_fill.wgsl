// WebGPU fill shader using the Slug algorithm.
//
// Renders VMobject fill with exact, analytical winding-number coverage and
// smooth sub-pixel anti-aliasing.  No CPU tessellation is required — raw
// quadratic bezier control points are uploaded once per frame in a storage
// buffer, and coverage is computed entirely in the fragment shader.
//
// Reference:
//   E. Lengyel, "GPU-Centered Font Rendering Directly from Glyph Outlines",
//   JCGT Vol. 6 No. 2, 2017.  https://github.com/EricLengyel/Slug
//   Patent dedicated to public domain.  Code: MIT license.
//
// Uniform layout (group 0, binding 0) — 144 bytes:
//   offset   0 — projection  mat4x4<f32>  (64 bytes)
//   offset  64 — view        mat4x4<f32>  (64 bytes)
//   offset 128 — light_pos   vec3<f32>    (12 bytes, padded to 16)
//
// Storage buffer (group 0, binding 1) — flat array of vec2<f32>:
//   curves[i*3 + 0] = p1  (start anchor of quadratic bezier i)
//   curves[i*3 + 1] = p2  (single control point)
//   curves[i*3 + 2] = p3  (end anchor)
//   All coordinates in Manim world space.
//
// Vertex attributes:
//   location 0 — in_pos       vec2<f32>   world-space bounding-quad corner
//   location 1 — in_color     vec4<f32>   RGBA fill colour
//   location 2 — curve_start  u32         first curve index in storage buffer
//   location 3 — n_curves     u32         number of quadratic bezier curves

struct Uniforms {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    light_pos  : vec3<f32>,
    _pad       : f32,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

// Flat array of vec2 control points: three entries per quadratic bezier curve.
@group(0) @binding(1) var<storage, read> curves : array<vec2<f32>>;

struct VertexInput {
    @location(0) in_pos      : vec2<f32>,
    @location(1) in_color    : vec4<f32>,
    @location(2) curve_start : u32,
    @location(3) n_curves    : u32,
};

struct VertexOutput {
    @builtin(position)              clip_pos    : vec4<f32>,
    @location(0)                    world_pos   : vec2<f32>,
    @location(1)                    v_color     : vec4<f32>,
    @location(2) @interpolate(flat) curve_start : u32,
    @location(3) @interpolate(flat) n_curves    : u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos    = u.projection * u.view * vec4<f32>(in.in_pos, 0.0, 1.0);
    out.world_pos   = in.in_pos;
    out.v_color     = in.in_color;
    out.curve_start = in.curve_start;
    out.n_curves    = in.n_curves;
    return out;
}

// ---------------------------------------------------------------------------
// Slug algorithm — adapted from Lengyel 2017 (HLSL → WGSL)
// ---------------------------------------------------------------------------

// Return root eligibility code for a sample-relative quadratic bezier.
// Extracts the sign bits of the three y-coordinates and maps them through
// a lookup table to determine which roots of the quadratic cross y = 0 in
// a winding-compatible direction.
// Result: bit 0 = root 1 eligible,  bit 8 = root 2 eligible.
fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = (bitcast<u32>(y1) >> 31u) & 1u;
    let i2 = (bitcast<u32>(y2) >> 30u) & 2u;
    let i3 = (bitcast<u32>(y3) >> 29u) & 4u;
    let shift = i3 | i2 | i1;
    return (0x2E74u >> shift) & 0x0101u;
}

// Solve quadratic bezier for y = 0 crossings; return x-coordinates.
// C(t) = (1-t)^2 p1 + 2t(1-t) p2 + t^2 p3,  t in [0,1].
// Polynomial: a*t^2 - 2*b*t + c = 0
//   a = p1.y - 2*p2.y + p3.y
//   b = p1.y - p2.y
//   c = p1.y                    (the sample has already been subtracted)
fn solve_horiz(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> vec2<f32> {
    let ay = p1.y - 2.0 * p2.y + p3.y;
    let by = p1.y - p2.y;
    let ax = p1.x - 2.0 * p2.x + p3.x;
    let bx = p1.x - p2.x;

    var t1: f32;
    var t2: f32;

    if abs(ay) < (1.0 / 65536.0) {
        // Nearly linear — solve -2*by*t + p1.y = 0.
        let denom = select(1.0, by, abs(by) > 1e-10);
        t1 = p1.y * 0.5 / denom;
        t2 = t1;
    } else {
        let ra = 1.0 / ay;
        let d  = sqrt(max(by * by - ay * p1.y, 0.0));
        t1 = (by - d) * ra;
        t2 = (by + d) * ra;
    }

    let x1 = (ax * t1 - bx * 2.0) * t1 + p1.x;
    let x2 = (ax * t2 - bx * 2.0) * t2 + p1.x;
    return vec2<f32>(x1, x2);
}

// Solve quadratic bezier for x = 0 crossings; return y-coordinates.
fn solve_vert(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> vec2<f32> {
    let ax = p1.x - 2.0 * p2.x + p3.x;
    let bx = p1.x - p2.x;
    let ay = p1.y - 2.0 * p2.y + p3.y;
    let by = p1.y - p2.y;

    var t1: f32;
    var t2: f32;

    if abs(ax) < (1.0 / 65536.0) {
        let denom = select(1.0, bx, abs(bx) > 1e-10);
        t1 = p1.x * 0.5 / denom;
        t2 = t1;
    } else {
        let ra = 1.0 / ax;
        let d  = sqrt(max(bx * bx - ax * p1.x, 0.0));
        t1 = (bx - d) * ra;
        t2 = (bx + d) * ra;
    }

    let y1 = (ay * t1 - by * 2.0) * t1 + p1.y;
    let y2 = (ay * t2 - by * 2.0) * t2 + p1.y;
    return vec2<f32>(y1, y2);
}

// Combine horizontal and vertical winding coverage into [0, 1].
// The weighted blend handles pixels where one ray's result is more reliable.
fn calc_coverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let blended = abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0);
    let fallback = min(abs(xcov), abs(ycov));
    return clamp(max(blended, fallback), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // World units per screen pixel — lets coverage calculations work in
    // pixel units regardless of zoom or resolution.
    let ems_per_pixel = fwidth(in.world_pos);
    let pixels_per_em = 1.0 / max(ems_per_pixel, vec2<f32>(1e-9));

    var xcov = 0.0;  var xwgt = 0.0;
    var ycov = 0.0;  var ywgt = 0.0;

    for (var i = 0u; i < in.n_curves; i = i + 1u) {
        let base = (in.curve_start + i) * 3u;

        // Shift curve so the current fragment is the origin.
        let p1 = curves[base      ] - in.world_pos;
        let p2 = curves[base + 1u ] - in.world_pos;
        let p3 = curves[base + 2u ] - in.world_pos;

        // ── Horizontal ray: accumulate x-coverage ────────────────────────
        let hcode = calc_root_code(p1.y, p2.y, p3.y);
        if hcode != 0u {
            let r = solve_horiz(p1, p2, p3) * pixels_per_em.x;
            if (hcode & 1u) != 0u {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt  = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if hcode > 1u {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt  = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }

        // ── Vertical ray: accumulate y-coverage ──────────────────────────
        let vcode = calc_root_code(p1.x, p2.x, p3.x);
        if vcode != 0u {
            let r = solve_vert(p1, p2, p3) * pixels_per_em.y;
            if (vcode & 1u) != 0u {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt  = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if vcode > 1u {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt  = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    let coverage = calc_coverage(xcov, ycov, xwgt, ywgt);
    if coverage <= 0.0 {
        discard;
    }
    return vec4<f32>(in.v_color.rgb, in.v_color.a * coverage);
}
