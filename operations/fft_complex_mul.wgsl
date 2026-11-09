// Elementwise product of two complex buffers.
struct Params { n: u32 }

@group(1) @binding(0) var<storage, read_write> result: array<f32>;
@group(1) @binding(1) var<storage, read> a: array<f32>;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> params: Params;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= params.n) {
        return;
    }

    let a_re = a[2u * idx];
    let a_im = a[2u * idx + 1u];
    let b_re = b[2u * idx];
    let b_im = b[2u * idx + 1u];

    result[2u * idx] = a_re * b_re - a_im * b_im;
    result[2u * idx + 1u] = a_re * b_im + a_im * b_re;
}
