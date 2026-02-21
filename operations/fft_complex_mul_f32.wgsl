@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 1>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[0];

    if (idx >= n) {
        return;
    }

    let a_re = a[2u * idx];
    let a_im = a[2u * idx + 1u];
    let b_re = b[2u * idx];
    let b_im = b[2u * idx + 1u];

    result[2u * idx] = a_re * b_re - a_im * b_im;
    result[2u * idx + 1u] = a_re * b_im + a_im * b_re;
}
