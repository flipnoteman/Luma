@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read> dims: array<u32, 4>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let rhs_len = arrayLength(&rhs);
    let rhs_idx = select(idx, 0u, rhs_len == 1u);
    output[idx] = output[idx] + rhs[rhs_idx];
}
