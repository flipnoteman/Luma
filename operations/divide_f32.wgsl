@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
// Shapes of `output` and `rhs`. Buffers may be larger than the array they hold, so element
// counts come from the shapes, never from arrayLength().
@group(0) @binding(2) var<storage, read> dims: array<u32, 4>;
@group(0) @binding(3) var<storage, read> rhs_dims: array<u32, 4>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let len = dims[0] * dims[1] * dims[2] * dims[3];
    if (idx >= len) {
        return;
    }
    let rhs_len = rhs_dims[0] * rhs_dims[1] * rhs_dims[2] * rhs_dims[3];
    let rhs_idx = select(idx, 0u, rhs_len == 1u);
    output[idx] = output[idx] / rhs[rhs_idx];
}
