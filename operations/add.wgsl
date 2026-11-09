@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read> dims: array<u32, 4>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= arrayLength(&output)) {
        return;
    }
    let rhs_len = arrayLength(&rhs);
    let rhs_idx = select(idx, 0u, rhs_len == 1u);
    output[idx] = output[idx] + rhs[rhs_idx];
}
