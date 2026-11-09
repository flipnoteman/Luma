@group(0) @binding(0) var<storage, read_write> v_indices: array<u32>; // used as both input and output
@group(0) @binding(1) var<storage, read> dims: array<u32, 4>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= arrayLength(&v_indices)) {
        return;
    }
    v_indices[idx] = v_indices[idx] * 2u;
}
