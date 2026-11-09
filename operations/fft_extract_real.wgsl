// Extracts the real parts of a complex buffer.
struct Params { output_len: u32 }

@group(1) @binding(0) var<storage, read_write> output: array<f32>;
@group(1) @binding(1) var<storage, read> input_data: array<f32>;
@group(1) @binding(2) var<uniform> params: Params;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= params.output_len) {
        return;
    }

    output[idx] = input_data[2u * idx];
}
