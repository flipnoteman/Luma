// Spatial 1D convolution (cross-correlation). Launched with one invocation per output element,
// folded over x and y workgroups (`out_idx = gid.x + gid.y * nwg.x * 64`). Buffers may be larger
// than the arrays they hold, so lengths come from the parameters, never from arrayLength().
struct Params {
    kernel_len: u32,
    stride: u32,
    padding: u32,
    input_len: u32,
    output_len: u32,
}

@group(1) @binding(0) var<storage, read_write> output: array<f32>;
@group(1) @binding(1) var<storage, read> input_data: array<f32>;
@group(1) @binding(2) var<storage, read> kernel: array<f32>;
@group(1) @binding(3) var<uniform> params: Params;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let out_idx = gid.x + gid.y * nwg.x * 64u;
    if (out_idx >= params.output_len) {
        return;
    }

    let in_start = i32(out_idx * params.stride) - i32(params.padding);
    var sum: f32 = 0.0;

    for (var k: u32 = 0u; k < params.kernel_len; k = k + 1u) {
        let in_idx = in_start + i32(k);
        if in_idx >= 0 && u32(in_idx) < params.input_len {
            sum = sum + input_data[u32(in_idx)] * kernel[k];
        }
    }

    output[out_idx] = sum;
}
