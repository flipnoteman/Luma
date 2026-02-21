@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 3>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let output_len = arrayLength(&output);
    if (out_idx >= output_len) {
        return;
    }

    let kernel_len = params[0];
    let stride = params[1];
    let padding = params[2];

    let in_start = i32(out_idx * stride) - i32(padding);
    var sum: f32 = 0.0;
    let input_len = arrayLength(&input_data);

    for (var k: u32 = 0u; k < kernel_len; k = k + 1u) {
        let in_idx = in_start + i32(k);
        if in_idx >= 0 && u32(in_idx) < input_len {
            sum = sum + input_data[u32(in_idx)] * kernel[k];
        }
    }

    output[out_idx] = sum;
}
