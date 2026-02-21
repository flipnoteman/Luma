@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 2>;

fn bit_reverse(x: u32, log2n: u32) -> u32 {
    var result: u32 = 0u;
    var val: u32 = x;
    for (var i: u32 = 0u; i < log2n; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

fn count_trailing_zeros_pow2(n: u32) -> u32 {
    var count: u32 = 0u;
    var val: u32 = n;
    while (val > 1u) {
        val = val >> 1u;
        count = count + 1u;
    }
    return count;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[0];
    let input_len = params[1];

    if (idx >= n) {
        return;
    }

    let log2n = count_trailing_zeros_pow2(n);
    let rev = bit_reverse(idx, log2n);

    var re: f32 = 0.0;
    if (rev < input_len) {
        re = input_data[rev];
    }

    output[2u * idx] = re;
    output[2u * idx + 1u] = 0.0;
}
