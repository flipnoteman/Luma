// Copies a real signal, zero-padded to `n`, into a complex buffer in bit-reversed order.
struct Params { n: u32, input_len: u32 }

@group(1) @binding(0) var<storage, read_write> output: array<f32>;
@group(1) @binding(1) var<storage, read> input_data: array<f32>;
@group(1) @binding(2) var<uniform> params: Params;

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
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= params.n) {
        return;
    }

    let log2n = count_trailing_zeros_pow2(params.n);
    let rev = bit_reverse(idx, log2n);

    var re: f32 = 0.0;
    if (rev < params.input_len) {
        re = input_data[rev];
    }

    output[2u * idx] = re;
    output[2u * idx + 1u] = 0.0;
}
