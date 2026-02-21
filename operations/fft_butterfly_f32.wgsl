@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32, 2>;

const PI: f32 = 3.14159265358979323846;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let n = params[0];
    let stage = params[1];

    let half_n = n / 2u;
    if (thread_id >= half_n) {
        return;
    }

    let pairs_per_group = 1u << (stage + 1u);
    let half_group = pairs_per_group / 2u;
    let group = thread_id / half_group;
    let pos_in_group = thread_id % half_group;

    let top_idx = group * pairs_per_group + pos_in_group;
    let bot_idx = top_idx + half_group;

    let angle = -2.0 * PI * f32(pos_in_group) / f32(pairs_per_group);
    let tw_re = cos(angle);
    let tw_im = sin(angle);

    let top_re = data[2u * top_idx];
    let top_im = data[2u * top_idx + 1u];
    let bot_re = data[2u * bot_idx];
    let bot_im = data[2u * bot_idx + 1u];

    // Twiddle * bottom
    let t_re = tw_re * bot_re - tw_im * bot_im;
    let t_im = tw_re * bot_im + tw_im * bot_re;

    data[2u * top_idx] = top_re + t_re;
    data[2u * top_idx + 1u] = top_im + t_im;
    data[2u * bot_idx] = top_re - t_re;
    data[2u * bot_idx + 1u] = top_im - t_im;
}
