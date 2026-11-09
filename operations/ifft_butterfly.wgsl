// One radix-2 butterfly stage of the inverse FFT (conjugate twiddles), in place. The last stage
// also divides by n. Launched with n / 2 invocations.
struct Params { n: u32, stage: u32, is_last_stage: u32 }

@group(1) @binding(0) var<storage, read_write> data: array<f32>;
@group(1) @binding(1) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let thread_id = gid.x + gid.y * nwg.x * 64u;

    let half_n = params.n / 2u;
    if (thread_id >= half_n) {
        return;
    }

    let pairs_per_group = 1u << (params.stage + 1u);
    let half_group = pairs_per_group / 2u;
    let group = thread_id / half_group;
    let pos_in_group = thread_id % half_group;

    let top_idx = group * pairs_per_group + pos_in_group;
    let bot_idx = top_idx + half_group;

    // Conjugate twiddle: +2π instead of -2π
    let angle = 2.0 * PI * f32(pos_in_group) / f32(pairs_per_group);
    let tw_re = cos(angle);
    let tw_im = sin(angle);

    let top_re = data[2u * top_idx];
    let top_im = data[2u * top_idx + 1u];
    let bot_re = data[2u * bot_idx];
    let bot_im = data[2u * bot_idx + 1u];

    let t_re = tw_re * bot_re - tw_im * bot_im;
    let t_im = tw_re * bot_im + tw_im * bot_re;

    var new_top_re = top_re + t_re;
    var new_top_im = top_im + t_im;
    var new_bot_re = top_re - t_re;
    var new_bot_im = top_im - t_im;

    // Normalize on last stage
    if (params.is_last_stage == 1u) {
        let inv_n = 1.0 / f32(params.n);
        new_top_re = new_top_re * inv_n;
        new_top_im = new_top_im * inv_n;
        new_bot_re = new_bot_re * inv_n;
        new_bot_im = new_bot_im * inv_n;
    }

    data[2u * top_idx] = new_top_re;
    data[2u * top_idx + 1u] = new_top_im;
    data[2u * bot_idx] = new_bot_re;
    data[2u * bot_idx + 1u] = new_bot_im;
}
