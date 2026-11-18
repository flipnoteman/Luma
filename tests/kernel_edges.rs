//! Kernels exercising control flow, integer and bit edges, vectors/aggregates, workgroup memory
//! and atomics. Every expectation is computed in Rust with WGSL semantics.

use luma::*;

fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

fn one_d<T: GpuType>(data: &[T]) -> Array<T> {
    Array::from_slice(data, [data.len(), 1, 1, 1]).unwrap()
}

#[test]
fn switch_with_fallthrough_inside_loop_with_continue() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> out: array<u32>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let x = gid.x;
        if (x >= 64u) { return; }
        var acc = 0u;
        var k = 0u;
        loop {
            if ((x + k) % 7u == 3u) { continue; }
            switch ((x * 3u + k) % 6u) {
                case 0u: { acc = acc + 1u; }
                case 1u, 2u: { acc = acc * 3u + 2u; }
                case 4u: {
                    if (acc > 50u) { break; }
                    acc = acc + 7u;
                }
                default: { acc = acc ^ (k << 4u); }
            }
            acc = acc + k;
            continuing {
                k = k + 1u;
                break if k >= 9u;
            }
        }
        out[x] = acc;
    }"#;
    let mut out = Array::<u32>::zeros([64, 1, 1, 1]).unwrap();
    ctx().kernel(SRC).build().unwrap().launch([64], args![&mut out]).unwrap();
    let expected: Vec<u32> = (0..64u32)
        .map(|x| {
            let mut acc = 0u32;
            for k in 0..9u32 {
                if (x + k) % 7 == 3 {
                    continue;
                }
                match (x * 3 + k) % 6 {
                    0 => acc = acc.wrapping_add(1),
                    1 | 2 => acc = acc.wrapping_mul(3).wrapping_add(2),
                    4 => {
                        if acc <= 50 {
                            acc = acc.wrapping_add(7);
                        }
                    }
                    _ => acc ^= k << 4,
                }
                acc = acc.wrapping_add(k);
            }
            acc
        })
        .collect();
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn integer_division_shift_and_cast_edges() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read> ia: array<i32>;
    @group(1) @binding(1) var<storage, read> ib: array<i32>;
    @group(1) @binding(2) var<storage, read> ua: array<u32>;
    @group(1) @binding(3) var<storage, read> ub: array<u32>;
    @group(1) @binding(4) var<storage, read> sh: array<u32>;
    @group(1) @binding(5) var<storage, read> fl: array<f32>;
    @group(1) @binding(6) var<storage, read_write> io: array<i32>;
    @group(1) @binding(7) var<storage, read_write> uo: array<u32>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= arrayLength(&ia)) { return; }
        io[i * 6u] = ia[i] / ib[i];
        io[i * 6u + 1u] = ia[i] % ib[i];
        io[i * 6u + 2u] = ia[i] << sh[i];
        io[i * 6u + 3u] = ia[i] >> sh[i];
        io[i * 6u + 4u] = i32(fl[i]);
        io[i * 6u + 5u] = 0;
        uo[i * 6u] = ua[i] / ub[i];
        uo[i * 6u + 1u] = ua[i] % ub[i];
        uo[i * 6u + 2u] = ua[i] << sh[i];
        uo[i * 6u + 3u] = ua[i] >> sh[i];
        uo[i * 6u + 4u] = u32(fl[i]);
        uo[i * 6u + 5u] = 0u;
    }"#;
    let ia = [7, -7, i32::MIN, i32::MIN, 5, -1, 123456, -98765];
    let ib = [2, 0, -1, 3, 0, -1, -7, 0];
    let ua = [7u32, 0, u32::MAX, 5, 0x8000_0000, 1, 99, 3];
    let ub = [2u32, 0, 7, 0, 3, 1, 0, 10];
    let sh = [0u32, 31, 32, 33, 1, 31, 32, 33];
    // Positive overflow is left out: WGSL clamps to the largest f32 in range (2147483520 for
    // i32), while CUDA saturates to the integer maximum. NaN differs too.
    let fl = [-1e10, f32::NEG_INFINITY, 2e9, -3e9, -1.5, 1.5, 0.0, 2147483520.0];
    let n = ia.len();
    let mut io = Array::<i32>::zeros([n * 6, 1, 1, 1]).unwrap();
    let mut uo = Array::<u32>::zeros([n * 6, 1, 1, 1]).unwrap();
    let (a, b, c, d, e, f) = (one_d(&ia), one_d(&ib), one_d(&ua), one_d(&ub), one_d(&sh), one_d(&fl));
    ctx().kernel(SRC).build().unwrap().launch([n as u32], args![&a, &b, &c, &d, &e, &f, &mut io, &mut uo]).unwrap();
    let mut ie = Vec::new();
    let mut ue = Vec::new();
    for i in 0..n {
        let s = sh[i] % 32;
        ie.extend([
            ia[i].checked_div(ib[i]).unwrap_or(ia[i]),
            ia[i].checked_rem(ib[i]).unwrap_or(0),
            ia[i].wrapping_shl(s),
            ia[i] >> s,
            fl[i] as i32,
            0,
        ]);
        ue.extend([
            ua[i].checked_div(ub[i]).unwrap_or(ua[i]),
            ua[i].checked_rem(ub[i]).unwrap_or(0),
            ua[i] << s,
            ua[i] >> s,
            fl[i] as u32,
            0,
        ]);
    }
    assert_eq!(io.to_vec().unwrap(), ie);
    assert_eq!(uo.to_vec().unwrap(), ue);
}

#[test]
fn bit_counting_edges() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read> x: array<u32>;
    @group(1) @binding(1) var<storage, read_write> out: array<u32>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= arrayLength(&x)) { return; }
        let v = x[i];
        out[i * 6u] = countOneBits(v);
        out[i * 6u + 1u] = countLeadingZeros(v);
        out[i * 6u + 2u] = countTrailingZeros(v);
        out[i * 6u + 3u] = firstLeadingBit(v);
        out[i * 6u + 4u] = bitcast<u32>(firstLeadingBit(bitcast<i32>(v)));
        out[i * 6u + 5u] = firstTrailingBit(v);
    }"#;
    let xs = [0u32, 1, 0x8000_0000, 0xffff_ffff, 0x00f0];
    let mut out = Array::<u32>::zeros([xs.len() * 6, 1, 1, 1]).unwrap();
    let input = one_d(&xs);
    ctx().kernel(SRC).build().unwrap().launch([xs.len() as u32], args![&input, &mut out]).unwrap();
    let flb = |x: u32| if x == 0 { u32::MAX } else { 31 - x.leading_zeros() };
    let expected: Vec<u32> = xs
        .iter()
        .flat_map(|&v| {
            let signed = if (v as i32) < 0 { !v } else { v };
            [
                v.count_ones(),
                v.leading_zeros(),
                v.trailing_zeros(),
                flb(v),
                flb(signed),
                if v == 0 { u32::MAX } else { v.trailing_zeros() },
            ]
        })
        .collect();
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn vectors_arrays_and_structs() {
    const SRC: &str = r#"
    struct S { v: vec3<f32>, w: f32 }
    fn mk(a: f32, i: u32) -> S {
        var s: S;
        s.v = vec3<f32>(a, a * 2.0, a * 3.0);
        s.v[i] = -1.0;
        s.w = a + f32(i);
        return s;
    }
    @group(1) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= 3u) { return; }
        let fi = f32(i);
        var v = vec3<f32>(1.0 + fi, 2.0, 3.0);
        let r = v[i];
        v[i] = 10.0 * fi;
        let m = vec3<bool>(i == 0u, i != 1u, true);
        let s = select(vec3<f32>(-1.0, -2.0, -3.0), v, m);
        let sw = s.zxy;
        var arr = array<f32, 4>(5.0, 6.0, 7.0, 8.0);
        arr[i] = arr[i + 1u] + fi;
        let st = mk(fi + 0.5, i);
        let o = i * 12u;
        out[o] = r;
        out[o + 1u] = sw.x;
        out[o + 2u] = sw.y;
        out[o + 3u] = sw.z;
        out[o + 4u] = arr[0];
        out[o + 5u] = arr[1];
        out[o + 6u] = arr[2];
        out[o + 7u] = arr[3];
        out[o + 8u] = st.v.x;
        out[o + 9u] = st.v.y;
        out[o + 10u] = st.v.z;
        out[o + 11u] = st.w;
    }"#;
    let mut out = Array::<f32>::zeros([36, 1, 1, 1]).unwrap();
    ctx().kernel(SRC).build().unwrap().launch([3], args![&mut out]).unwrap();
    let mut expected = Vec::new();
    for i in 0..3usize {
        let fi = i as f32;
        let mut v = [1.0 + fi, 2.0, 3.0];
        let r = v[i];
        v[i] = 10.0 * fi;
        let m = [i == 0, i != 1, true];
        let alt = [-1.0, -2.0, -3.0];
        let s: Vec<f32> = (0..3).map(|k| if m[k] { v[k] } else { alt[k] }).collect();
        let mut arr = [5.0, 6.0, 7.0, 8.0];
        arr[i] = arr[i + 1] + fi;
        let a = fi + 0.5;
        let mut sv = [a, a * 2.0, a * 3.0];
        sv[i] = -1.0;
        expected.extend([r, s[2], s[0], s[1]]);
        expected.extend(arr);
        expected.extend(sv);
        expected.push(a + fi);
    }
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn workgroup_memory_starts_zeroed() {
    const SRC: &str = r#"
    var<workgroup> wg: array<u32, 64>;
    @group(1) @binding(0) var<storage, read_write> out: array<u32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
        out[gid.x] = wg[63u - lid.x];
        workgroupBarrier();
        wg[lid.x] = gid.x + 1u;
    }"#;
    let n = 1024usize;
    let mut out = Array::<u32>::from_slice(&vec![7u32; n], [n, 1, 1, 1]).unwrap();
    ctx().kernel(SRC).build().unwrap().launch([n as u32], args![&mut out]).unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![0u32; n]);
}

#[test]
fn storage_atomics() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> a: array<atomic<u32>>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let x = gid.x;
        if (x >= 4096u) { return; }
        atomicAdd(&a[0], x);
        atomicMax(&a[1], x * 7u % 4099u);
        atomicMin(&a[2], (x * 13u + 5u) % 4096u + 3u);
        atomicOr(&a[3], 1u << (x % 32u));
        atomicExchange(&a[4], x + 1u);
        let r = atomicCompareExchangeWeak(&a[5], 0u, 1u);
        if (r.exchanged) { atomicAdd(&a[6], 1u); }
    }"#;
    let mut a = one_d(&[0u32, 0, u32::MAX, 0, 0, 0, 0, 0]);
    ctx().kernel(SRC).build().unwrap().launch([4096], args![&mut a]).unwrap();
    let got = a.to_vec().unwrap();
    let xs = 0..4096u32;
    assert_eq!(got[0], xs.clone().sum::<u32>());
    assert_eq!(got[1], xs.clone().map(|x| x * 7 % 4099).max().unwrap());
    assert_eq!(got[2], xs.clone().map(|x| (x * 13 + 5) % 4096 + 3).min().unwrap());
    assert_eq!(got[3], u32::MAX);
    assert!((1..=4096).contains(&got[4]), "exchange left {}", got[4]);
    assert_eq!(got[5], 1);
    assert_eq!(got[6], 1, "exactly one invocation takes the lock");
    assert_eq!(got[7], 0);
}
