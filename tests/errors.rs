mod common;

use luma::*;

fn f32s(data: &[f32]) -> Array<f32> {
    Array::from_slice(data, data.len()).unwrap()
}

#[test]
fn shape_mismatch_mid_chain_surfaces_at_to_vec_with_the_location_of_the_faulty_line() {
    let a = f32s(&[1.0, 2.0, 3.0]);
    let b = f32s(&[1.0, 2.0]);

    let ok = &a * 2.0;
    let faulty_line = line!() + 1;
    let bad = &ok + &b; // shape mismatch: this is the line that has to be reported
    let c = (bad * 3.0).sqrt() - 1.0; // later operations just carry the poison along
    let d = &c + &a; // ... even where their own operands are fine

    // Nothing has waited for the GPU, but the failure is already visible.
    let early = d.error().expect("poisoned array must carry its error");
    assert!(matches!(early.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [2, 1, 1, 1] }));
    assert!(d.check().is_err());

    // It surfaces as an Err at every synchronization point, never as a panic.
    let err = d.to_vec().unwrap_err();
    assert!(
        matches!(err.root(), LumaError::ShapeMismatch { lhs: [3, 1, 1, 1], rhs: [2, 1, 1, 1] }),
        "{err}"
    );
    let location = err.location().expect("error must record the location of the failed op");
    assert!(location.file().ends_with("errors.rs"), "{}", location.file());
    assert_eq!(location.line(), faulty_line, "location must be the failing op, not a later one");
    assert!(location.column() > 0);

    let text = err.to_string();
    assert!(text.contains(&format!("errors.rs:{faulty_line}:")), "{text}");
    assert!(text.contains("add"), "{text}");
    assert!(text.contains("Dimension mismatch"), "{text}");

    assert!(matches!(d.eval().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    assert!(d.sum().is_err());
    assert!(d.min().is_err());
    assert!(d.max().is_err());
    assert!(d.mean().is_err());
    assert!(d.norm().is_err());

    // The arrays the failed op was built from are unharmed.
    assert_eq!(ok.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn poison_keeps_the_location_of_the_first_failure() {
    let a = f32s(&[1.0, 2.0, 3.0]);
    let b = f32s(&[1.0, 2.0]);
    let first = line!() + 1;
    let bad = &a - &b;
    let worse = &bad * &b; // different shapes again, but `bad` already failed
    let err = worse.to_vec().unwrap_err();
    assert_eq!(err.location().unwrap().line(), first);
}

#[test]
fn compound_assignment_poisons_the_left_hand_side() {
    let mut a = f32s(&[1.0, 2.0, 3.0]);
    let b = f32s(&[1.0, 2.0]);
    let line = line!() + 1;
    a += &b;
    let err = a.error().expect("a is poisoned");
    assert_eq!(err.location().unwrap().line(), line);
    assert!(matches!(a.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    // b is fine, and a poisoned lhs stays poisoned
    assert_eq!(b.to_vec().unwrap(), vec![1.0, 2.0]);
    a *= 2.0;
    assert_eq!(a.to_vec().unwrap_err().location().unwrap().line(), line);

    // A poisoned right-hand side poisons a healthy lhs.
    let mut x = f32s(&[1.0, 2.0]);
    let bad = &f32s(&[1.0]) + &f32s(&[1.0, 2.0]);
    x -= &bad;
    assert!(x.error().is_some());
    assert!(x.to_vec().is_err());
}

#[test]
fn try_methods_fail_immediately() {
    let a = f32s(&[1.0, 2.0, 3.0]);
    let b = f32s(&[1.0, 2.0]);

    for result in [a.try_add(&b), a.try_sub(&b), a.try_mul(&b), a.try_div(&b)] {
        let err = result.unwrap_err();
        assert!(matches!(err, LumaError::ShapeMismatch { .. }), "{err}");
        assert!(err.location().is_none(), "try_* returns the plain error");
    }

    // They work like the operators when nothing is wrong, with arrays or scalars.
    let c = f32s(&[10.0, 20.0, 30.0]);
    assert_eq!(a.try_add(&c).unwrap().to_vec().unwrap(), vec![11.0, 22.0, 33.0]);
    assert_eq!(a.try_sub(1.0).unwrap().to_vec().unwrap(), vec![0.0, 1.0, 2.0]);
    assert_eq!(c.try_mul(&a).unwrap().to_vec().unwrap(), vec![10.0, 40.0, 90.0]);
    assert_eq!(c.try_div(10.0).unwrap().to_vec().unwrap(), vec![1.0, 2.0, 3.0]);

    // A poisoned operand is reported by try_* too.
    let bad = &a + &b;
    let err = bad.try_add(&a).unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }));
}

#[test]
fn poisoned_arrays_reject_every_sink() {
    let a = f32s(&[1.0, 2.0, 3.0]);
    let mut bad = &a + &f32s(&[1.0]);

    assert!(bad.write(&[1.0, 2.0, 3.0]).is_err());
    assert!(bad.to_vec().is_err());

    // A poisoned array handed to a custom kernel is an Err at launch, with the original error.
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read_write> out: array<f32>;
    @compute @workgroup_size(1)
    fn main() { out[0] = 1.0; }"#;
    let kernel = Kernel::new(SRC).unwrap();
    let err = kernel.launch([1], args![&mut bad]).unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
    assert!(err.location().is_some());
}

#[test]
fn every_math_method_propagates_poison() {
    let bad = &f32s(&[1.0, 2.0]) + &f32s(&[1.0]);
    let outputs: Vec<Array<f32>> = vec![
        bad.sqrt(),
        bad.exp(),
        bad.ln(),
        bad.abs(),
        bad.sin(),
        bad.cos(),
        bad.tan(),
        bad.powf(2.0),
        bad.clamp(0.0, 1.0),
        bad.minimum(0.5),
        bad.maximum(&f32s(&[1.0, 2.0])),
        bad.shift(1),
        bad.reshape(2),
        -&bad,
        &bad * 2.0,
        2.0 - &bad,
        bad.conv1d(&f32s(&[1.0]), 1, 0),
        bad.conv1d_fft(&f32s(&[1.0])),
        select(&bad.gt(0.0), &bad, &bad),
    ];
    for (i, out) in outputs.iter().enumerate() {
        let err = out.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "output {i}: {err}");
    }
    assert!(bad.lt(1.0).to_vec().is_err());
    assert!(bad.eq(&bad).to_vec().is_err());
}

#[test]
fn reshape_with_the_wrong_element_count_is_poisoned() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let line = line!() + 1;
    let r = a.reshape([3, 2]);
    let err = r.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    assert_eq!(err.location().unwrap().line(), line);
    assert!(a.reshape([0, 4]).error().is_some());
    assert_eq!(a.reshape([2, 2]).to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arrays_that_share_a_buffer_cannot_be_written_and_read_in_one_operation() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let mut view = a.reshape([2, 2]);
    let same_shape = a.reshape([2, 2]);
    view += &same_shape; // would read and write one buffer in a single dispatch
    let err = view.to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::KernelArgAlias { .. }), "{err}");

    // The original is unharmed.
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn kernel_launch_rejects_aliased_arrays() {
    const SRC: &str = r#"
    @group(1) @binding(0) var<storage, read> src: array<f32>;
    @group(1) @binding(1) var<storage, read_write> dst: array<f32>;
    @compute @workgroup_size(4)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) { dst[gid.x] = src[gid.x]; }"#;
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let mut view = a.reshape([2, 2]);
    let err = Kernel::new(SRC).unwrap().launch([4], args![&a, &mut view]).unwrap_err();
    assert!(matches!(err, LumaError::KernelArgAlias { first: 0, second: 1 }), "{err}");
}

#[test]
fn default_context_is_shared_and_usable() {
    let first = default_context().unwrap();
    let second = default_context().unwrap();
    let a = Array::from_slice(&[1.0f32, 2.0], 2).unwrap();
    assert!(a.context().sync().is_ok());

    // Arrays on the default context and arrays created through a clone of it combine freely.
    let b = Array::from_slice_on(&first, &[3.0f32, 4.0], 2).unwrap();
    let c = Array::from_slice_on(&second, &[5.0f32, 6.0], 2).unwrap();
    assert_eq!((&a + &b + &c).to_vec().unwrap(), vec![9.0, 12.0]);

    // ... but not with arrays of another context.
    let other = Context::new().unwrap();
    let d = Array::from_slice_on(&other, &[1.0f32, 1.0], 2).unwrap();
    let err = (&a + &d).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ContextMismatch), "{err}");
}

#[test]
fn shape_conversions() {
    assert_eq!(Shape::from(5), Shape::new([5, 1, 1, 1]));
    assert_eq!(Shape::from([5]), Shape::new([5, 1, 1, 1]));
    assert_eq!(Shape::from([2, 3]), Shape::new([2, 3, 1, 1]));
    assert_eq!(Shape::from([2, 3, 4]), Shape::new([2, 3, 4, 1]));
    assert_eq!(Shape::from([2, 3, 4, 5]).len(), 120);
    assert_eq!(Shape::from([2, 3, 4, 5]).dims(), [2, 3, 4, 5]);
    let a = Array::<f32>::zeros([2, 3, 4]).unwrap();
    assert_eq!(a.len(), 24);
    assert_eq!(a.shape().dims(), [2, 3, 4, 1]);
}
