//! `AnyArray`, `Dyn` and `ElemKind`: arrays that forget their element type and get it back.

use luma::*;

fn describe(args: &[&dyn AnyArray]) -> Vec<String> {
    args.iter().map(|a| format!("{} {}", a.elem(), a.shape())).collect()
}

#[test]
fn arrays_of_every_type_are_any_array_and_say_what_they_hold() {
    let ctx = Context::new().unwrap();
    let f = Array::<f32>::zeros_on(&ctx, [2, 3]).unwrap();
    let u = Array::<u32>::zeros_on(&ctx, 4).unwrap();
    let i = Array::<i32>::zeros_on(&ctx, 5).unwrap();
    let b = Array::<bool>::zeros_on(&ctx, 6).unwrap();
    let u8s = Array::<u8>::zeros_on(&ctx, 7).unwrap();
    let i8s = Array::<i8>::zeros_on(&ctx, 8).unwrap();
    let u16s = Array::<u16>::zeros_on(&ctx, 9).unwrap();
    let i16s = Array::<i16>::zeros_on(&ctx, 10).unwrap();
    let h = Array::<f16>::zeros_on(&ctx, 11).unwrap();
    let bh = Array::<bf16>::zeros_on(&ctx, 12).unwrap();
    let d = Array::<Df32>::zeros_on(&ctx, 13).unwrap();
    let c = Array::<C32>::zeros_on(&ctx, 14).unwrap();
    let v = Array::<[f32; 4]>::zeros_on(&ctx, 15).unwrap();
    let got = describe(&[&f, &u, &i, &b, &u8s, &i8s, &u16s, &i16s, &h, &bh, &d, &c, &v]);
    let want = [
        "f32 [2, 3, 1, 1]",
        "u32 [4, 1, 1, 1]",
        "i32 [5, 1, 1, 1]",
        "bool [6, 1, 1, 1]",
        "u8 [7, 1, 1, 1]",
        "i8 [8, 1, 1, 1]",
        "u16 [9, 1, 1, 1]",
        "i16 [10, 1, 1, 1]",
        "f16 [11, 1, 1, 1]",
        "bf16 [12, 1, 1, 1]",
        "df32 [13, 1, 1, 1]",
        "c32 [14, 1, 1, 1]",
        "[f32; 4] [15, 1, 1, 1]",
    ];
    assert_eq!(got, want);
    assert_eq!(AnyArray::elem(&f), ElemKind::F32);
    assert!(std::ptr::eq(AnyArray::context(&f), f.context()));
}

#[test]
fn a_dyn_gives_the_array_back_with_its_data() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1u32, 2, 3, 4], [2, 2]).unwrap();
    let dynamic: Dyn = a.clone().into();
    assert_eq!(dynamic.elem(), ElemKind::U32);
    assert_eq!(dynamic.shape(), Shape::from([2, 2]));
    let back = dynamic.take::<u32>().unwrap();
    assert_eq!(back.to_vec().unwrap(), [1, 2, 3, 4]);
    assert_eq!(back.shape(), a.shape());
}

#[test]
fn taking_the_wrong_type_says_what_the_array_holds() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    let message = Dyn::from(a.clone()).take::<u32>().unwrap_err().to_string();
    assert!(message.contains("holds f32, not u32"), "{message}");
    let message = Dyn::from(a).take::<[f32; 4]>().unwrap_err().to_string();
    assert!(message.contains("holds f32, not [f32; 4]"), "{message}");
}

#[test]
fn a_dyn_is_an_any_array_and_a_clone_shares_the_data() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[5i32, 6, 7], 3).unwrap();
    let first = Dyn::from(a);
    let second = first.clone();
    assert_eq!(describe(&[&first, &second]), ["i32 [3, 1, 1, 1]", "i32 [3, 1, 1, 1]"]);
    assert_eq!(first.take::<i32>().unwrap().to_vec().unwrap(), [5, 6, 7]);
    assert_eq!(second.take::<i32>().unwrap().to_vec().unwrap(), [5, 6, 7]);
}

#[test]
fn a_lazy_array_keeps_its_expression_behind_a_dyn() {
    let ctx = Context::new().unwrap();
    let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    let lazy = &x * 2.0 + 1.0;
    let dynamic: Dyn = lazy.into();
    assert_eq!(dynamic.take::<f32>().unwrap().to_vec().unwrap(), [3.0, 5.0, 7.0]);
}

#[test]
fn element_kinds_print_their_rust_names() {
    let shown: Vec<String> = [ElemKind::F32, ElemKind::U32, ElemKind::I32, ElemKind::Bool, ElemKind::Bf16, ElemKind::Df32, ElemKind::C32]
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(shown, ["f32", "u32", "i32", "bool", "bf16", "df32", "c32"]);
}
