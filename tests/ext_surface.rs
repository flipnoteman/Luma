//! Names every item of the extension API (`tools/public-api/ext.txt`): a rename or removal fails to compile.

use luma::ext::{params_bytes, ArgInfo, Bind, Fit, Lower, NumericsClass, Op, OpBuilder, OutShape, ScratchId, Strategy};
use luma::{elementwise_fn, AnyArray, ElemFnType, ElementwiseFn, Context, DecisionSite, Dyn, ElemKind, Features, Kernel, LumaError, Reach, Result, Shape};

#[allow(dead_code, clippy::type_complexity)]
fn surface(ctx: &Context, kernel: &Kernel, x: &dyn AnyArray) -> Result<()> {
    let _: Reach = kernel.reach();
    let _ = ctx.kernel("").elementwise();
    let _ = ctx.kernel("").halo(1);
    let reach = match Reach::Whole {
        Reach::Whole | Reach::Elementwise | Reach::Halo { .. } => 0,
        _ => 1,
    };
    let _ = (reach, DecisionSite::Extension);
    let _ = [LumaError::Declined];
    let _: fn(usize, String) -> LumaError = |index, message| LumaError::KernelReach { index, message };
    let _: fn(String, String, String) -> LumaError = |op, strategy, message| LumaError::Extension { op, strategy, message };
    let _: fn(String, Vec<(String, String)>) -> LumaError = |op, reasons| LumaError::NoStrategy { op, reasons };

    let strategy: Strategy = Strategy::new("s", |lower: &mut Lower<'_>| {
        let _: &[ArgInfo] = lower.inputs();
        let _: &[ArgInfo] = lower.outputs();
        let _: &[u8] = lower.params();
        let id: ScratchId = lower.scratch(16)?;
        let kernel = lower.kernel("").build()?;
        lower.launch(&kernel, [1u32, 1, 1], &[Bind::Input(0), Bind::Output(0), Bind::Scratch(id), Bind::Uniform(vec![])])
    })
    .needs(Features::NONE)
    .numerics(NumericsClass::Exact)
    .numerics(NumericsClass::Reassociating)
    .numerics(NumericsClass::Reduced)
    .reach(Reach::Elementwise)
    .fits(|fit: &Fit<'_>| fit.inputs().len() == fit.outputs().len() && fit.params().is_empty() && fit.features() == Features::NONE && fit.numerics().reassociate)
    .cost(|fit: &Fit<'_>| fit.cost_model().launch_us);
    let builder: OpBuilder<'_> = ctx
        .custom_op("surface")
        .input::<f32>()
        .output::<f32>(OutShape::Same(0))
        .output::<f32>(OutShape::Fixed(Shape::from(1)))
        .output::<f32>(OutShape::with(|inputs: &[ArgInfo], _params| Ok(inputs[0].shape)))
        .params(0)
        .strategy(strategy);
    let op: Op = builder.build()?;
    let _: &str = op.name();
    let _: Vec<Dyn> = op.apply(&[x], &[])?;
    let _ = op.apply1::<f32>(&[x], &[])?;
    let _ = ArgInfo::new(ElemKind::F32, Shape::from(1));
    let _ = params_bytes(&1.0f32)?;
    let f: ElementwiseFn<f32> = ctx.elementwise_fn::<f32>("surface_f", "fn surface_f(x: f32) -> f32 { return x; }")?;
    let g: ElementwiseFn<u32> = elementwise_fn::<u32>("surface_g", "fn surface_g(x: u32) -> u32 { return x; }")?;
    let _ = (f.clone().weight(2.0).name(), g.name());
    let a = luma::Array::<f32>::zeros(1)?;
    let _ = (f.call1(&a), f.call2(&a, 1.0f32), f.call3(&a, &a, 1.0f32));
    fn takes<T: ElemFnType>() {}
    takes::<f32>();
    takes::<u32>();
    takes::<i32>();
    Ok(())
}

#[test]
fn the_surface_compiles() {}
