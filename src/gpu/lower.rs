//! Building kernels as `luma_core::kir` modules without printing source text.
//!
//! Luma's pure kernels (elementwise programs, stencils, the simple built-ins) are constructed
//! with [`ModuleBuilder`]. They still call pure helper functions that are easier to keep as WGSL
//! (the double-float library, the generators, the edge-handling functions): a [`Library`] lowers
//! such WGSL once (see [`library`]) and [`finish`] attaches the functions to the built module.
//! While building, a helper is called through [`Library::call`], which names it with a virtual
//! function id that [`finish`] renumbers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use luma_core::kir::{
    walk_stmts, Expr, FnBuilder, FnId, IrUnOp, KernelModule, KirFunction, KirType, ModuleBuilder, Stmt, StructDecl, Val,
};
use luma_core::naga::{self, TypeInner};
use luma_core::{BackendError, Features};

use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;
use crate::gpu::lock;

/// Virtual function ids of a [`Library`] start here; [`finish`] maps them to real ones.
const LIBRARY_BASE: u32 = 1 << 24;

/// Pure WGSL helper functions lowered to KIR.
#[derive(Debug)]
pub(crate) struct Library {
    functions: Vec<KirFunction>,
    /// The structs the functions use; their type indices are the library's own.
    structs: Vec<StructDecl>,
    ids: HashMap<String, u32>,
}

impl Library {
    /// The helper `name` called with `args`; the value has the helper's return type.
    ///
    /// Panics if the library has no such function or it returns nothing: the helpers a generator
    /// names are written next to it.
    pub(crate) fn call(&self, b: &FnBuilder<'_>, name: &str, args: &[Val]) -> Val {
        let (id, ret) = self.lookup(name);
        b.call(id, args, ret.unwrap_or_else(|| panic!("library function `{name}` returns nothing")))
    }


    /// Whether the library defines `name`.
    pub(crate) fn has(&self, name: &str) -> bool {
        self.ids.contains_key(name)
    }

    /// The id of `name` and its return type; a struct in it is named by a virtual index (the
    /// library's index plus [`LIBRARY_BASE`]), which [`finish`] maps to the attached struct.
    fn lookup(&self, name: &str) -> (FnId, Option<KirType>) {
        let index = *self.ids.get(name).unwrap_or_else(|| panic!("library has no function `{name}`"));
        let ret = self.functions[index as usize].ret_ty.clone().map(|mut ty| {
            remap_type(&mut ty, &|id| if id == u32::MAX { id } else { LIBRARY_BASE + id });
            ty
        });
        (FnId(LIBRARY_BASE + index), ret)
    }
}

/// The text of the zero value of `ty` as a WGSL constructor, for the stub that calls a helper.
fn zero_value(module: &naga::Module, ty: naga::Handle<naga::Type>) -> std::result::Result<String, String> {
    let scalar = |s: &naga::Scalar| match (s.kind, s.width) {
        (naga::ScalarKind::Float, 4) => Ok("f32"),
        (naga::ScalarKind::Float, 8) => Ok("f64"),
        (naga::ScalarKind::Sint, 4) => Ok("i32"),
        (naga::ScalarKind::Uint, 4) => Ok("u32"),
        (naga::ScalarKind::Bool, _) => Ok("bool"),
        other => Err(format!("library scalar {other:?}")),
    };
    match &module.types[ty].inner {
        TypeInner::Scalar(s) => Ok(format!("{}()", scalar(s)?)),
        TypeInner::Vector { size, scalar: s } => Ok(format!("vec{}<{}>()", *size as u32, scalar(s)?)),
        TypeInner::Struct { .. } => Ok(format!("{}()", module.types[ty].name.as_deref().ok_or("an unnamed struct")?)),
        other => Err(format!("library parameter of type {other:?}")),
    }
}

fn lower(wgsl: &str) -> std::result::Result<Library, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
    let mut calls = String::new();
    for (_, function) in module.functions.iter() {
        let Some(name) = function.name.as_deref() else { continue };
        let args = function
            .arguments
            .iter()
            .map(|arg| zero_value(&module, arg.ty))
            .collect::<std::result::Result<Vec<_>, _>>()?
            .join(", ");
        let discard = if function.result.is_some() { "_ = " } else { "" };
        calls.push_str(&format!("    {discard}{name}({args});\n"));
    }
    let source = format!("{wgsl}\n@compute @workgroup_size(1)\nfn luma_library() {{\n{calls}}}\n");
    let features = Features::SHADER_F64 | Features::SHADER_F16;
    let (_, lowered, ..) =
        luma_core::kir::parse_wgsl(&source, "luma_library", None, features, [256; 3], 256, 64, true).map_err(|e| e.to_string())?;
    let ids = lowered.functions.iter().enumerate().map(|(i, f)| (f.name.clone(), i as u32)).collect();
    Ok(Library { functions: lowered.functions, structs: lowered.structs, ids })
}

/// The functions of the WGSL `wgsl` (pure helpers: no bindings, no `luma` grid), lowered once per
/// distinct text. Their calls of each other are resolved; functions call each other freely.
pub(crate) fn library(wgsl: &str) -> Result<Arc<Library>> {
    static CACHE: Mutex<Option<FxMap<String, Arc<Library>>>> = Mutex::new(None);
    if let Some(known) = lock(&CACHE).as_ref().and_then(|cache| cache.get(wgsl)) {
        return Ok(Arc::clone(known));
    }
    let built = Arc::new(lower(wgsl).map_err(|message| LumaError::from(BackendError::ShaderCompile(message)))?);
    lock(&CACHE).get_or_insert_with(FxMap::default).insert(wgsl.to_owned(), Arc::clone(&built));
    Ok(built)
}

/// Shifts every call of a function with id `>= LIBRARY_BASE` by `to` and every other by `0`.
fn renumber(block: &mut [Stmt], f: &dyn Fn(u32) -> u32) {
    luma_core::kir::rewrite_exprs(block, &mut |e| {
        if let Expr::Call { func, .. } = e {
            func.0 = f(func.0);
        }
    });
}

/// `ty` with every struct index in it (also inside arrays) mapped by `f`.
fn remap_type(ty: &mut KirType, f: &dyn Fn(u32) -> u32) {
    match ty {
        KirType::Struct(index) => *index = f(*index),
        KirType::Array { elem, .. } => remap_type(elem, f),
        _ => {}
    }
}

/// The struct indices in the types an expression node carries, mapped by `f`.
fn remap_expr(e: &mut Expr, f: &dyn Fn(u32) -> u32) {
    match e {
        Expr::ZeroVal(ty)
        | Expr::UniformRead { ty, .. }
        | Expr::Load { ty, .. }
        | Expr::StorageRead { ty, .. }
        | Expr::SharedLoad { ty, .. }
        | Expr::SharedRead { ty, .. }
        | Expr::Access { ty, .. }
        | Expr::Swizzle { ty, .. }
        | Expr::Compose { ty, .. }
        | Expr::Unary { ty, .. }
        | Expr::Binary { ty, .. }
        | Expr::Ternary { ty, .. }
        | Expr::Subgroup { ty, .. } => remap_type(ty, f),
        Expr::Cast { src_ty, dst_ty, .. } => {
            remap_type(src_ty, f);
            remap_type(dst_ty, f);
        }
        Expr::Call { ty: Some(ty), .. } => remap_type(ty, f),
        _ => {}
    }
}

/// [`remap_expr`] on every expression of `block`.
fn remap_block(block: &mut [Stmt], f: &dyn Fn(u32) -> u32) {
    luma_core::kir::rewrite_exprs(block, &mut |e| remap_expr(e, f));
}

/// The struct indices of a function's parameters, result, locals and body, mapped by `f`.
fn remap_function(function: &mut KirFunction, f: &dyn Fn(u32) -> u32) {
    for (_, ty) in &mut function.params {
        remap_type(ty, f);
    }
    if let Some(ty) = &mut function.ret_ty {
        remap_type(ty, f);
    }
    for (_, ty, _) in &mut function.locals {
        remap_type(ty, f);
    }
    remap_block(&mut function.body, f);
}

/// The finished module of `mb`, with the functions of `library` it calls attached (the ones
/// nothing calls are dropped) and `uses_exact_f32` set when the module relies on exact
/// arithmetic (an [`IrUnOp::OpaqueGuard`], or the library's `luma_o` / `luma_bits`).
pub(crate) fn finish(mb: ModuleBuilder, library: Option<&Library>) -> KernelModule {
    let mut m = mb.finish();
    if let Some(library) = library {
        let own = m.functions.len() as u32;
        let off = m.structs.len() as u32;
        let map = move |id: u32| if id >= LIBRARY_BASE { id - LIBRARY_BASE + own } else { id };
        // A struct the module's own code got from a library call is a virtual index.
        let own_ty = move |id: u32| if id != u32::MAX && id >= LIBRARY_BASE { id - LIBRARY_BASE + off } else { id };
        let library_ty = move |id: u32| if id == u32::MAX { id } else { id + off };
        renumber(&mut m.entry_body, &map);
        remap_block(&mut m.entry_body, &own_ty);
        for (_, ty, _) in &mut m.entry_locals {
            remap_type(ty, &own_ty);
        }
        for private in &mut m.privates {
            remap_type(&mut private.ty, &own_ty);
            if let Some(init) = &mut private.init {
                init.walk_mut(&mut |e| remap_expr(e, &own_ty));
            }
        }
        for binding in &mut m.bindings {
            remap_type(&mut binding.elem_ty, &own_ty);
        }
        for shared in &mut m.shared {
            remap_type(&mut shared.elem_ty, &own_ty);
        }
        for f in &mut m.functions {
            renumber(&mut f.body, &map);
            remap_function(f, &own_ty);
        }
        for decl in &library.structs {
            let mut decl = decl.clone();
            for field in &mut decl.fields {
                remap_type(&mut field.ty, &library_ty);
            }
            m.structs.push(decl);
        }
        for f in &library.functions {
            let mut f = f.clone();
            renumber(&mut f.body, &move |id| id + own);
            remap_function(&mut f, &library_ty);
            m.functions.push(f);
        }
        // Drop what nothing calls and renumber the rest.
        let live = m.reachable_functions();
        if live.iter().any(|live| !live) {
            let mut remap = vec![u32::MAX; live.len()];
            let mut next = 0;
            for (old, keep) in live.iter().enumerate() {
                if *keep {
                    remap[old] = next;
                    next += 1;
                }
            }
            let mut keep = live.iter();
            m.functions.retain(|_| *keep.next().expect("one flag per function"));
            let to_new = move |id: u32| remap[id as usize];
            renumber(&mut m.entry_body, &to_new);
            for f in &mut m.functions {
                renumber(&mut f.body, &to_new);
            }
        }
    }
    let mut exact = m.uses_exact_f32 || m.functions.iter().any(|f| matches!(f.name.as_str(), "luma_o" | "luma_bits"));
    if !exact {
        let mut scan = |block: &[Stmt]| {
            walk_stmts(block, &mut |s| {
                s.for_each_expr(&mut |e| {
                    e.walk(&mut |x| {
                        if matches!(x, Expr::Unary { op: IrUnOp::OpaqueGuard, .. }) {
                            exact = true;
                        }
                    })
                })
            });
        };
        scan(&m.entry_body);
        m.functions.iter().for_each(|f| scan(&f.body));
    }
    m.uses_exact_f32 = exact;
    m
}
