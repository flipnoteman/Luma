//! The functions of `Context::elementwise_fn`: a process-wide table of WGSL scalar functions that [`OpKind::Call`] names by id.
//!
//! The id is interned by (name, source), so two contexts that register the same function share it, and two programs that call
//! different functions never share a kernel (the id is part of the [`Op`](super::Op), which every key hashes). The source of every
//! function a kernel calls is appended to the kernel's helper library, in id order, so that the kernel text (and with it the
//! on-disk kernel cache key) is deterministic.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};

use super::{Instr, OpKind};
use crate::error::{LumaError, Result};
use crate::gpu::hash::FxMap;

/// Ids are `u16`: the table holds at most this many functions.
const MAX_FUNCTIONS: usize = u16::MAX as usize;

/// What the cost model charges one call, in f32 operations: about a mid-sized special function.
pub(crate) const DEFAULT_WEIGHT: f32 = 4.0;

/// A registered function.
pub(crate) struct FnDef {
    /// The function's name in `source`.
    pub(crate) name: String,
    /// WGSL: the function and its helpers, all named `name` or `name_*`.
    pub(crate) source: String,
    /// Operands: 1 to 3.
    pub(crate) arity: u8,
    /// The cost model's weight of one call, as `f32` bits.
    weight: AtomicU32,
}

impl FnDef {
    /// The weight of one call in the cost model (f32 operations).
    pub(crate) fn weight(&self) -> f32 {
        f32::from_bits(self.weight.load(Ordering::Relaxed))
    }

    pub(crate) fn set_weight(&self, weight: f32) {
        self.weight.store(weight.to_bits(), Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Table {
    defs: Vec<Arc<FnDef>>,
    by_key: FxMap<(String, String), u16>,
}

fn table() -> &'static RwLock<Table> {
    static TABLE: OnceLock<RwLock<Table>> = OnceLock::new();
    TABLE.get_or_init(RwLock::default)
}

/// The id of the function `name` defined by `source` (the same id for the same pair), registering it first if need be.
pub(crate) fn register(name: &str, source: &str, arity: u8) -> Result<u16> {
    let key = (name.to_owned(), source.to_owned());
    if let Some(&id) = table().read().unwrap_or_else(PoisonError::into_inner).by_key.get(&key) {
        return Ok(id);
    }
    let mut table = table().write().unwrap_or_else(PoisonError::into_inner);
    if let Some(&id) = table.by_key.get(&key) {
        return Ok(id);
    }
    if table.defs.len() >= MAX_FUNCTIONS {
        return Err(LumaError::InvalidShape("too many elementwise functions".into()));
    }
    let id = table.defs.len() as u16;
    table.defs.push(Arc::new(FnDef { name: name.to_owned(), source: source.to_owned(), arity, weight: AtomicU32::new(DEFAULT_WEIGHT.to_bits()) }));
    table.by_key.insert(key, id);
    Ok(id)
}

/// The function `id`, which `register` returned.
pub(crate) fn def(id: u16) -> Arc<FnDef> {
    Arc::clone(&table().read().unwrap_or_else(PoisonError::into_inner).defs[id as usize])
}

/// The name of function `id` in the generated code.
pub(crate) fn fn_name(id: u16) -> String {
    def(id).name.clone()
}

/// The source of every function the instructions of `programs` call, once each, in id order.
pub(super) fn helpers<'a>(programs: impl IntoIterator<Item = &'a [Instr]>) -> String {
    let ids: BTreeSet<u16> = programs
        .into_iter()
        .flatten()
        .filter_map(|instr| match instr.op.kind {
            OpKind::Call { id, .. } => Some(id),
            _ => None,
        })
        .collect();
    let mut text = String::new();
    for id in ids {
        text.push_str(&def(id).source);
        text.push('\n');
    }
    text
}
