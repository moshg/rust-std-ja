//! This module specifies the type based interner for constants.
//!
//! After a const evaluation has computed a value, before we destroy the const evaluator's session
//! memory, we need to extract all memory allocations to the global memory pool so they stay around.

use rustc::ty::{Ty, ParamEnv, self};
use rustc::mir::interpret::{InterpResult, ErrorHandled};
use rustc::hir;
use rustc::hir::def_id::DefId;
use super::validity::RefTracking;
use rustc_data_structures::fx::FxHashSet;

use syntax::ast::Mutability;

use super::{
    ValueVisitor, MemoryKind, AllocId, MPlaceTy, Scalar,
};
use crate::const_eval::{CompileTimeInterpreter, CompileTimeEvalContext};

struct InternVisitor<'rt, 'mir, 'tcx> {
    /// previously encountered safe references
    ref_tracking: &'rt mut RefTracking<(MPlaceTy<'tcx>, Mutability, InternMode)>,
    ecx: &'rt mut CompileTimeEvalContext<'mir, 'tcx>,
    param_env: ParamEnv<'tcx>,
    /// The root node of the value that we're looking at. This field is never mutated and only used
    /// for sanity assertions that will ICE when `const_qualif` screws up.
    mode: InternMode,
    /// This field stores the mutability of the value *currently* being checked.
    /// When encountering a mutable reference, we determine the pointee mutability
    /// taking into account the mutability of the context: `& &mut i32` is entirely immutable,
    /// despite the nested mutable reference!
    /// The field gets updated when an `UnsafeCell` is encountered.
    mutability: Mutability,
    /// A list of all encountered relocations. After type-based interning, we traverse this list to
    /// also intern allocations that are only referenced by a raw pointer or inside a union.
    leftover_relocations: &'rt mut FxHashSet<AllocId>,
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
enum InternMode {
    /// Mutable references must in fact be immutable due to their surrounding immutability in a
    /// `static`. In a `static mut` we start out as mutable and thus can also contain further `&mut`
    /// that will actually be treated as mutable.
    Static,
    /// UnsafeCell is OK in the value of a constant: `const FOO = Cell::new(0)` creates
    /// a new cell every time it is used.
    ConstBase,
    /// `UnsafeCell` ICEs.
    Const,
}

/// Signalling data structure to ensure we don't recurse
/// into the memory of other constants or statics
struct IsStaticOrFn;

impl<'rt, 'mir, 'tcx> InternVisitor<'rt, 'mir, 'tcx> {
    /// Intern an allocation without looking at its children.
    /// `mutablity` is the mutability of the place to be interned; even if that says
    /// `immutable` things might become mutable if `ty` is not frozen.
    fn intern_shallow(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
        ty: Option<Ty<'tcx>>,
    ) -> InterpResult<'tcx, Option<IsStaticOrFn>> {
        trace!(
            "InternVisitor::intern {:?} with {:?}",
            alloc_id, mutability,
        );
        // remove allocation
        let tcx = self.ecx.tcx;
        let memory = self.ecx.memory_mut();
        let (kind, mut alloc) = match memory.alloc_map.remove(&alloc_id) {
            Some(entry) => entry,
            None => {
                // Pointer not found in local memory map. It is either a pointer to the global
                // map, or dangling.
                // If the pointer is dangling (neither in local nor global memory), we leave it
                // to validation to error. The `delay_span_bug` ensures that we don't forget such
                // a check in validation.
                if tcx.alloc_map.lock().get(alloc_id).is_none() {
                    tcx.sess.delay_span_bug(self.ecx.tcx.span, "tried to intern dangling pointer");
                }
                // treat dangling pointers like other statics
                // just to stop trying to recurse into them
                return Ok(Some(IsStaticOrFn));
            },
        };
        // This match is just a canary for future changes to `MemoryKind`, which most likely need
        // changes in this function.
        match kind {
            MemoryKind::Stack | MemoryKind::Vtable => {},
        }
        // Set allocation mutability as appropriate. This is used by LLVM to put things into
        // read-only memory, and also by Miri when evluating other constants/statics that
        // access this one.
        if self.mode == InternMode::Static {
            let frozen = ty.map_or(true, |ty| ty.is_freeze(
                self.ecx.tcx.tcx,
                self.param_env,
                self.ecx.tcx.span,
            ));
            // For statics, allocation mutability is the combination of the place mutability and
            // the type mutability.
            // The entire allocation needs to be mutable if it contains an `UnsafeCell` anywhere.
            if mutability == Mutability::Immutable && frozen {
                alloc.mutability = Mutability::Immutable;
            } else {
                // Just making sure we are not "upgrading" an immutable allocation to mutable.
                assert_eq!(alloc.mutability, Mutability::Mutable);
            }
        } else {
            // We *could* be non-frozen at `ConstBase`, for constants like `Cell::new(0)`.
            // But we still intern that as immutable as the memory cannot be changed once the
            // initial value was computed.
            // Constants are never mutable.
            alloc.mutability = Mutability::Immutable;
        };
        // link the alloc id to the actual allocation
        let alloc = tcx.intern_const_alloc(alloc);
        self.leftover_relocations.extend(alloc.relocations().iter().map(|&(_, ((), reloc))| reloc));
        tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
        Ok(None)
    }
}

impl<'rt, 'mir, 'tcx>
    ValueVisitor<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>
for
    InternVisitor<'rt, 'mir, 'tcx>
{
    type V = MPlaceTy<'tcx>;

    #[inline(always)]
    fn ecx(&self) -> &CompileTimeEvalContext<'mir, 'tcx> {
        &self.ecx
    }

    fn visit_aggregate(
        &mut self,
        mplace: MPlaceTy<'tcx>,
        fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
    ) -> InterpResult<'tcx> {
        if let Some(def) = mplace.layout.ty.ty_adt_def() {
            if Some(def.did) == self.ecx.tcx.lang_items().unsafe_cell_type() {
                // We are crossing over an `UnsafeCell`, we can mutate again. This means that
                // References we encounter inside here are interned as pointing to mutable
                // allocations.
                let old = std::mem::replace(&mut self.mutability, Mutability::Mutable);
                assert_ne!(
                    self.mode, InternMode::Const,
                    "UnsafeCells are not allowed behind references in constants. This should have \
                    been prevented statically by const qualification. If this were allowed one \
                    would be able to change a constant at one use site and other use sites could \
                    observe that mutation.",
                );
                let walked = self.walk_aggregate(mplace, fields);
                self.mutability = old;
                return walked;
            }
        }
        self.walk_aggregate(mplace, fields)
    }

    fn visit_primitive(&mut self, mplace: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        // Handle Reference types, as these are the only relocations supported by const eval.
        // Raw pointers (and boxes) are handled by the `leftover_relocations` logic.
        let ty = mplace.layout.ty;
        if let ty::Ref(_, referenced_ty, mutability) = ty.sty {
            let value = self.ecx.read_immediate(mplace.into())?;
            // Handle trait object vtables
            if let Ok(meta) = value.to_meta() {
                if let ty::Dynamic(..) =
                    self.ecx.tcx.struct_tail_erasing_lifetimes(referenced_ty, self.param_env).sty
                {
                    if let Ok(vtable) = meta.unwrap().to_ptr() {
                        // explitly choose `Immutable` here, since vtables are immutable, even
                        // if the reference of the fat pointer is mutable
                        self.intern_shallow(vtable.alloc_id, Mutability::Immutable, None)?;
                    }
                }
            }
            let mplace = self.ecx.ref_to_mplace(value)?;
            // Check if we have encountered this pointer+layout combination before.
            // Only recurse for allocation-backed pointers.
            if let Scalar::Ptr(ptr) = mplace.ptr {
                // We do not have any `frozen` logic here, because it's essentially equivalent to
                // the mutability except for the outermost item. Only `UnsafeCell` can "unfreeze",
                // and we check that in `visit_aggregate`.
                // This is not an inherent limitation, but one that we know to be true, because
                // const qualification enforces it. We can lift it in the future.
                match (self.mode, mutability) {
                    // immutable references are fine everywhere
                    (_, hir::Mutability::MutImmutable) => {},
                    // all is "good and well" in the unsoundness of `static mut`

                    // mutable references are ok in `static`. Either they are treated as immutable
                    // because they are behind an immutable one, or they are behind an `UnsafeCell`
                    // and thus ok.
                    (InternMode::Static, hir::Mutability::MutMutable) => {},
                    // we statically prevent `&mut T` via `const_qualif` and double check this here
                    (InternMode::ConstBase, hir::Mutability::MutMutable) |
                    (InternMode::Const, hir::Mutability::MutMutable) => {
                        match referenced_ty.sty {
                            ty::Array(_, n)
                                if n.eval_usize(self.ecx.tcx.tcx, self.param_env) == 0 => {}
                            ty::Slice(_)
                                if value.to_meta().unwrap().unwrap().to_usize(self.ecx)? == 0 => {}
                            _ => bug!("const qualif failed to prevent mutable references"),
                        }
                    },
                }
                // Compute the mutability with which we'll start visiting the allocation. This is
                // what gets changed when we encounter an `UnsafeCell`
                let mutability = match (self.mutability, mutability) {
                    // The only way a mutable reference actually works as a mutable reference is
                    // by being in a `static mut` directly or behind another mutable reference.
                    // If there's an immutable reference or we are inside a static, then our
                    // mutable reference is equivalent to an immutable one. As an example:
                    // `&&mut Foo` is semantically equivalent to `&&Foo`
                    (Mutability::Mutable, hir::Mutability::MutMutable) => Mutability::Mutable,
                    _ => Mutability::Immutable,
                };
                // Recursing behind references changes the intern mode for constants in order to
                // cause assertions to trigger if we encounter any `UnsafeCell`s.
                let mode = match self.mode {
                    InternMode::ConstBase => InternMode::Const,
                    other => other,
                };
                match self.intern_shallow(ptr.alloc_id, mutability, Some(mplace.layout.ty))? {
                    // No need to recurse, these are interned already and statics may have
                    // cycles, so we don't want to recurse there
                    Some(IsStaticOrFn) => {},
                    // intern everything referenced by this value. The mutability is taken from the
                    // reference. It is checked above that mutable references only happen in
                    // `static mut`
                    None => self.ref_tracking.track((mplace, mutability, mode), || ()),
                }
            }
        }
        Ok(())
    }
}

pub fn intern_const_alloc_recursive(
    ecx: &mut CompileTimeEvalContext<'mir, 'tcx>,
    def_id: DefId,
    ret: MPlaceTy<'tcx>,
    // FIXME(oli-obk): can we scrap the param env? I think we can, the final value of a const eval
    // must always be monomorphic, right?
    param_env: ty::ParamEnv<'tcx>,
) -> InterpResult<'tcx> {
    let tcx = ecx.tcx;
    // this `mutability` is the mutability of the place, ignoring the type
    let (base_mutability, base_intern_mode) = match tcx.static_mutability(def_id) {
        Some(hir::Mutability::MutImmutable) => (Mutability::Immutable, InternMode::Static),
        None => (Mutability::Immutable, InternMode::ConstBase),
        // `static mut` doesn't care about interior mutability, it's mutable anyway
        Some(hir::Mutability::MutMutable) => (Mutability::Mutable, InternMode::Static),
    };

    // type based interning
    let mut ref_tracking = RefTracking::new((ret, base_mutability, base_intern_mode));
    let leftover_relocations = &mut FxHashSet::default();

    // start with the outermost allocation
    InternVisitor {
        ref_tracking: &mut ref_tracking,
        ecx,
        mode: base_intern_mode,
        leftover_relocations,
        param_env,
        mutability: base_mutability,
    }.intern_shallow(ret.ptr.to_ptr()?.alloc_id, base_mutability, Some(ret.layout.ty))?;

    while let Some(((mplace, mutability, mode), _)) = ref_tracking.todo.pop() {
        let interned = InternVisitor {
            ref_tracking: &mut ref_tracking,
            ecx,
            mode,
            leftover_relocations,
            param_env,
            mutability,
        }.visit_value(mplace);
        if let Err(error) = interned {
            // This can happen when e.g. the tag of an enum is not a valid discriminant. We do have
            // to read enum discriminants in order to find references in enum variant fields.
            if let err_unsup!(ValidationFailure(_)) = error.kind {
                let err = crate::const_eval::error_to_const_error(&ecx, error);
                match err.struct_error(ecx.tcx, "it is undefined behavior to use this value") {
                    Ok(mut diag) => {
                        diag.note(crate::const_eval::note_on_undefined_behavior_error());
                        diag.emit();
                    }
                    Err(ErrorHandled::TooGeneric) |
                    Err(ErrorHandled::Reported) => {},
                }
            }
        }
    }

    // Intern the rest of the allocations as mutable. These might be inside unions, padding, raw
    // pointers, ... So we can't intern them according to their type rules

    let mut todo: Vec<_> = leftover_relocations.iter().cloned().collect();
    while let Some(alloc_id) = todo.pop() {
        if let Some((_, mut alloc)) = ecx.memory_mut().alloc_map.remove(&alloc_id) {
            // We can't call the `intern_shallow` method here, as its logic is tailored to safe
            // references. So we hand-roll the interning logic here again.
            if base_intern_mode != InternMode::Static {
                // If it's not a static, it *must* be immutable.
                // We cannot have mutable memory inside a constant.
                alloc.mutability = Mutability::Immutable;
            }
            let alloc = tcx.intern_const_alloc(alloc);
            tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
            for &(_, ((), reloc)) in alloc.relocations().iter() {
                if leftover_relocations.insert(reloc) {
                    todo.push(reloc);
                }
            }
        } else if ecx.memory().dead_alloc_map.contains_key(&alloc_id) {
            // dangling pointer
            throw_unsup!(ValidationFailure("encountered dangling pointer in final constant".into()))
        }
    }
    Ok(())
}
