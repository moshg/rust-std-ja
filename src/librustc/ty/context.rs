// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! type context book-keeping

use dep_graph::DepGraph;
use dep_graph::{DepNode, DepConstructor};
use errors::DiagnosticBuilder;
use session::Session;
use session::config::{BorrowckMode, OutputFilenames};
use middle;
use hir::{TraitCandidate, HirId, ItemLocalId};
use hir::def::{Def, Export};
use hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use hir::map as hir_map;
use hir::map::DefPathHash;
use lint::{self, Lint};
use ich::{StableHashingContext, NodeIdHashingMode};
use infer::outlives::free_region_map::FreeRegionMap;
use middle::const_val::ConstVal;
use middle::cstore::{CrateStore, LinkMeta};
use middle::cstore::EncodedMetadata;
use middle::lang_items;
use middle::resolve_lifetime::{self, ObjectLifetimeDefault};
use middle::stability;
use mir::{self, Mir, interpret};
use mir::interpret::{Value, PrimVal};
use ty::subst::{Kind, Substs};
use ty::ReprOptions;
use ty::Instance;
use traits;
use ty::{self, Ty, TypeAndMut};
use ty::{TyS, TypeVariants, Slice};
use ty::{AdtKind, AdtDef, ClosureSubsts, GeneratorInterior, Region, Const};
use ty::{PolyFnSig, InferTy, ParamTy, ProjectionTy, ExistentialPredicate, Predicate};
use ty::RegionKind;
use ty::{TyVar, TyVid, IntVar, IntVid, FloatVar, FloatVid};
use ty::TypeVariants::*;
use ty::layout::{LayoutDetails, TargetDataLayout};
use ty::maps;
use ty::steal::Steal;
use ty::BindingMode;
use util::nodemap::{NodeMap, DefIdSet, ItemLocalMap};
use util::nodemap::{FxHashMap, FxHashSet};
use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::stable_hasher::{HashStable, hash_stable_hashmap,
                                           StableHasher, StableHasherResult,
                                           StableVec};
use arena::{TypedArena, DroplessArena};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::sync::Lrc;
use std::any::Any;
use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::hash_map::{self, Entry};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::iter;
use std::sync::mpsc;
use std::sync::Arc;
use syntax::abi;
use syntax::ast::{self, Name, NodeId};
use syntax::attr;
use syntax::codemap::MultiSpan;
use syntax::feature_gate;
use syntax::symbol::{Symbol, keywords};
use syntax_pos::Span;

use hir;

pub struct AllArenas<'tcx> {
    pub global: GlobalArenas<'tcx>,
    pub interner: DroplessArena,
}

impl<'tcx> AllArenas<'tcx> {
    pub fn new() -> Self {
        AllArenas {
            global: GlobalArenas::new(),
            interner: DroplessArena::new(),
        }
    }
}

/// Internal storage
pub struct GlobalArenas<'tcx> {
    // internings
    layout: TypedArena<LayoutDetails>,

    // references
    generics: TypedArena<ty::Generics>,
    trait_def: TypedArena<ty::TraitDef>,
    adt_def: TypedArena<ty::AdtDef>,
    steal_mir: TypedArena<Steal<Mir<'tcx>>>,
    mir: TypedArena<Mir<'tcx>>,
    tables: TypedArena<ty::TypeckTables<'tcx>>,
    /// miri allocations
    const_allocs: TypedArena<interpret::Allocation>,
}

impl<'tcx> GlobalArenas<'tcx> {
    pub fn new() -> GlobalArenas<'tcx> {
        GlobalArenas {
            layout: TypedArena::new(),
            generics: TypedArena::new(),
            trait_def: TypedArena::new(),
            adt_def: TypedArena::new(),
            steal_mir: TypedArena::new(),
            mir: TypedArena::new(),
            tables: TypedArena::new(),
            const_allocs: TypedArena::new(),
        }
    }
}

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc are allocated from
    arena: &'tcx DroplessArena,

    /// Specifically use a speedy hash algorithm for these hash sets,
    /// they're accessed quite often.
    type_: RefCell<FxHashSet<Interned<'tcx, TyS<'tcx>>>>,
    type_list: RefCell<FxHashSet<Interned<'tcx, Slice<Ty<'tcx>>>>>,
    substs: RefCell<FxHashSet<Interned<'tcx, Substs<'tcx>>>>,
    region: RefCell<FxHashSet<Interned<'tcx, RegionKind>>>,
    existential_predicates: RefCell<FxHashSet<Interned<'tcx, Slice<ExistentialPredicate<'tcx>>>>>,
    predicates: RefCell<FxHashSet<Interned<'tcx, Slice<Predicate<'tcx>>>>>,
    const_: RefCell<FxHashSet<Interned<'tcx, Const<'tcx>>>>,
}

impl<'gcx: 'tcx, 'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx DroplessArena) -> CtxtInterners<'tcx> {
        CtxtInterners {
            arena,
            type_: RefCell::new(FxHashSet()),
            type_list: RefCell::new(FxHashSet()),
            substs: RefCell::new(FxHashSet()),
            region: RefCell::new(FxHashSet()),
            existential_predicates: RefCell::new(FxHashSet()),
            predicates: RefCell::new(FxHashSet()),
            const_: RefCell::new(FxHashSet()),
        }
    }

    /// Intern a type. global_interners is Some only if this is
    /// a local interner and global_interners is its counterpart.
    fn intern_ty(&self, st: TypeVariants<'tcx>,
                 global_interners: Option<&CtxtInterners<'gcx>>)
                 -> Ty<'tcx> {
        let ty = {
            let mut interner = self.type_.borrow_mut();
            let global_interner = global_interners.map(|interners| {
                interners.type_.borrow_mut()
            });
            if let Some(&Interned(ty)) = interner.get(&st) {
                return ty;
            }
            if let Some(ref interner) = global_interner {
                if let Some(&Interned(ty)) = interner.get(&st) {
                    return ty;
                }
            }

            let flags = super::flags::FlagComputation::for_sty(&st);
            let ty_struct = TyS {
                sty: st,
                flags: flags.flags,
                region_depth: flags.depth,
            };

            // HACK(eddyb) Depend on flags being accurate to
            // determine that all contents are in the global tcx.
            // See comments on Lift for why we can't use that.
            if !flags.flags.intersects(ty::TypeFlags::KEEP_IN_LOCAL_TCX) {
                if let Some(interner) = global_interners {
                    let ty_struct: TyS<'gcx> = unsafe {
                        mem::transmute(ty_struct)
                    };
                    let ty: Ty<'gcx> = interner.arena.alloc(ty_struct);
                    global_interner.unwrap().insert(Interned(ty));
                    return ty;
                }
            } else {
                // Make sure we don't end up with inference
                // types/regions in the global tcx.
                if global_interners.is_none() {
                    drop(interner);
                    bug!("Attempted to intern `{:?}` which contains \
                          inference types/regions in the global type context",
                         &ty_struct);
                }
            }

            // Don't be &mut TyS.
            let ty: Ty<'tcx> = self.arena.alloc(ty_struct);
            interner.insert(Interned(ty));
            ty
        };

        debug!("Interned type: {:?} Pointer: {:?}",
            ty, ty as *const TyS);
        ty
    }

}

pub struct CommonTypes<'tcx> {
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub isize: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub i128: Ty<'tcx>,
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub u128: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub never: Ty<'tcx>,
    pub err: Ty<'tcx>,

    pub re_empty: Region<'tcx>,
    pub re_static: Region<'tcx>,
    pub re_erased: Region<'tcx>,
}

pub struct LocalTableInContext<'a, V: 'a> {
    local_id_root: Option<DefId>,
    data: &'a ItemLocalMap<V>
}

/// Validate that the given HirId (respectively its `local_id` part) can be
/// safely used as a key in the tables of a TypeckTable. For that to be
/// the case, the HirId must have the same `owner` as all the other IDs in
/// this table (signified by `local_id_root`). Otherwise the HirId
/// would be in a different frame of reference and using its `local_id`
/// would result in lookup errors, or worse, in silently wrong data being
/// stored/returned.
fn validate_hir_id_for_typeck_tables(local_id_root: Option<DefId>,
                                     hir_id: hir::HirId,
                                     mut_access: bool) {
    if cfg!(debug_assertions) {
        if let Some(local_id_root) = local_id_root {
            if hir_id.owner != local_id_root.index {
                ty::tls::with(|tcx| {
                    let node_id = tcx.hir
                                     .definitions()
                                     .find_node_for_hir_id(hir_id);

                    bug!("node {} with HirId::owner {:?} cannot be placed in \
                          TypeckTables with local_id_root {:?}",
                          tcx.hir.node_to_string(node_id),
                          DefId::local(hir_id.owner),
                          local_id_root)
                });
            }
        } else {
            // We use "Null Object" TypeckTables in some of the analysis passes.
            // These are just expected to be empty and their `local_id_root` is
            // `None`. Therefore we cannot verify whether a given `HirId` would
            // be a valid key for the given table. Instead we make sure that
            // nobody tries to write to such a Null Object table.
            if mut_access {
                bug!("access to invalid TypeckTables")
            }
        }
    }
}

impl<'a, V> LocalTableInContext<'a, V> {
    pub fn contains_key(&self, id: hir::HirId) -> bool {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.data.contains_key(&id.local_id)
    }

    pub fn get(&self, id: hir::HirId) -> Option<&V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.data.get(&id.local_id)
    }

    pub fn iter(&self) -> hash_map::Iter<hir::ItemLocalId, V> {
        self.data.iter()
    }
}

impl<'a, V> ::std::ops::Index<hir::HirId> for LocalTableInContext<'a, V> {
    type Output = V;

    fn index(&self, key: hir::HirId) -> &V {
        self.get(key).expect("LocalTableInContext: key not found")
    }
}

pub struct LocalTableInContextMut<'a, V: 'a> {
    local_id_root: Option<DefId>,
    data: &'a mut ItemLocalMap<V>
}

impl<'a, V> LocalTableInContextMut<'a, V> {
    pub fn get_mut(&mut self, id: hir::HirId) -> Option<&mut V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.get_mut(&id.local_id)
    }

    pub fn entry(&mut self, id: hir::HirId) -> Entry<hir::ItemLocalId, V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.entry(id.local_id)
    }

    pub fn insert(&mut self, id: hir::HirId, val: V) -> Option<V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.insert(id.local_id, val)
    }

    pub fn remove(&mut self, id: hir::HirId) -> Option<V> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, true);
        self.data.remove(&id.local_id)
    }
}

#[derive(RustcEncodable, RustcDecodable, Debug)]
pub struct TypeckTables<'tcx> {
    /// The HirId::owner all ItemLocalIds in this table are relative to.
    pub local_id_root: Option<DefId>,

    /// Resolved definitions for `<T>::X` associated paths and
    /// method calls, including those of overloaded operators.
    type_dependent_defs: ItemLocalMap<Def>,

    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    node_types: ItemLocalMap<Ty<'tcx>>,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    node_substs: ItemLocalMap<&'tcx Substs<'tcx>>,

    adjustments: ItemLocalMap<Vec<ty::adjustment::Adjustment<'tcx>>>,

    /// Stores the actual binding mode for all instances of hir::BindingAnnotation.
    pat_binding_modes: ItemLocalMap<BindingMode>,

    /// Stores the types which were implicitly dereferenced in pattern binding modes
    /// for later usage in HAIR lowering. For example,
    ///
    /// ```
    /// match &&Some(5i32) {
    ///     Some(n) => {},
    ///     _ => {},
    /// }
    /// ```
    /// leads to a `vec![&&Option<i32>, &Option<i32>]`. Empty vectors are not stored.
    ///
    /// See:
    /// https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#definitions
    pat_adjustments: ItemLocalMap<Vec<Ty<'tcx>>>,

    /// Borrows
    pub upvar_capture_map: ty::UpvarCaptureMap<'tcx>,

    /// Records the reasons that we picked the kind of each closure;
    /// not all closures are present in the map.
    closure_kind_origins: ItemLocalMap<(Span, ast::Name)>,

    /// For each fn, records the "liberated" types of its arguments
    /// and return type. Liberated means that all bound regions
    /// (including late-bound regions) are replaced with free
    /// equivalents. This table is not used in trans (since regions
    /// are erased there) and hence is not serialized to metadata.
    liberated_fn_sigs: ItemLocalMap<ty::FnSig<'tcx>>,

    /// For each FRU expression, record the normalized types of the fields
    /// of the struct - this is needed because it is non-trivial to
    /// normalize while preserving regions. This table is used only in
    /// MIR construction and hence is not serialized to metadata.
    fru_field_types: ItemLocalMap<Vec<Ty<'tcx>>>,

    /// Maps a cast expression to its kind. This is keyed on the
    /// *from* expression of the cast, not the cast itself.
    cast_kinds: ItemLocalMap<ty::cast::CastKind>,

    /// Set of trait imports actually used in the method resolution.
    /// This is used for warning unused imports. During type
    /// checking, this `Lrc` should not be cloned: it must have a ref-count
    /// of 1 so that we can insert things into the set mutably.
    pub used_trait_imports: Lrc<DefIdSet>,

    /// If any errors occurred while type-checking this body,
    /// this field will be set to `true`.
    pub tainted_by_errors: bool,

    /// Stores the free-region relationships that were deduced from
    /// its where clauses and parameter types. These are then
    /// read-again by borrowck.
    pub free_region_map: FreeRegionMap<'tcx>,
}

impl<'tcx> TypeckTables<'tcx> {
    pub fn empty(local_id_root: Option<DefId>) -> TypeckTables<'tcx> {
        TypeckTables {
            local_id_root,
            type_dependent_defs: ItemLocalMap(),
            node_types: ItemLocalMap(),
            node_substs: ItemLocalMap(),
            adjustments: ItemLocalMap(),
            pat_binding_modes: ItemLocalMap(),
            pat_adjustments: ItemLocalMap(),
            upvar_capture_map: FxHashMap(),
            closure_kind_origins: ItemLocalMap(),
            liberated_fn_sigs: ItemLocalMap(),
            fru_field_types: ItemLocalMap(),
            cast_kinds: ItemLocalMap(),
            used_trait_imports: Lrc::new(DefIdSet()),
            tainted_by_errors: false,
            free_region_map: FreeRegionMap::new(),
        }
    }

    /// Returns the final resolution of a `QPath` in an `Expr` or `Pat` node.
    pub fn qpath_def(&self, qpath: &hir::QPath, id: hir::HirId) -> Def {
        match *qpath {
            hir::QPath::Resolved(_, ref path) => path.def,
            hir::QPath::TypeRelative(..) => {
                validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
                self.type_dependent_defs.get(&id.local_id).cloned().unwrap_or(Def::Err)
            }
        }
    }

    pub fn type_dependent_defs(&self) -> LocalTableInContext<Def> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.type_dependent_defs
        }
    }

    pub fn type_dependent_defs_mut(&mut self) -> LocalTableInContextMut<Def> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.type_dependent_defs
        }
    }

    pub fn node_types(&self) -> LocalTableInContext<Ty<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.node_types
        }
    }

    pub fn node_types_mut(&mut self) -> LocalTableInContextMut<Ty<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.node_types
        }
    }

    pub fn node_id_to_type(&self, id: hir::HirId) -> Ty<'tcx> {
        match self.node_id_to_type_opt(id) {
            Some(ty) => ty,
            None => {
                bug!("node_id_to_type: no type for node `{}`",
                    tls::with(|tcx| {
                        let id = tcx.hir.definitions().find_node_for_hir_id(id);
                        tcx.hir.node_to_string(id)
                    }))
            }
        }
    }

    pub fn node_id_to_type_opt(&self, id: hir::HirId) -> Option<Ty<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_types.get(&id.local_id).cloned()
    }

    pub fn node_substs_mut(&mut self) -> LocalTableInContextMut<&'tcx Substs<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.node_substs
        }
    }

    pub fn node_substs(&self, id: hir::HirId) -> &'tcx Substs<'tcx> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned().unwrap_or(Substs::empty())
    }

    pub fn node_substs_opt(&self, id: hir::HirId) -> Option<&'tcx Substs<'tcx>> {
        validate_hir_id_for_typeck_tables(self.local_id_root, id, false);
        self.node_substs.get(&id.local_id).cloned()
    }

    // Returns the type of a pattern as a monotype. Like @expr_ty, this function
    // doesn't provide type parameter substitutions.
    pub fn pat_ty(&self, pat: &hir::Pat) -> Ty<'tcx> {
        self.node_id_to_type(pat.hir_id)
    }

    pub fn pat_ty_opt(&self, pat: &hir::Pat) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(pat.hir_id)
    }

    // Returns the type of an expression as a monotype.
    //
    // NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
    // some cases, we insert `Adjustment` annotations such as auto-deref or
    // auto-ref.  The type returned by this function does not consider such
    // adjustments.  See `expr_ty_adjusted()` instead.
    //
    // NB (2): This type doesn't provide type parameter substitutions; e.g. if you
    // ask for the type of "id" in "id(3)", it will return "fn(&isize) -> isize"
    // instead of "fn(ty) -> T with T = isize".
    pub fn expr_ty(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.node_id_to_type(expr.hir_id)
    }

    pub fn expr_ty_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.node_id_to_type_opt(expr.hir_id)
    }

    pub fn adjustments(&self) -> LocalTableInContext<Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.adjustments
        }
    }

    pub fn adjustments_mut(&mut self)
                           -> LocalTableInContextMut<Vec<ty::adjustment::Adjustment<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.adjustments
        }
    }

    pub fn expr_adjustments(&self, expr: &hir::Expr)
                            -> &[ty::adjustment::Adjustment<'tcx>] {
        validate_hir_id_for_typeck_tables(self.local_id_root, expr.hir_id, false);
        self.adjustments.get(&expr.hir_id.local_id).map_or(&[], |a| &a[..])
    }

    /// Returns the type of `expr`, considering any `Adjustment`
    /// entry recorded for that expression.
    pub fn expr_ty_adjusted(&self, expr: &hir::Expr) -> Ty<'tcx> {
        self.expr_adjustments(expr)
            .last()
            .map_or_else(|| self.expr_ty(expr), |adj| adj.target)
    }

    pub fn expr_ty_adjusted_opt(&self, expr: &hir::Expr) -> Option<Ty<'tcx>> {
        self.expr_adjustments(expr)
            .last()
            .map(|adj| adj.target)
            .or_else(|| self.expr_ty_opt(expr))
    }

    pub fn is_method_call(&self, expr: &hir::Expr) -> bool {
        // Only paths and method calls/overloaded operators have
        // entries in type_dependent_defs, ignore the former here.
        if let hir::ExprPath(_) = expr.node {
            return false;
        }

        match self.type_dependent_defs().get(expr.hir_id) {
            Some(&Def::Method(_)) => true,
            _ => false
        }
    }

    pub fn pat_binding_modes(&self) -> LocalTableInContext<BindingMode> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.pat_binding_modes
        }
    }

    pub fn pat_binding_modes_mut(&mut self)
                           -> LocalTableInContextMut<BindingMode> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.pat_binding_modes
        }
    }

    pub fn pat_adjustments(&self) -> LocalTableInContext<Vec<Ty<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.pat_adjustments,
        }
    }

    pub fn pat_adjustments_mut(&mut self)
                           -> LocalTableInContextMut<Vec<Ty<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.pat_adjustments,
        }
    }

    pub fn upvar_capture(&self, upvar_id: ty::UpvarId) -> ty::UpvarCapture<'tcx> {
        self.upvar_capture_map[&upvar_id]
    }

    pub fn closure_kind_origins(&self) -> LocalTableInContext<(Span, ast::Name)> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.closure_kind_origins
        }
    }

    pub fn closure_kind_origins_mut(&mut self) -> LocalTableInContextMut<(Span, ast::Name)> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.closure_kind_origins
        }
    }

    pub fn liberated_fn_sigs(&self) -> LocalTableInContext<ty::FnSig<'tcx>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.liberated_fn_sigs
        }
    }

    pub fn liberated_fn_sigs_mut(&mut self) -> LocalTableInContextMut<ty::FnSig<'tcx>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.liberated_fn_sigs
        }
    }

    pub fn fru_field_types(&self) -> LocalTableInContext<Vec<Ty<'tcx>>> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.fru_field_types
        }
    }

    pub fn fru_field_types_mut(&mut self) -> LocalTableInContextMut<Vec<Ty<'tcx>>> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.fru_field_types
        }
    }

    pub fn cast_kinds(&self) -> LocalTableInContext<ty::cast::CastKind> {
        LocalTableInContext {
            local_id_root: self.local_id_root,
            data: &self.cast_kinds
        }
    }

    pub fn cast_kinds_mut(&mut self) -> LocalTableInContextMut<ty::cast::CastKind> {
        LocalTableInContextMut {
            local_id_root: self.local_id_root,
            data: &mut self.cast_kinds
        }
    }
}

impl<'a, 'gcx> HashStable<StableHashingContext<'a>> for TypeckTables<'gcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ty::TypeckTables {
            local_id_root,
            ref type_dependent_defs,
            ref node_types,
            ref node_substs,
            ref adjustments,
            ref pat_binding_modes,
            ref pat_adjustments,
            ref upvar_capture_map,
            ref closure_kind_origins,
            ref liberated_fn_sigs,
            ref fru_field_types,

            ref cast_kinds,

            ref used_trait_imports,
            tainted_by_errors,
            ref free_region_map,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            type_dependent_defs.hash_stable(hcx, hasher);
            node_types.hash_stable(hcx, hasher);
            node_substs.hash_stable(hcx, hasher);
            adjustments.hash_stable(hcx, hasher);
            pat_binding_modes.hash_stable(hcx, hasher);
            pat_adjustments.hash_stable(hcx, hasher);
            hash_stable_hashmap(hcx, hasher, upvar_capture_map, |up_var_id, hcx| {
                let ty::UpvarId {
                    var_id,
                    closure_expr_id
                } = *up_var_id;

                let local_id_root =
                    local_id_root.expect("trying to hash invalid TypeckTables");

                let var_owner_def_id = DefId {
                    krate: local_id_root.krate,
                    index: var_id.owner,
                };
                let closure_def_id = DefId {
                    krate: local_id_root.krate,
                    index: closure_expr_id.to_def_id().index,
                };
                (hcx.def_path_hash(var_owner_def_id),
                 var_id.local_id,
                 hcx.def_path_hash(closure_def_id))
            });

            closure_kind_origins.hash_stable(hcx, hasher);
            liberated_fn_sigs.hash_stable(hcx, hasher);
            fru_field_types.hash_stable(hcx, hasher);
            cast_kinds.hash_stable(hcx, hasher);
            used_trait_imports.hash_stable(hcx, hasher);
            tainted_by_errors.hash_stable(hcx, hasher);
            free_region_map.hash_stable(hcx, hasher);
        })
    }
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonTypes<'tcx> {
        let mk = |sty| interners.intern_ty(sty, None);
        let mk_region = |r| {
            if let Some(r) = interners.region.borrow().get(&r) {
                return r.0;
            }
            let r = interners.arena.alloc(r);
            interners.region.borrow_mut().insert(Interned(r));
            &*r
        };
        CommonTypes {
            bool: mk(TyBool),
            char: mk(TyChar),
            never: mk(TyNever),
            err: mk(TyError),
            isize: mk(TyInt(ast::IntTy::Isize)),
            i8: mk(TyInt(ast::IntTy::I8)),
            i16: mk(TyInt(ast::IntTy::I16)),
            i32: mk(TyInt(ast::IntTy::I32)),
            i64: mk(TyInt(ast::IntTy::I64)),
            i128: mk(TyInt(ast::IntTy::I128)),
            usize: mk(TyUint(ast::UintTy::Usize)),
            u8: mk(TyUint(ast::UintTy::U8)),
            u16: mk(TyUint(ast::UintTy::U16)),
            u32: mk(TyUint(ast::UintTy::U32)),
            u64: mk(TyUint(ast::UintTy::U64)),
            u128: mk(TyUint(ast::UintTy::U128)),
            f32: mk(TyFloat(ast::FloatTy::F32)),
            f64: mk(TyFloat(ast::FloatTy::F64)),

            re_empty: mk_region(RegionKind::ReEmpty),
            re_static: mk_region(RegionKind::ReStatic),
            re_erased: mk_region(RegionKind::ReErased),
        }
    }
}

/// The central data structure of the compiler. It stores references
/// to the various **arenas** and also houses the results of the
/// various **compiler queries** that have been performed. See the
/// [rustc guide] for more details.
///
/// [rustc guide]: https://rust-lang-nursery.github.io/rustc-guide/ty.html
#[derive(Copy, Clone)]
pub struct TyCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    gcx: &'a GlobalCtxt<'gcx>,
    interners: &'a CtxtInterners<'tcx>
}

impl<'a, 'gcx, 'tcx> Deref for TyCtxt<'a, 'gcx, 'tcx> {
    type Target = &'a GlobalCtxt<'gcx>;
    fn deref(&self) -> &Self::Target {
        &self.gcx
    }
}

pub struct GlobalCtxt<'tcx> {
    global_arenas: &'tcx GlobalArenas<'tcx>,
    global_interners: CtxtInterners<'tcx>,

    cstore: &'tcx dyn CrateStore,

    pub sess: &'tcx Session,

    pub dep_graph: DepGraph,

    /// This provides access to the incr. comp. on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure in `ty::maps`.
    pub(crate) on_disk_query_result_cache: maps::OnDiskCache<'tcx>,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    trait_map: FxHashMap<DefIndex,
                         Lrc<FxHashMap<ItemLocalId,
                                       Lrc<StableVec<TraitCandidate>>>>>,

    /// Export map produced by name resolution.
    export_map: FxHashMap<DefId, Lrc<Vec<Export>>>,

    pub hir: hir_map::Map<'tcx>,

    /// A map from DefPathHash -> DefId. Includes DefIds from the local crate
    /// as well as all upstream crates. Only populated in incremental mode.
    pub def_path_hash_to_def_id: Option<FxHashMap<DefPathHash, DefId>>,

    pub maps: maps::Maps<'tcx>,

    // Records the free variables refrenced by every closure
    // expression. Do not track deps for this, just recompute it from
    // scratch every time.
    freevars: FxHashMap<DefId, Lrc<Vec<hir::Freevar>>>,

    maybe_unused_trait_imports: FxHashSet<DefId>,

    maybe_unused_extern_crates: Vec<(DefId, Span)>,

    // Internal cache for metadata decoding. No need to track deps on this.
    pub rcache: RefCell<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    /// The definite name of the current crate after taking into account
    /// attributes, commandline parameters, etc.
    pub crate_name: Symbol,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    /// Used to prevent layout from recursing too deeply.
    pub layout_depth: Cell<usize>,

    /// Map from function to the `#[derive]` mode that it's defining. Only used
    /// by `proc-macro` crates.
    pub derive_macros: RefCell<NodeMap<Symbol>>,

    stability_interner: RefCell<FxHashSet<&'tcx attr::Stability>>,

    pub interpret_interner: InterpretInterner<'tcx>,

    layout_interner: RefCell<FxHashSet<&'tcx LayoutDetails>>,

    /// A vector of every trait accessible in the whole crate
    /// (i.e. including those from subcrates). This is used only for
    /// error reporting, and so is lazily initialized and generally
    /// shouldn't taint the common path (hence the RefCell).
    pub all_traits: RefCell<Option<Vec<DefId>>>,

    /// A general purpose channel to throw data out the back towards LLVM worker
    /// threads.
    ///
    /// This is intended to only get used during the trans phase of the compiler
    /// when satisfying the query for a particular codegen unit. Internally in
    /// the query it'll send data along this channel to get processed later.
    pub tx_to_llvm_workers: mpsc::Sender<Box<dyn Any + Send>>,

    output_filenames: Arc<OutputFilenames>,
}

/// Everything needed to efficiently work with interned allocations
#[derive(Debug, Default)]
pub struct InterpretInterner<'tcx> {
    inner: RefCell<InterpretInternerInner<'tcx>>,
}

#[derive(Debug, Default)]
struct InterpretInternerInner<'tcx> {
    /// Stores the value of constants (and deduplicates the actual memory)
    allocs: FxHashSet<&'tcx interpret::Allocation>,

    /// Allows obtaining function instance handles via a unique identifier
    functions: FxHashMap<interpret::AllocId, Instance<'tcx>>,

    /// Inverse map of `interpret_functions`.
    /// Used so we don't allocate a new pointer every time we need one
    function_cache: FxHashMap<Instance<'tcx>, interpret::AllocId>,

    /// Allows obtaining const allocs via a unique identifier
    alloc_by_id: FxHashMap<interpret::AllocId, &'tcx interpret::Allocation>,

    /// Reverse map of `alloc_cache`
    global_cache: FxHashMap<interpret::AllocId, DefId>,

    /// The AllocId to assign to the next new regular allocation.
    /// Always incremented, never gets smaller.
    next_id: interpret::AllocId,

    /// Allows checking whether a static already has an allocation
    ///
    /// This is only important for detecting statics referring to themselves
    // FIXME(oli-obk) move it to the EvalContext?
    alloc_cache: FxHashMap<DefId, interpret::AllocId>,

    /// A cache for basic byte allocations keyed by their contents. This is used to deduplicate
    /// allocations for string and bytestring literals.
    literal_alloc_cache: FxHashMap<Vec<u8>, interpret::AllocId>,
}

impl<'tcx> InterpretInterner<'tcx> {
    pub fn create_fn_alloc(&self, instance: Instance<'tcx>) -> interpret::AllocId {
        if let Some(&alloc_id) = self.inner.borrow().function_cache.get(&instance) {
            return alloc_id;
        }
        let id = self.reserve();
        debug!("creating fn ptr: {}", id);
        let mut inner = self.inner.borrow_mut();
        inner.functions.insert(id, instance);
        inner.function_cache.insert(instance, id);
        id
    }

    pub fn get_fn(
        &self,
        id: interpret::AllocId,
    ) -> Option<Instance<'tcx>> {
        self.inner.borrow().functions.get(&id).cloned()
    }

    pub fn get_alloc(
        &self,
        id: interpret::AllocId,
    ) -> Option<&'tcx interpret::Allocation> {
        self.inner.borrow().alloc_by_id.get(&id).cloned()
    }

    pub fn get_cached(
        &self,
        static_id: DefId,
    ) -> Option<interpret::AllocId> {
        self.inner.borrow().alloc_cache.get(&static_id).cloned()
    }

    pub fn cache(
        &self,
        static_id: DefId,
        alloc_id: interpret::AllocId,
    ) {
        let mut inner = self.inner.borrow_mut();
        inner.global_cache.insert(alloc_id, static_id);
        if let Some(old) = inner.alloc_cache.insert(static_id, alloc_id) {
            bug!("tried to cache {:?}, but was already existing as {:#?}", static_id, old);
        }
    }

    pub fn get_corresponding_static_def_id(
        &self,
        ptr: interpret::AllocId,
    ) -> Option<DefId> {
        self.inner.borrow().global_cache.get(&ptr).cloned()
    }

    pub fn intern_at_reserved(
        &self,
        id: interpret::AllocId,
        alloc: &'tcx interpret::Allocation,
    ) {
        if let Some(old) = self.inner.borrow_mut().alloc_by_id.insert(id, alloc) {
            bug!("tried to intern allocation at {}, but was already existing as {:#?}", id, old);
        }
    }

    /// obtains a new allocation ID that can be referenced but does not
    /// yet have an allocation backing it.
    pub fn reserve(
        &self,
    ) -> interpret::AllocId {
        let mut inner = self.inner.borrow_mut();
        let next = inner.next_id;
        inner.next_id.0 = inner.next_id.0
            .checked_add(1)
            .expect("You overflowed a u64 by incrementing by 1... \
                     You've just earned yourself a free drink if we ever meet. \
                     Seriously, how did you do that?!");
        next
    }
}

impl<'tcx> GlobalCtxt<'tcx> {
    /// Get the global TyCtxt.
    pub fn global_tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        TyCtxt {
            gcx: self,
            interners: &self.global_interners
        }
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn alloc_generics(self, generics: ty::Generics) -> &'gcx ty::Generics {
        self.global_arenas.generics.alloc(generics)
    }

    pub fn alloc_steal_mir(self, mir: Mir<'gcx>) -> &'gcx Steal<Mir<'gcx>> {
        self.global_arenas.steal_mir.alloc(Steal::new(mir))
    }

    pub fn alloc_mir(self, mir: Mir<'gcx>) -> &'gcx Mir<'gcx> {
        self.global_arenas.mir.alloc(mir)
    }

    pub fn alloc_tables(self, tables: ty::TypeckTables<'gcx>) -> &'gcx ty::TypeckTables<'gcx> {
        self.global_arenas.tables.alloc(tables)
    }

    pub fn alloc_trait_def(self, def: ty::TraitDef) -> &'gcx ty::TraitDef {
        self.global_arenas.trait_def.alloc(def)
    }

    pub fn alloc_adt_def(self,
                         did: DefId,
                         kind: AdtKind,
                         variants: Vec<ty::VariantDef>,
                         repr: ReprOptions)
                         -> &'gcx ty::AdtDef {
        let def = ty::AdtDef::new(self, did, kind, variants, repr);
        self.global_arenas.adt_def.alloc(def)
    }

    pub fn alloc_byte_array(self, bytes: &[u8]) -> &'gcx [u8] {
        if bytes.is_empty() {
            &[]
        } else {
            self.global_interners.arena.alloc_slice(bytes)
        }
    }

    pub fn alloc_const_slice(self, values: &[&'tcx ty::Const<'tcx>])
                             -> &'tcx [&'tcx ty::Const<'tcx>] {
        if values.is_empty() {
            &[]
        } else {
            self.interners.arena.alloc_slice(values)
        }
    }

    pub fn alloc_name_const_slice(self, values: &[(ast::Name, &'tcx ty::Const<'tcx>)])
                                  -> &'tcx [(ast::Name, &'tcx ty::Const<'tcx>)] {
        if values.is_empty() {
            &[]
        } else {
            self.interners.arena.alloc_slice(values)
        }
    }

    pub fn intern_const_alloc(
        self,
        alloc: interpret::Allocation,
    ) -> &'gcx interpret::Allocation {
        if let Some(alloc) = self.interpret_interner.inner.borrow().allocs.get(&alloc) {
            return alloc;
        }

        let interned = self.global_arenas.const_allocs.alloc(alloc);
        if let Some(prev) = self.interpret_interner.inner.borrow_mut().allocs.replace(interned) {
            bug!("Tried to overwrite interned Allocation: {:#?}", prev)
        }
        interned
    }

    /// Allocates a byte or string literal for `mir::interpret`
    pub fn allocate_cached(self, bytes: &[u8]) -> interpret::AllocId {
        // check whether we already allocated this literal or a constant with the same memory
        if let Some(&alloc_id) = self.interpret_interner.inner.borrow()
                                     .literal_alloc_cache.get(bytes) {
            return alloc_id;
        }
        // create an allocation that just contains these bytes
        let alloc = interpret::Allocation::from_bytes(bytes);
        let alloc = self.intern_const_alloc(alloc);

        // the next unique id
        let id = self.interpret_interner.reserve();
        // make the allocation identifiable
        self.interpret_interner.inner.borrow_mut().alloc_by_id.insert(id, alloc);
        // cache it for the future
        self.interpret_interner.inner.borrow_mut().literal_alloc_cache.insert(bytes.to_owned(), id);
        id
    }

    pub fn intern_stability(self, stab: attr::Stability) -> &'gcx attr::Stability {
        if let Some(st) = self.stability_interner.borrow().get(&stab) {
            return st;
        }

        let interned = self.global_interners.arena.alloc(stab);
        if let Some(prev) = self.stability_interner.borrow_mut().replace(interned) {
            bug!("Tried to overwrite interned Stability: {:?}", prev)
        }
        interned
    }

    pub fn intern_layout(self, layout: LayoutDetails) -> &'gcx LayoutDetails {
        if let Some(layout) = self.layout_interner.borrow().get(&layout) {
            return layout;
        }

        let interned = self.global_arenas.layout.alloc(layout);
        if let Some(prev) = self.layout_interner.borrow_mut().replace(interned) {
            bug!("Tried to overwrite interned Layout: {:?}", prev)
        }
        interned
    }

    pub fn lift<T: ?Sized + Lift<'tcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self)
    }

    /// Like lift, but only tries in the global tcx.
    pub fn lift_to_global<T: ?Sized + Lift<'gcx>>(self, value: &T) -> Option<T::Lifted> {
        value.lift_to_tcx(self.global_tcx())
    }

    /// Returns true if self is the same as self.global_tcx().
    fn is_global(self) -> bool {
        let local = self.interners as *const _;
        let global = &self.global_interners as *const _;
        local as usize == global as usize
    }

    /// Create a type context and call the closure with a `TyCtxt` reference
    /// to the context. The closure enforces that the type context and any interned
    /// value (types, substs, etc.) can only be used while `ty::tls` has a valid
    /// reference to the context, to allow formatting values that need it.
    pub fn create_and_enter<F, R>(s: &'tcx Session,
                                  cstore: &'tcx dyn CrateStore,
                                  local_providers: ty::maps::Providers<'tcx>,
                                  extern_providers: ty::maps::Providers<'tcx>,
                                  arenas: &'tcx AllArenas<'tcx>,
                                  resolutions: ty::Resolutions,
                                  hir: hir_map::Map<'tcx>,
                                  on_disk_query_result_cache: maps::OnDiskCache<'tcx>,
                                  crate_name: &str,
                                  tx: mpsc::Sender<Box<dyn Any + Send>>,
                                  output_filenames: &OutputFilenames,
                                  f: F) -> R
                                  where F: for<'b> FnOnce(TyCtxt<'b, 'tcx, 'tcx>) -> R
    {
        let data_layout = TargetDataLayout::parse(s);
        let interners = CtxtInterners::new(&arenas.interner);
        let common_types = CommonTypes::new(&interners);
        let dep_graph = hir.dep_graph.clone();
        let max_cnum = cstore.crates_untracked().iter().map(|c| c.as_usize()).max().unwrap_or(0);
        let mut providers = IndexVec::from_elem_n(extern_providers, max_cnum + 1);
        providers[LOCAL_CRATE] = local_providers;

        let def_path_hash_to_def_id = if s.opts.build_dep_graph() {
            let upstream_def_path_tables: Vec<(CrateNum, Lrc<_>)> = cstore
                .crates_untracked()
                .iter()
                .map(|&cnum| (cnum, cstore.def_path_table(cnum)))
                .collect();

            let def_path_tables = || {
                upstream_def_path_tables
                    .iter()
                    .map(|&(cnum, ref rc)| (cnum, &**rc))
                    .chain(iter::once((LOCAL_CRATE, hir.definitions().def_path_table())))
            };

            // Precompute the capacity of the hashmap so we don't have to
            // re-allocate when populating it.
            let capacity = def_path_tables().map(|(_, t)| t.size()).sum::<usize>();

            let mut map: FxHashMap<_, _> = FxHashMap::with_capacity_and_hasher(
                capacity,
                ::std::default::Default::default()
            );

            for (cnum, def_path_table) in def_path_tables() {
                def_path_table.add_def_path_hashes_to(cnum, &mut map);
            }

            Some(map)
        } else {
            None
        };

        let mut trait_map = FxHashMap();
        for (k, v) in resolutions.trait_map {
            let hir_id = hir.node_to_hir_id(k);
            let map = trait_map.entry(hir_id.owner)
                .or_insert_with(|| Lrc::new(FxHashMap()));
            Lrc::get_mut(map).unwrap()
                            .insert(hir_id.local_id,
                                    Lrc::new(StableVec::new(v)));
        }

        tls::enter_global(GlobalCtxt {
            sess: s,
            cstore,
            global_arenas: &arenas.global,
            global_interners: interners,
            dep_graph: dep_graph.clone(),
            on_disk_query_result_cache,
            types: common_types,
            trait_map,
            export_map: resolutions.export_map.into_iter().map(|(k, v)| {
                (k, Lrc::new(v))
            }).collect(),
            freevars: resolutions.freevars.into_iter().map(|(k, v)| {
                (hir.local_def_id(k), Lrc::new(v))
            }).collect(),
            maybe_unused_trait_imports:
                resolutions.maybe_unused_trait_imports
                    .into_iter()
                    .map(|id| hir.local_def_id(id))
                    .collect(),
            maybe_unused_extern_crates:
                resolutions.maybe_unused_extern_crates
                    .into_iter()
                    .map(|(id, sp)| (hir.local_def_id(id), sp))
                    .collect(),
            hir,
            def_path_hash_to_def_id,
            maps: maps::Maps::new(providers),
            rcache: RefCell::new(FxHashMap()),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            crate_name: Symbol::intern(crate_name),
            data_layout,
            layout_interner: RefCell::new(FxHashSet()),
            layout_depth: Cell::new(0),
            derive_macros: RefCell::new(NodeMap()),
            stability_interner: RefCell::new(FxHashSet()),
            interpret_interner: Default::default(),
            all_traits: RefCell::new(None),
            tx_to_llvm_workers: tx,
            output_filenames: Arc::new(output_filenames.clone()),
       }, f)
    }

    pub fn consider_optimizing<T: Fn() -> String>(&self, msg: T) -> bool {
        let cname = self.crate_name(LOCAL_CRATE).as_str();
        self.sess.consider_optimizing(&cname, msg)
    }

    pub fn lang_items(self) -> Lrc<middle::lang_items::LanguageItems> {
        self.get_lang_items(LOCAL_CRATE)
    }

    /// Due to missing llvm support for lowering 128 bit math to software emulation
    /// (on some targets), the lowering can be done in MIR.
    ///
    /// This function only exists until said support is implemented.
    pub fn is_binop_lang_item(&self, def_id: DefId) -> Option<(mir::BinOp, bool)> {
        let items = self.lang_items();
        let def_id = Some(def_id);
        if items.i128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.u128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.i128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.u128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.i128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.u128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.i128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.u128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.i128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.u128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.i128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.u128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.i128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.u128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.i128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.u128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.i128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.u128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.i128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.u128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.i128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.u128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.i128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else if items.u128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else { None }
    }

    pub fn stability(self) -> Lrc<stability::Index<'tcx>> {
        self.stability_index(LOCAL_CRATE)
    }

    pub fn crates(self) -> Lrc<Vec<CrateNum>> {
        self.all_crate_nums(LOCAL_CRATE)
    }

    pub fn features(self) -> Lrc<feature_gate::Features> {
        self.features_query(LOCAL_CRATE)
    }

    pub fn def_key(self, id: DefId) -> hir_map::DefKey {
        if id.is_local() {
            self.hir.def_key(id)
        } else {
            self.cstore.def_key(id)
        }
    }

    /// Convert a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned def-path).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> hir_map::DefPath {
        if id.is_local() {
            self.hir.def_path(id)
        } else {
            self.cstore.def_path(id)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> hir_map::DefPathHash {
        if def_id.is_local() {
            self.hir.definitions().def_path_hash(def_id.index)
        } else {
            self.cstore.def_path_hash(def_id)
        }
    }

    pub fn def_path_debug_str(self, def_id: DefId) -> String {
        // We are explicitly not going through queries here in order to get
        // crate name and disambiguator since this code is called from debug!()
        // statements within the query system and we'd run into endless
        // recursion otherwise.
        let (crate_name, crate_disambiguator) = if def_id.is_local() {
            (self.crate_name.clone(),
             self.sess.local_crate_disambiguator())
        } else {
            (self.cstore.crate_name_untracked(def_id.krate),
             self.cstore.crate_disambiguator_untracked(def_id.krate))
        };

        format!("{}[{}]{}",
                crate_name,
                // Don't print the whole crate disambiguator. That's just
                // annoying in debug output.
                &(crate_disambiguator.to_fingerprint().to_hex())[..4],
                self.def_path(def_id).to_string_no_crate())
    }

    pub fn metadata_encoding_version(self) -> Vec<u8> {
        self.cstore.metadata_encoding_version().to_vec()
    }

    // Note that this is *untracked* and should only be used within the query
    // system if the result is otherwise tracked through queries
    pub fn crate_data_as_rc_any(self, cnum: CrateNum) -> Lrc<dyn Any> {
        self.cstore.crate_data_as_rc_any(cnum)
    }

    pub fn create_stable_hashing_context(self) -> StableHashingContext<'a> {
        let krate = self.dep_graph.with_ignore(|| self.gcx.hir.krate());

        StableHashingContext::new(self.sess,
                                  krate,
                                  self.hir.definitions(),
                                  self.cstore)
    }

    // This method makes sure that we have a DepNode and a Fingerprint for
    // every upstream crate. It needs to be called once right after the tcx is
    // created.
    // With full-fledged red/green, the method will probably become unnecessary
    // as this will be done on-demand.
    pub fn allocate_metadata_dep_nodes(self) {
        // We cannot use the query versions of crates() and crate_hash(), since
        // those would need the DepNodes that we are allocating here.
        for cnum in self.cstore.crates_untracked() {
            let dep_node = DepNode::new(self, DepConstructor::CrateMetadata(cnum));
            let crate_hash = self.cstore.crate_hash_untracked(cnum);
            self.dep_graph.with_task(dep_node,
                                     self,
                                     crate_hash,
                                     |_, x| x // No transformation needed
            );
        }
    }

    // This method exercises the `in_scope_traits_map` query for all possible
    // values so that we have their fingerprints available in the DepGraph.
    // This is only required as long as we still use the old dependency tracking
    // which needs to have the fingerprints of all input nodes beforehand.
    pub fn precompute_in_scope_traits_hashes(self) {
        for &def_index in self.trait_map.keys() {
            self.in_scope_traits_map(def_index);
        }
    }

    pub fn serialize_query_result_cache<E>(self,
                                           encoder: &mut E)
                                           -> Result<(), E::Error>
        where E: ty::codec::TyEncoder
    {
        self.on_disk_query_result_cache.serialize(self.global_tcx(), encoder)
    }

    /// If true, we should use NLL-style region checking instead of
    /// lexical style.
    pub fn nll(self) -> bool {
        self.features().nll || self.sess.opts.debugging_opts.nll
    }

    /// If true, we should use the MIR-based borrowck (we may *also* use
    /// the AST-based borrowck).
    pub fn use_mir(self) -> bool {
        self.borrowck_mode().use_mir()
    }

    /// If true, we should enable two-phase borrows checks. This is
    /// done with either `-Ztwo-phase-borrows` or with
    /// `#![feature(nll)]`.
    pub fn two_phase_borrows(self) -> bool {
        self.features().nll || self.sess.opts.debugging_opts.two_phase_borrows
    }

    /// What mode(s) of borrowck should we run? AST? MIR? both?
    /// (Also considers the `#![feature(nll)]` setting.)
    pub fn borrowck_mode(&self) -> BorrowckMode {
        match self.sess.opts.borrowck_mode {
            mode @ BorrowckMode::Mir |
            mode @ BorrowckMode::Compare => mode,

            mode @ BorrowckMode::Ast => {
                if self.nll() {
                    BorrowckMode::Mir
                } else {
                    mode
                }
            }

        }
    }

    /// Should we emit EndRegion MIR statements? These are consumed by
    /// MIR borrowck, but not when NLL is used. They are also consumed
    /// by the validation stuff.
    pub fn emit_end_regions(self) -> bool {
        // FIXME(#46875) -- we should not emit end regions when NLL is enabled,
        // but for now we can't stop doing so because it causes false positives
        self.sess.opts.debugging_opts.emit_end_regions ||
            self.sess.opts.debugging_opts.mir_emit_validate > 0 ||
            self.use_mir()
    }
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    pub fn encode_metadata(self, link_meta: &LinkMeta)
        -> EncodedMetadata
    {
        self.cstore.encode_metadata(self, link_meta)
    }
}

impl<'gcx: 'tcx, 'tcx> GlobalCtxt<'gcx> {
    /// Call the closure with a local `TyCtxt` using the given arena.
    pub fn enter_local<F, R>(&self, arena: &'tcx DroplessArena, f: F) -> R
        where F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let interners = CtxtInterners::new(arena);
        tls::enter(self, &interners, f)
    }
}

/// A trait implemented for all X<'a> types which can be safely and
/// efficiently converted to X<'tcx> as long as they are part of the
/// provided TyCtxt<'tcx>.
/// This can be done, for example, for Ty<'tcx> or &'tcx Substs<'tcx>
/// by looking them up in their respective interners.
///
/// However, this is still not the best implementation as it does
/// need to compare the components, even for interned values.
/// It would be more efficient if TypedArena provided a way to
/// determine whether the address is in the allocated range.
///
/// None is returned if the value or one of the components is not part
/// of the provided context.
/// For Ty, None can be returned if either the type interner doesn't
/// contain the TypeVariants key or if the address of the interned
/// pointer differs. The latter case is possible if a primitive type,
/// e.g. `()` or `u8`, was interned in a different context.
pub trait Lift<'tcx> {
    type Lifted: 'tcx;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted>;
}

impl<'a, 'tcx> Lift<'tcx> for Ty<'a> {
    type Lifted = Ty<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Ty<'tcx>> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for Region<'a> {
    type Lifted = Region<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Region<'tcx>> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Const<'a> {
    type Lifted = &'tcx Const<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<&'tcx Const<'tcx>> {
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Substs<'a> {
    type Lifted = &'tcx Substs<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<&'tcx Substs<'tcx>> {
        if self.len() == 0 {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(&self[..] as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Slice<Ty<'a>> {
    type Lifted = &'tcx Slice<Ty<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<&'tcx Slice<Ty<'tcx>>> {
        if self.len() == 0 {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Slice<ExistentialPredicate<'a>> {
    type Lifted = &'tcx Slice<ExistentialPredicate<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
        -> Option<&'tcx Slice<ExistentialPredicate<'tcx>>> {
        if self.is_empty() {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for &'a Slice<Predicate<'a>> {
    type Lifted = &'tcx Slice<Predicate<'tcx>>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
        -> Option<&'tcx Slice<Predicate<'tcx>>> {
        if self.is_empty() {
            return Some(Slice::empty());
        }
        if tcx.interners.arena.in_arena(*self as *const _) {
            return Some(unsafe { mem::transmute(*self) });
        }
        // Also try in the global tcx if we're not that.
        if !tcx.is_global() {
            self.lift_to_tcx(tcx.global_tcx())
        } else {
            None
        }
    }
}

pub mod tls {
    use super::{CtxtInterners, GlobalCtxt, TyCtxt};

    use std::cell::Cell;
    use std::fmt;
    use syntax_pos;

    /// Marker types used for the scoped TLS slot.
    /// The type context cannot be used directly because the scoped TLS
    /// in libstd doesn't allow types generic over lifetimes.
    enum ThreadLocalGlobalCtxt {}
    enum ThreadLocalInterners {}

    thread_local! {
        static TLS_TCX: Cell<Option<(*const ThreadLocalGlobalCtxt,
                                     *const ThreadLocalInterners)>> = Cell::new(None)
    }

    fn span_debug(span: syntax_pos::Span, f: &mut fmt::Formatter) -> fmt::Result {
        with(|tcx| {
            write!(f, "{}", tcx.sess.codemap().span_to_string(span))
        })
    }

    pub fn enter_global<'gcx, F, R>(gcx: GlobalCtxt<'gcx>, f: F) -> R
        where F: for<'a> FnOnce(TyCtxt<'a, 'gcx, 'gcx>) -> R
    {
        syntax_pos::SPAN_DEBUG.with(|span_dbg| {
            let original_span_debug = span_dbg.get();
            span_dbg.set(span_debug);
            let result = enter(&gcx, &gcx.global_interners, f);
            span_dbg.set(original_span_debug);
            result
        })
    }

    pub fn enter<'a, 'gcx: 'tcx, 'tcx, F, R>(gcx: &'a GlobalCtxt<'gcx>,
                                             interners: &'a CtxtInterners<'tcx>,
                                             f: F) -> R
        where F: FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        let gcx_ptr = gcx as *const _ as *const ThreadLocalGlobalCtxt;
        let interners_ptr = interners as *const _ as *const ThreadLocalInterners;
        TLS_TCX.with(|tls| {
            let prev = tls.get();
            tls.set(Some((gcx_ptr, interners_ptr)));
            let ret = f(TyCtxt {
                gcx,
                interners,
            });
            tls.set(prev);
            ret
        })
    }

    pub fn with<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(TyCtxt<'a, 'gcx, 'tcx>) -> R
    {
        TLS_TCX.with(|tcx| {
            let (gcx, interners) = tcx.get().unwrap();
            let gcx = unsafe { &*(gcx as *const GlobalCtxt) };
            let interners = unsafe { &*(interners as *const CtxtInterners) };
            f(TyCtxt {
                gcx,
                interners,
            })
        })
    }

    pub fn with_opt<F, R>(f: F) -> R
        where F: for<'a, 'gcx, 'tcx> FnOnce(Option<TyCtxt<'a, 'gcx, 'tcx>>) -> R
    {
        if TLS_TCX.with(|tcx| tcx.get().is_some()) {
            with(|v| f(Some(v)))
        } else {
            f(None)
        }
    }
}

macro_rules! sty_debug_print {
    ($ctxt: expr, $($variant: ident),*) => {{
        // curious inner module to allow variant names to be used as
        // variable names.
        #[allow(non_snake_case)]
        mod inner {
            use ty::{self, TyCtxt};
            use ty::context::Interned;

            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                region_infer: usize,
                ty_infer: usize,
                both_infer: usize,
            }

            pub fn go(tcx: TyCtxt) {
                let mut total = DebugStat {
                    total: 0,
                    region_infer: 0, ty_infer: 0, both_infer: 0,
                };
                $(let mut $variant = total;)*


                for &Interned(t) in tcx.interners.type_.borrow().iter() {
                    let variant = match t.sty {
                        ty::TyBool | ty::TyChar | ty::TyInt(..) | ty::TyUint(..) |
                            ty::TyFloat(..) | ty::TyStr | ty::TyNever => continue,
                        ty::TyError => /* unimportant */ continue,
                        $(ty::$variant(..) => &mut $variant,)*
                    };
                    let region = t.flags.intersects(ty::TypeFlags::HAS_RE_INFER);
                    let ty = t.flags.intersects(ty::TypeFlags::HAS_TY_INFER);

                    variant.total += 1;
                    total.total += 1;
                    if region { total.region_infer += 1; variant.region_infer += 1 }
                    if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                    if region && ty { total.both_infer += 1; variant.both_infer += 1 }
                }
                println!("Ty interner             total           ty region  both");
                $(println!("    {:18}: {uses:6} {usespc:4.1}%, \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                           stringify!($variant),
                           uses = $variant.total,
                           usespc = $variant.total as f64 * 100.0 / total.total as f64,
                           ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                           region = $variant.region_infer as f64 * 100.0  / total.total as f64,
                           both = $variant.both_infer as f64 * 100.0  / total.total as f64);
                  )*
                println!("                  total {uses:6}        \
{ty:4.1}% {region:5.1}% {both:4.1}%",
                         uses = total.total,
                         ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                         region = total.region_infer as f64 * 100.0  / total.total as f64,
                         both = total.both_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($ctxt)
    }}
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    pub fn print_debug_stats(self) {
        sty_debug_print!(
            self,
            TyAdt, TyArray, TySlice, TyRawPtr, TyRef, TyFnDef, TyFnPtr,
            TyGenerator, TyGeneratorWitness, TyDynamic, TyClosure, TyTuple,
            TyParam, TyInfer, TyProjection, TyAnon, TyForeign);

        println!("Substs interner: #{}", self.interners.substs.borrow().len());
        println!("Region interner: #{}", self.interners.region.borrow().len());
        println!("Stability interner: #{}", self.stability_interner.borrow().len());
        println!("Interpret interner: #{}", self.interpret_interner.inner.borrow().allocs.len());
        println!("Layout interner: #{}", self.layout_interner.borrow().len());
    }
}


/// An entry in an interner.
struct Interned<'tcx, T: 'tcx+?Sized>(&'tcx T);

// NB: An Interned<Ty> compares and hashes as a sty.
impl<'tcx> PartialEq for Interned<'tcx, TyS<'tcx>> {
    fn eq(&self, other: &Interned<'tcx, TyS<'tcx>>) -> bool {
        self.0.sty == other.0.sty
    }
}

impl<'tcx> Eq for Interned<'tcx, TyS<'tcx>> {}

impl<'tcx> Hash for Interned<'tcx, TyS<'tcx>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0.sty.hash(s)
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<TypeVariants<'lcx>> for Interned<'tcx, TyS<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a TypeVariants<'lcx> {
        &self.0.sty
    }
}

// NB: An Interned<Slice<T>> compares and hashes as its elements.
impl<'tcx, T: PartialEq> PartialEq for Interned<'tcx, Slice<T>> {
    fn eq(&self, other: &Interned<'tcx, Slice<T>>) -> bool {
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for Interned<'tcx, Slice<T>> {}

impl<'tcx, T: Hash> Hash for Interned<'tcx, Slice<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.0[..].hash(s)
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Ty<'lcx>]> for Interned<'tcx, Slice<Ty<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Ty<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Kind<'lcx>]> for Interned<'tcx, Substs<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a [Kind<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx> Borrow<RegionKind> for Interned<'tcx, RegionKind> {
    fn borrow<'a>(&'a self) -> &'a RegionKind {
        &self.0
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[ExistentialPredicate<'lcx>]>
    for Interned<'tcx, Slice<ExistentialPredicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [ExistentialPredicate<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<[Predicate<'lcx>]>
    for Interned<'tcx, Slice<Predicate<'tcx>>> {
    fn borrow<'a>(&'a self) -> &'a [Predicate<'lcx>] {
        &self.0[..]
    }
}

impl<'tcx: 'lcx, 'lcx> Borrow<Const<'lcx>> for Interned<'tcx, Const<'tcx>> {
    fn borrow<'a>(&'a self) -> &'a Const<'lcx> {
        &self.0
    }
}

macro_rules! intern_method {
    ($lt_tcx:tt, $name:ident: $method:ident($alloc:ty,
                                            $alloc_method:ident,
                                            $alloc_to_key:expr,
                                            $alloc_to_ret:expr,
                                            $needs_infer:expr) -> $ty:ty) => {
        impl<'a, 'gcx, $lt_tcx> TyCtxt<'a, 'gcx, $lt_tcx> {
            pub fn $method(self, v: $alloc) -> &$lt_tcx $ty {
                {
                    let key = ($alloc_to_key)(&v);
                    if let Some(i) = self.interners.$name.borrow().get(key) {
                        return i.0;
                    }
                    if !self.is_global() {
                        if let Some(i) = self.global_interners.$name.borrow().get(key) {
                            return i.0;
                        }
                    }
                }

                // HACK(eddyb) Depend on flags being accurate to
                // determine that all contents are in the global tcx.
                // See comments on Lift for why we can't use that.
                if !($needs_infer)(&v) {
                    if !self.is_global() {
                        let v = unsafe {
                            mem::transmute(v)
                        };
                        let i = ($alloc_to_ret)(self.global_interners.arena.$alloc_method(v));
                        self.global_interners.$name.borrow_mut().insert(Interned(i));
                        return i;
                    }
                } else {
                    // Make sure we don't end up with inference
                    // types/regions in the global tcx.
                    if self.is_global() {
                        bug!("Attempted to intern `{:?}` which contains \
                              inference types/regions in the global type context",
                             v);
                    }
                }

                let i = ($alloc_to_ret)(self.interners.arena.$alloc_method(v));
                self.interners.$name.borrow_mut().insert(Interned(i));
                i
            }
        }
    }
}

macro_rules! direct_interners {
    ($lt_tcx:tt, $($name:ident: $method:ident($needs_infer:expr) -> $ty:ty),+) => {
        $(impl<$lt_tcx> PartialEq for Interned<$lt_tcx, $ty> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<$lt_tcx> Eq for Interned<$lt_tcx, $ty> {}

        impl<$lt_tcx> Hash for Interned<$lt_tcx, $ty> {
            fn hash<H: Hasher>(&self, s: &mut H) {
                self.0.hash(s)
            }
        }

        intern_method!($lt_tcx, $name: $method($ty, alloc, |x| x, |x| x, $needs_infer) -> $ty);)+
    }
}

pub fn keep_local<'tcx, T: ty::TypeFoldable<'tcx>>(x: &T) -> bool {
    x.has_type_flags(ty::TypeFlags::KEEP_IN_LOCAL_TCX)
}

direct_interners!('tcx,
    region: mk_region(|r| {
        match r {
            &ty::ReVar(_) | &ty::ReSkolemized(..) => true,
            _ => false
        }
    }) -> RegionKind,
    const_: mk_const(|c: &Const| keep_local(&c.ty) || keep_local(&c.val)) -> Const<'tcx>
);

macro_rules! slice_interners {
    ($($field:ident: $method:ident($ty:ident)),+) => (
        $(intern_method!('tcx, $field: $method(&[$ty<'tcx>], alloc_slice, Deref::deref,
                                               |xs: &[$ty]| -> &Slice<$ty> {
            unsafe { mem::transmute(xs) }
        }, |xs: &[$ty]| xs.iter().any(keep_local)) -> Slice<$ty<'tcx>>);)+
    )
}

slice_interners!(
    existential_predicates: _intern_existential_predicates(ExistentialPredicate),
    predicates: _intern_predicates(Predicate),
    type_list: _intern_type_list(Ty),
    substs: _intern_substs(Kind)
);

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        assert_eq!(sig.unsafety(), hir::Unsafety::Normal);
        self.mk_fn_ptr(sig.map_bound(|sig| ty::FnSig {
            unsafety: hir::Unsafety::Unsafe,
            ..sig
        }))
    }

    /// Given a closure signature `sig`, returns an equivalent `fn`
    /// type with the same signature. Detuples and so forth -- so
    /// e.g. if we have a sig with `Fn<(u32, i32)>` then you would get
    /// a `fn(u32, i32)`.
    pub fn coerce_closure_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        let converted_sig = sig.map_bound(|s| {
            let params_iter = match s.inputs()[0].sty {
                ty::TyTuple(params, _) => {
                    params.into_iter().cloned()
                }
                _ => bug!(),
            };
            self.mk_fn_sig(
                params_iter,
                s.output(),
                s.variadic,
                hir::Unsafety::Normal,
                abi::Abi::Rust,
            )
        });

        self.mk_fn_ptr(converted_sig)
    }

    // Interns a type/name combination, stores the resulting box in cx.interners,
    // and returns the box as cast to an unsafe ptr (see comments for Ty above).
    pub fn mk_ty(self, st: TypeVariants<'tcx>) -> Ty<'tcx> {
        let global_interners = if !self.is_global() {
            Some(&self.global_interners)
        } else {
            None
        };
        self.interners.intern_ty(st, global_interners)
    }

    pub fn mk_mach_int(self, tm: ast::IntTy) -> Ty<'tcx> {
        match tm {
            ast::IntTy::Isize   => self.types.isize,
            ast::IntTy::I8   => self.types.i8,
            ast::IntTy::I16  => self.types.i16,
            ast::IntTy::I32  => self.types.i32,
            ast::IntTy::I64  => self.types.i64,
            ast::IntTy::I128  => self.types.i128,
        }
    }

    pub fn mk_mach_uint(self, tm: ast::UintTy) -> Ty<'tcx> {
        match tm {
            ast::UintTy::Usize   => self.types.usize,
            ast::UintTy::U8   => self.types.u8,
            ast::UintTy::U16  => self.types.u16,
            ast::UintTy::U32  => self.types.u32,
            ast::UintTy::U64  => self.types.u64,
            ast::UintTy::U128  => self.types.u128,
        }
    }

    pub fn mk_mach_float(self, tm: ast::FloatTy) -> Ty<'tcx> {
        match tm {
            ast::FloatTy::F32  => self.types.f32,
            ast::FloatTy::F64  => self.types.f64,
        }
    }

    pub fn mk_str(self) -> Ty<'tcx> {
        self.mk_ty(TyStr)
    }

    pub fn mk_static_str(self) -> Ty<'tcx> {
        self.mk_imm_ref(self.types.re_static, self.mk_str())
    }

    pub fn mk_adt(self, def: &'tcx AdtDef, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // take a copy of substs so that we own the vectors inside
        self.mk_ty(TyAdt(def, substs))
    }

    pub fn mk_foreign(self, def_id: DefId) -> Ty<'tcx> {
        self.mk_ty(TyForeign(def_id))
    }

    pub fn mk_box(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let def_id = self.require_lang_item(lang_items::OwnedBoxLangItem);
        let adt_def = self.adt_def(def_id);
        let substs = self.mk_substs(iter::once(Kind::from(ty)));
        self.mk_ty(TyAdt(adt_def, substs))
    }

    pub fn mk_ptr(self, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRawPtr(tm))
    }

    pub fn mk_ref(self, r: Region<'tcx>, tm: TypeAndMut<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyRef(r, tm))
    }

    pub fn mk_mut_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ref(self, r: Region<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ref(r, TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_mut_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutMutable})
    }

    pub fn mk_imm_ptr(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ptr(TypeAndMut {ty: ty, mutbl: hir::MutImmutable})
    }

    pub fn mk_nil_ptr(self) -> Ty<'tcx> {
        self.mk_imm_ptr(self.mk_nil())
    }

    pub fn mk_array(self, ty: Ty<'tcx>, n: u64) -> Ty<'tcx> {
        self.mk_ty(TyArray(ty, self.mk_const(ty::Const {
            val: ConstVal::Value(Value::ByVal(PrimVal::Bytes(n.into()))),
            ty: self.types.usize
        })))
    }

    pub fn mk_slice(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TySlice(ty))
    }

    pub fn intern_tup(self, ts: &[Ty<'tcx>], defaulted: bool) -> Ty<'tcx> {
        self.mk_ty(TyTuple(self.intern_type_list(ts), defaulted))
    }

    pub fn mk_tup<I: InternAs<[Ty<'tcx>], Ty<'tcx>>>(self, iter: I,
                                                     defaulted: bool) -> I::Output {
        iter.intern_with(|ts| self.mk_ty(TyTuple(self.intern_type_list(ts), defaulted)))
    }

    pub fn mk_nil(self) -> Ty<'tcx> {
        self.intern_tup(&[], false)
    }

    pub fn mk_diverging_default(self) -> Ty<'tcx> {
        if self.features().never_type {
            self.types.never
        } else {
            self.intern_tup(&[], true)
        }
    }

    pub fn mk_bool(self) -> Ty<'tcx> {
        self.mk_ty(TyBool)
    }

    pub fn mk_fn_def(self, def_id: DefId,
                     substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyFnDef(def_id, substs))
    }

    pub fn mk_fn_ptr(self, fty: PolyFnSig<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyFnPtr(fty))
    }

    pub fn mk_dynamic(
        self,
        obj: ty::Binder<&'tcx Slice<ExistentialPredicate<'tcx>>>,
        reg: ty::Region<'tcx>
    ) -> Ty<'tcx> {
        self.mk_ty(TyDynamic(obj, reg))
    }

    pub fn mk_projection(self,
                         item_def_id: DefId,
                         substs: &'tcx Substs<'tcx>)
        -> Ty<'tcx> {
            self.mk_ty(TyProjection(ProjectionTy {
                item_def_id,
                substs,
            }))
        }

    pub fn mk_closure(self,
                      closure_id: DefId,
                      substs: ClosureSubsts<'tcx>)
                      -> Ty<'tcx> {
        self.mk_closure_from_closure_substs(closure_id, substs)
    }

    pub fn mk_closure_from_closure_substs(self,
                                          closure_id: DefId,
                                          closure_substs: ClosureSubsts<'tcx>)
                                          -> Ty<'tcx> {
        self.mk_ty(TyClosure(closure_id, closure_substs))
    }

    pub fn mk_generator(self,
                        id: DefId,
                        closure_substs: ClosureSubsts<'tcx>,
                        interior: GeneratorInterior<'tcx>)
                        -> Ty<'tcx> {
        self.mk_ty(TyGenerator(id, closure_substs, interior))
    }

    pub fn mk_generator_witness(self, types: ty::Binder<&'tcx Slice<Ty<'tcx>>>) -> Ty<'tcx> {
        self.mk_ty(TyGeneratorWitness(types))
    }

    pub fn mk_var(self, v: TyVid) -> Ty<'tcx> {
        self.mk_infer(TyVar(v))
    }

    pub fn mk_int_var(self, v: IntVid) -> Ty<'tcx> {
        self.mk_infer(IntVar(v))
    }

    pub fn mk_float_var(self, v: FloatVid) -> Ty<'tcx> {
        self.mk_infer(FloatVar(v))
    }

    pub fn mk_infer(self, it: InferTy) -> Ty<'tcx> {
        self.mk_ty(TyInfer(it))
    }

    pub fn mk_param(self,
                    index: u32,
                    name: Name) -> Ty<'tcx> {
        self.mk_ty(TyParam(ParamTy { idx: index, name: name }))
    }

    pub fn mk_self_type(self) -> Ty<'tcx> {
        self.mk_param(0, keywords::SelfType.name())
    }

    pub fn mk_param_from_def(self, def: &ty::TypeParameterDef) -> Ty<'tcx> {
        self.mk_param(def.index, def.name)
    }

    pub fn mk_anon(self, def_id: DefId, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        self.mk_ty(TyAnon(def_id, substs))
    }

    pub fn intern_existential_predicates(self, eps: &[ExistentialPredicate<'tcx>])
        -> &'tcx Slice<ExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(eps.windows(2).all(|w| w[0].cmp(self, &w[1]) != Ordering::Greater));
        self._intern_existential_predicates(eps)
    }

    pub fn intern_predicates(self, preds: &[Predicate<'tcx>])
        -> &'tcx Slice<Predicate<'tcx>> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        if preds.len() == 0 {
            // The macro-generated method below asserts we don't intern an empty slice.
            Slice::empty()
        } else {
            self._intern_predicates(preds)
        }
    }

    pub fn intern_type_list(self, ts: &[Ty<'tcx>]) -> &'tcx Slice<Ty<'tcx>> {
        if ts.len() == 0 {
            Slice::empty()
        } else {
            self._intern_type_list(ts)
        }
    }

    pub fn intern_substs(self, ts: &[Kind<'tcx>]) -> &'tcx Slice<Kind<'tcx>> {
        if ts.len() == 0 {
            Slice::empty()
        } else {
            self._intern_substs(ts)
        }
    }

    pub fn mk_fn_sig<I>(self,
                        inputs: I,
                        output: I::Item,
                        variadic: bool,
                        unsafety: hir::Unsafety,
                        abi: abi::Abi)
        -> <I::Item as InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>>::Output
        where I: Iterator,
              I::Item: InternIteratorElement<Ty<'tcx>, ty::FnSig<'tcx>>
    {
        inputs.chain(iter::once(output)).intern_with(|xs| ty::FnSig {
            inputs_and_output: self.intern_type_list(xs),
            variadic, unsafety, abi
        })
    }

    pub fn mk_existential_predicates<I: InternAs<[ExistentialPredicate<'tcx>],
                                     &'tcx Slice<ExistentialPredicate<'tcx>>>>(self, iter: I)
                                     -> I::Output {
        iter.intern_with(|xs| self.intern_existential_predicates(xs))
    }

    pub fn mk_predicates<I: InternAs<[Predicate<'tcx>],
                                     &'tcx Slice<Predicate<'tcx>>>>(self, iter: I)
                                     -> I::Output {
        iter.intern_with(|xs| self.intern_predicates(xs))
    }

    pub fn mk_type_list<I: InternAs<[Ty<'tcx>],
                        &'tcx Slice<Ty<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_type_list(xs))
    }

    pub fn mk_substs<I: InternAs<[Kind<'tcx>],
                     &'tcx Slice<Kind<'tcx>>>>(self, iter: I) -> I::Output {
        iter.intern_with(|xs| self.intern_substs(xs))
    }

    pub fn mk_substs_trait(self,
                     s: Ty<'tcx>,
                     t: &[Ty<'tcx>])
                    -> &'tcx Substs<'tcx>
    {
        self.mk_substs(iter::once(s).chain(t.into_iter().cloned()).map(Kind::from))
    }

    pub fn lint_node<S: Into<MultiSpan>>(self,
                                         lint: &'static Lint,
                                         id: NodeId,
                                         span: S,
                                         msg: &str) {
        self.struct_span_lint_node(lint, id, span.into(), msg).emit()
    }

    pub fn lint_node_note<S: Into<MultiSpan>>(self,
                                              lint: &'static Lint,
                                              id: NodeId,
                                              span: S,
                                              msg: &str,
                                              note: &str) {
        let mut err = self.struct_span_lint_node(lint, id, span.into(), msg);
        err.note(note);
        err.emit()
    }

    pub fn lint_level_at_node(self, lint: &'static Lint, mut id: NodeId)
        -> (lint::Level, lint::LintSource)
    {
        // Right now we insert a `with_ignore` node in the dep graph here to
        // ignore the fact that `lint_levels` below depends on the entire crate.
        // For now this'll prevent false positives of recompiling too much when
        // anything changes.
        //
        // Once red/green incremental compilation lands we should be able to
        // remove this because while the crate changes often the lint level map
        // will change rarely.
        self.dep_graph.with_ignore(|| {
            let sets = self.lint_levels(LOCAL_CRATE);
            loop {
                let hir_id = self.hir.definitions().node_to_hir_id(id);
                if let Some(pair) = sets.level_and_source(lint, hir_id, self.sess) {
                    return pair
                }
                let next = self.hir.get_parent_node(id);
                if next == id {
                    bug!("lint traversal reached the root of the crate");
                }
                id = next;
            }
        })
    }

    pub fn struct_span_lint_node<S: Into<MultiSpan>>(self,
                                                     lint: &'static Lint,
                                                     id: NodeId,
                                                     span: S,
                                                     msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, id);
        lint::struct_lint_level(self.sess, lint, level, src, Some(span.into()), msg)
    }

    pub fn struct_lint_node(self, lint: &'static Lint, id: NodeId, msg: &str)
        -> DiagnosticBuilder<'tcx>
    {
        let (level, src) = self.lint_level_at_node(lint, id);
        lint::struct_lint_level(self.sess, lint, level, src, None, msg)
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<Lrc<StableVec<TraitCandidate>>> {
        self.in_scope_traits_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn named_region(self, id: HirId) -> Option<resolve_lifetime::Region> {
        self.named_region_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }

    pub fn is_late_bound(self, id: HirId) -> bool {
        self.is_late_bound_map(id.owner)
            .map(|set| set.contains(&id.local_id))
            .unwrap_or(false)
    }

    pub fn object_lifetime_defaults(self, id: HirId)
        -> Option<Lrc<Vec<ObjectLifetimeDefault>>>
    {
        self.object_lifetime_defaults_map(id.owner)
            .and_then(|map| map.get(&id.local_id).cloned())
    }
}

pub trait InternAs<T: ?Sized, R> {
    type Output;
    fn intern_with<F>(self, f: F) -> Self::Output
        where F: FnOnce(&T) -> R;
}

impl<I, T, R, E> InternAs<[T], R> for I
    where E: InternIteratorElement<T, R>,
          I: Iterator<Item=E> {
    type Output = E::Output;
    fn intern_with<F>(self, f: F) -> Self::Output
        where F: FnOnce(&[T]) -> R {
        E::intern_with(self, f)
    }
}

pub trait InternIteratorElement<T, R>: Sized {
    type Output;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output;
}

impl<T, R> InternIteratorElement<T, R> for T {
    type Output = R;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.collect::<AccumulateVec<[_; 8]>>())
    }
}

impl<'a, T, R> InternIteratorElement<T, R> for &'a T
    where T: Clone + 'a
{
    type Output = R;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        f(&iter.cloned().collect::<AccumulateVec<[_; 8]>>())
    }
}

impl<T, R, E> InternIteratorElement<T, R> for Result<T, E> {
    type Output = Result<R, E>;
    fn intern_with<I: Iterator<Item=Self>, F: FnOnce(&[T]) -> R>(iter: I, f: F) -> Self::Output {
        Ok(f(&iter.collect::<Result<AccumulateVec<[_; 8]>, _>>()?))
    }
}

pub fn provide(providers: &mut ty::maps::Providers) {
    // FIXME(#44234) - almost all of these queries have no sub-queries and
    // therefore no actual inputs, they're just reading tables calculated in
    // resolve! Does this work? Unsure! That's what the issue is about
    providers.in_scope_traits_map = |tcx, id| tcx.gcx.trait_map.get(&id).cloned();
    providers.module_exports = |tcx, id| tcx.gcx.export_map.get(&id).cloned();
    providers.crate_name = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.crate_name
    };
    providers.get_lang_items = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        // FIXME(#42293) Right now we insert a `with_ignore` node in the dep
        // graph here to ignore the fact that `get_lang_items` below depends on
        // the entire crate.  For now this'll prevent false positives of
        // recompiling too much when anything changes.
        //
        // Once red/green incremental compilation lands we should be able to
        // remove this because while the crate changes often the lint level map
        // will change rarely.
        tcx.dep_graph.with_ignore(|| Lrc::new(middle::lang_items::collect(tcx)))
    };
    providers.freevars = |tcx, id| tcx.gcx.freevars.get(&id).cloned();
    providers.maybe_unused_trait_import = |tcx, id| {
        tcx.maybe_unused_trait_imports.contains(&id)
    };
    providers.maybe_unused_extern_crates = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.maybe_unused_extern_crates.clone())
    };

    providers.stability_index = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(stability::Index::new(tcx))
    };
    providers.lookup_stability = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir.definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_stability(id)
    };
    providers.lookup_deprecation_entry = |tcx, id| {
        assert_eq!(id.krate, LOCAL_CRATE);
        let id = tcx.hir.definitions().def_index_to_hir_id(id.index);
        tcx.stability().local_deprecation_entry(id)
    };
    providers.extern_mod_stmt_cnum = |tcx, id| {
        let id = tcx.hir.as_local_node_id(id).unwrap();
        tcx.cstore.extern_mod_stmt_cnum_untracked(id)
    };
    providers.all_crate_nums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.cstore.crates_untracked())
    };
    providers.postorder_cnums = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.cstore.postorder_cnums_untracked())
    };
    providers.output_filenames = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.output_filenames.clone()
    };
    providers.has_copy_closures = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.features().copy_closures
    };
    providers.has_clone_closures = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        tcx.features().clone_closures
    };
    providers.fully_normalize_monormophic_ty = |tcx, ty| {
        tcx.fully_normalize_associated_types_in(&ty)
    };
    providers.features_query = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Lrc::new(tcx.sess.features_untracked().clone())
    };
    providers.is_panic_runtime = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir.krate_attrs(), "panic_runtime")
    };
    providers.is_compiler_builtins = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        attr::contains_name(tcx.hir.krate_attrs(), "compiler_builtins")
    };
}
