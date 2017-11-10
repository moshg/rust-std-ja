// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::mir::Mir;
use rustc::mir::transform::MirSource;
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc::ty::steal::Steal;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::DefIdSet;
use std::borrow::Cow;
use std::rc::Rc;
use syntax::ast;
use syntax_pos::Span;

pub mod add_validation;
pub mod clean_end_regions;
pub mod check_unsafety;
pub mod simplify_branches;
pub mod simplify;
pub mod erase_regions;
pub mod no_landing_pads;
pub mod type_check;
pub mod rustc_peek;
pub mod elaborate_drops;
pub mod add_call_guards;
pub mod promote_consts;
pub mod qualify_consts;
pub mod dump_mir;
pub mod deaggregator;
pub mod instcombine;
pub mod copy_prop;
pub mod generator;
pub mod inline;
pub mod nll;

pub(crate) fn provide(providers: &mut Providers) {
    self::qualify_consts::provide(providers);
    self::check_unsafety::provide(providers);
    *providers = Providers {
        mir_keys,
        mir_built,
        mir_const,
        mir_validated,
        optimized_mir,
        is_mir_available,
        ..*providers
    };
}

fn is_mir_available<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> bool {
    tcx.mir_keys(def_id.krate).contains(&def_id)
}

/// Finds the full set of def-ids within the current crate that have
/// MIR associated with them.
fn mir_keys<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, krate: CrateNum)
                      -> Rc<DefIdSet> {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = DefIdSet();

    // All body-owners have MIR associated with them.
    set.extend(tcx.body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx: 'a> {
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        set: &'a mut DefIdSet,
    }
    impl<'a, 'tcx> Visitor<'tcx> for GatherCtors<'a, 'tcx> {
        fn visit_variant_data(&mut self,
                              v: &'tcx hir::VariantData,
                              _: ast::Name,
                              _: &'tcx hir::Generics,
                              _: ast::NodeId,
                              _: Span) {
            if let hir::VariantData::Tuple(_, node_id) = *v {
                self.set.insert(self.tcx.hir.local_def_id(node_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        fn nested_visit_map<'b>(&'b mut self) -> NestedVisitorMap<'b, 'tcx> {
            NestedVisitorMap::None
        }
    }
    tcx.hir.krate().visit_all_item_likes(&mut GatherCtors {
        tcx,
        set: &mut set,
    }.as_deep_visitor());

    Rc::new(set)
}

fn mir_built<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let mir = build::mir_build(tcx, def_id);
    tcx.alloc_steal_mir(mir)
}

/// Generates a default name for the pass based on the name of the
/// type `T`.
pub fn default_name<T: ?Sized>() -> Cow<'static, str> {
    let name = unsafe { ::std::intrinsics::type_name::<T>() };
    if let Some(tail) = name.rfind(":") {
        Cow::from(&name[tail+1..])
    } else {
        Cow::from(name)
    }
}

/// A streamlined trait that you can implement to create a pass; the
/// pass will be named after the type, and it will consist of a main
/// loop that goes over each available MIR and applies `run_pass`.
pub trait MirPass {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        default_name::<Self>()
    }

    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>);
}

pub macro run_passes($tcx:ident, $mir:ident, $source:ident, $suite_index:expr; $($pass:expr,)*) {{
    let suite_index: usize = $suite_index;
    let run_passes = |mir: &mut _, source| {
        let mut index = 0;
        let mut run_pass = |pass: &MirPass| {
            let run_hooks = |mir: &_, index, is_after| {
                dump_mir::on_mir_pass($tcx, &format_args!("{:03}-{:03}", suite_index, index),
                                      &pass.name(), source, mir, is_after);
            };
            run_hooks(mir, index, false);
            pass.run_pass($tcx, source, mir);
            run_hooks(mir, index, true);

            index += 1;
        };
        $(run_pass(&$pass);)*
    };
    run_passes(&mut $mir, $source);

    for (index, promoted_mir) in $mir.promoted.iter_enumerated_mut() {
        run_passes(promoted_mir, MirSource::Promoted($source.item_id(), index));

        // Let's make sure we don't miss any nested instances
        assert!(promoted_mir.promoted.is_empty());
    }
}}

fn mir_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    // Unsafety check uses the raw mir, so make sure it is run
    let _ = tcx.unsafety_check_result(def_id);

    let mut mir = tcx.mir_built(def_id).steal();
    let source = MirSource::from_local_def_id(tcx, def_id);
    run_passes![tcx, mir, source, 0;
        // Remove all `EndRegion` statements that are not involved in borrows.
        clean_end_regions::CleanEndRegions,

        // What we need to do constant evaluation.
        simplify::SimplifyCfg::new("initial"),
        type_check::TypeckMir,
        rustc_peek::SanityCheck,
    ];
    tcx.alloc_steal_mir(mir)
}

fn mir_validated<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let source = MirSource::from_local_def_id(tcx, def_id);
    if let MirSource::Const(_) = source {
        // Ensure that we compute the `mir_const_qualif` for constants at
        // this point, before we steal the mir-const result.
        let _ = tcx.mir_const_qualif(def_id);
    }

    let mut mir = tcx.mir_const(def_id).steal();
    run_passes![tcx, mir, source, 1;
        // What we need to run borrowck etc.
        qualify_consts::QualifyAndPromoteConstants,
        simplify::SimplifyCfg::new("qualify-consts"),
    ];
    tcx.alloc_steal_mir(mir)
}

fn optimized_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Mir<'tcx> {
    // (Mir-)Borrowck uses `mir_validated`, so we have to force it to
    // execute before we can steal.
    let _ = tcx.mir_borrowck(def_id);
    let _ = tcx.borrowck(def_id);

    let mut mir = tcx.mir_validated(def_id).steal();
    let source = MirSource::from_local_def_id(tcx, def_id);
    run_passes![tcx, mir, source, 2;
        no_landing_pads::NoLandingPads,
        simplify_branches::SimplifyBranches::new("initial"),

        // These next passes must be executed together
        add_call_guards::CriticalCallEdges,
        elaborate_drops::ElaborateDrops,
        no_landing_pads::NoLandingPads,
        // AddValidation needs to run after ElaborateDrops and before EraseRegions, and it needs
        // an AllCallEdges pass right before it.
        add_call_guards::AllCallEdges,
        add_validation::AddValidation,
        simplify::SimplifyCfg::new("elaborate-drops"),
        // No lifetime analysis based on borrowing can be done from here on out.

        // From here on out, regions are gone.
        erase_regions::EraseRegions,

        // Optimizations begin.
        inline::Inline,
        instcombine::InstCombine,
        deaggregator::Deaggregator,
        copy_prop::CopyPropagation,
        simplify::SimplifyLocals,

        generator::StateTransform,
        add_call_guards::CriticalCallEdges,
        dump_mir::Marker("PreTrans"),
    ];
    tcx.alloc_mir(mir)
}
