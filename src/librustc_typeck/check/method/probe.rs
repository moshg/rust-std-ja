// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::MethodError;
use super::NoMatchData;
use super::{CandidateSource, ImplSource, TraitSource};
use super::suggest;

use check::FnCtxt;
use hir::def_id::DefId;
use hir::def::Def;
use namespace::Namespace;
use rustc::ty::subst::{Subst, Substs};
use rustc::traits::{self, ObligationCause};
use rustc::ty::{self, Ty, ToPolyTraitRef, ToPredicate, TraitRef, TypeFoldable};
use rustc::infer::type_variable::TypeVariableOrigin;
use rustc::util::nodemap::FxHashSet;
use rustc::infer::{self, InferOk};
use syntax::ast;
use syntax::util::lev_distance::{lev_distance, find_best_match_for_name};
use syntax_pos::Span;
use rustc::hir;
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use std::cmp::max;

use self::CandidateKind::*;
pub use self::PickKind::*;

/// Boolean flag used to indicate if this search is for a suggestion
/// or not.  If true, we can allow ambiguity and so forth.
pub struct IsSuggestion(pub bool);

struct ProbeContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    span: Span,
    mode: Mode,
    method_name: Option<ast::Name>,
    return_type: Option<Ty<'tcx>>,
    steps: Rc<Vec<CandidateStep<'tcx>>>,
    inherent_candidates: Vec<Candidate<'tcx>>,
    extension_candidates: Vec<Candidate<'tcx>>,
    impl_dups: FxHashSet<DefId>,

    /// Collects near misses when the candidate functions are missing a `self` keyword and is only
    /// used for error reporting
    static_candidates: Vec<CandidateSource>,

    /// When probing for names, include names that are close to the
    /// requested name (by Levensthein distance)
    allow_similar_names: bool,

    /// Some(candidate) if there is a private candidate
    private_candidate: Option<Def>,

    /// Collects near misses when trait bounds for type parameters are unsatisfied and is only used
    /// for error reporting
    unsatisfied_predicates: Vec<TraitRef<'tcx>>,
}

impl<'a, 'gcx, 'tcx> Deref for ProbeContext<'a, 'gcx, 'tcx> {
    type Target = FnCtxt<'a, 'gcx, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

#[derive(Debug)]
struct CandidateStep<'tcx> {
    self_ty: Ty<'tcx>,
    autoderefs: usize,
    // true if the type results from a dereference of a raw pointer.
    // when assembling candidates, we include these steps, but not when
    // picking methods. This so that if we have `foo: *const Foo` and `Foo` has methods
    // `fn by_raw_ptr(self: *const Self)` and `fn by_ref(&self)`, then
    // `foo.by_raw_ptr()` will work and `foo.by_ref()` won't.
    from_unsafe_deref: bool,
    unsize: bool,
}

#[derive(Debug)]
struct Candidate<'tcx> {
    xform_self_ty: Ty<'tcx>,
    xform_ret_ty: Option<Ty<'tcx>>,
    item: ty::AssociatedItem,
    kind: CandidateKind<'tcx>,
    import_id: Option<ast::NodeId>,
}

#[derive(Debug)]
enum CandidateKind<'tcx> {
    InherentImplCandidate(&'tcx Substs<'tcx>,
                          // Normalize obligations
                          Vec<traits::PredicateObligation<'tcx>>),
    ObjectCandidate,
    TraitCandidate(ty::TraitRef<'tcx>),
    WhereClauseCandidate(// Trait
                         ty::PolyTraitRef<'tcx>),
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum ProbeResult {
    NoMatch,
    BadReturnType,
    Match,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Pick<'tcx> {
    pub item: ty::AssociatedItem,
    pub kind: PickKind<'tcx>,
    pub import_id: Option<ast::NodeId>,

    // Indicates that the source expression should be autoderef'd N times
    //
    // A = expr | *expr | **expr | ...
    pub autoderefs: usize,

    // Indicates that an autoref is applied after the optional autoderefs
    //
    // B = A | &A | &mut A
    pub autoref: Option<hir::Mutability>,

    // Indicates that the source expression should be "unsized" to a
    // target type. This should probably eventually go away in favor
    // of just coercing method receivers.
    //
    // C = B | unsize(B)
    pub unsize: Option<Ty<'tcx>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickKind<'tcx> {
    InherentImplPick,
    ObjectPick,
    TraitPick,
    WhereClausePick(// Trait
                    ty::PolyTraitRef<'tcx>),
}

pub type PickResult<'tcx> = Result<Pick<'tcx>, MethodError<'tcx>>;

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Mode {
    // An expression of the form `receiver.method_name(...)`.
    // Autoderefs are performed on `receiver`, lookup is done based on the
    // `self` argument  of the method, and static methods aren't considered.
    MethodCall,
    // An expression of the form `Type::item` or `<T>::item`.
    // No autoderefs are performed, lookup is done based on the type each
    // implementation is for, and static methods are included.
    Path,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum ProbeScope {
    // Assemble candidates coming only from traits in scope.
    TraitsInScope,

    // Assemble candidates coming from all traits.
    AllTraits,
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    /// This is used to offer suggestions to users. It returns methods
    /// that could have been called which have the desired return
    /// type. Some effort is made to rule out methods that, if called,
    /// would result in an error (basically, the same criteria we
    /// would use to decide if a method is a plausible fit for
    /// ambiguity purposes).
    pub fn probe_for_return_type(&self,
                                 span: Span,
                                 mode: Mode,
                                 return_type: Ty<'tcx>,
                                 self_ty: Ty<'tcx>,
                                 scope_expr_id: ast::NodeId)
                                 -> Vec<ty::AssociatedItem> {
        debug!("probe(self_ty={:?}, return_type={}, scope_expr_id={})",
               self_ty,
               return_type,
               scope_expr_id);
        let method_names =
            self.probe_op(span, mode, None, Some(return_type), IsSuggestion(true),
                          self_ty, scope_expr_id, ProbeScope::TraitsInScope,
                          |probe_cx| Ok(probe_cx.candidate_method_names()))
                .unwrap_or(vec![]);
         method_names
             .iter()
             .flat_map(|&method_name| {
                 self.probe_op(
                     span, mode, Some(method_name), Some(return_type),
                     IsSuggestion(true), self_ty, scope_expr_id,
                     ProbeScope::TraitsInScope, |probe_cx| probe_cx.pick()
                 ).ok().map(|pick| pick.item)
             })
            .collect()
    }

    pub fn probe_for_name(&self,
                          span: Span,
                          mode: Mode,
                          item_name: ast::Name,
                          is_suggestion: IsSuggestion,
                          self_ty: Ty<'tcx>,
                          scope_expr_id: ast::NodeId,
                          scope: ProbeScope)
                          -> PickResult<'tcx> {
        debug!("probe(self_ty={:?}, item_name={}, scope_expr_id={})",
               self_ty,
               item_name,
               scope_expr_id);
        self.probe_op(span,
                      mode,
                      Some(item_name),
                      None,
                      is_suggestion,
                      self_ty,
                      scope_expr_id,
                      scope,
                      |probe_cx| probe_cx.pick())
    }

    fn probe_op<OP,R>(&'a self,
                      span: Span,
                      mode: Mode,
                      method_name: Option<ast::Name>,
                      return_type: Option<Ty<'tcx>>,
                      is_suggestion: IsSuggestion,
                      self_ty: Ty<'tcx>,
                      scope_expr_id: ast::NodeId,
                      scope: ProbeScope,
                      op: OP)
                      -> Result<R, MethodError<'tcx>>
        where OP: FnOnce(ProbeContext<'a, 'gcx, 'tcx>) -> Result<R, MethodError<'tcx>>
    {
        // FIXME(#18741) -- right now, creating the steps involves evaluating the
        // `*` operator, which registers obligations that then escape into
        // the global fulfillment context and thus has global
        // side-effects. This is a bit of a pain to refactor. So just let
        // it ride, although it's really not great, and in fact could I
        // think cause spurious errors. Really though this part should
        // take place in the `self.probe` below.
        let steps = if mode == Mode::MethodCall {
            match self.create_steps(span, self_ty, is_suggestion) {
                Some(steps) => steps,
                None => {
                    return Err(MethodError::NoMatch(NoMatchData::new(Vec::new(),
                                                                     Vec::new(),
                                                                     Vec::new(),
                                                                     None,
                                                                     mode)))
                }
            }
        } else {
            vec![CandidateStep {
                     self_ty,
                     autoderefs: 0,
                     from_unsafe_deref: false,
                     unsize: false,
                 }]
        };

        debug!("ProbeContext: steps for self_ty={:?} are {:?}",
               self_ty,
               steps);

        // this creates one big transaction so that all type variables etc
        // that we create during the probe process are removed later
        self.probe(|_| {
            let mut probe_cx =
                ProbeContext::new(self, span, mode, method_name, return_type, Rc::new(steps));

            probe_cx.assemble_inherent_candidates();
            match scope {
                ProbeScope::TraitsInScope =>
                    probe_cx.assemble_extension_candidates_for_traits_in_scope(scope_expr_id)?,
                ProbeScope::AllTraits =>
                    probe_cx.assemble_extension_candidates_for_all_traits()?,
            };
            op(probe_cx)
        })
    }

    fn create_steps(&self,
                    span: Span,
                    self_ty: Ty<'tcx>,
                    is_suggestion: IsSuggestion)
                    -> Option<Vec<CandidateStep<'tcx>>> {
        // FIXME: we don't need to create the entire steps in one pass

        let mut autoderef = self.autoderef(span, self_ty).include_raw_pointers();
        let mut reached_raw_pointer = false;
        let mut steps: Vec<_> = autoderef.by_ref()
            .map(|(ty, d)| {
                let step = CandidateStep {
                    self_ty: ty,
                    autoderefs: d,
                    from_unsafe_deref: reached_raw_pointer,
                    unsize: false,
                };
                if let ty::TyRawPtr(_) = ty.sty {
                    // all the subsequent steps will be from_unsafe_deref
                    reached_raw_pointer = true;
                }
                step
            })
            .collect();

        let final_ty = autoderef.maybe_ambiguous_final_ty();
        match final_ty.sty {
            ty::TyInfer(ty::TyVar(_)) => {
                // Ended in an inference variable. If we are doing
                // a real method lookup, this is a hard error (it's an
                // ambiguity and we can't make progress).
                if !is_suggestion.0 {
                    let t = self.structurally_resolved_type(span, final_ty);
                    assert_eq!(t, self.tcx.types.err);
                    return None
                } else {
                    // If we're just looking for suggestions,
                    // though, ambiguity is no big thing, we can
                    // just ignore it.
                }
            }
            ty::TyArray(elem_ty, _) => {
                let dereferences = steps.len() - 1;

                steps.push(CandidateStep {
                    self_ty: self.tcx.mk_slice(elem_ty),
                    autoderefs: dereferences,
                    // this could be from an unsafe deref if we had
                    // a *mut/const [T; N]
                    from_unsafe_deref: reached_raw_pointer,
                    unsize: true,
                });
            }
            ty::TyError => return None,
            _ => (),
        }

        debug!("create_steps: steps={:?}", steps);

        Some(steps)
    }
}

impl<'a, 'gcx, 'tcx> ProbeContext<'a, 'gcx, 'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
           span: Span,
           mode: Mode,
           method_name: Option<ast::Name>,
           return_type: Option<Ty<'tcx>>,
           steps: Rc<Vec<CandidateStep<'tcx>>>)
           -> ProbeContext<'a, 'gcx, 'tcx> {
        ProbeContext {
            fcx,
            span,
            mode,
            method_name,
            return_type,
            inherent_candidates: Vec::new(),
            extension_candidates: Vec::new(),
            impl_dups: FxHashSet(),
            steps: steps,
            static_candidates: Vec::new(),
            allow_similar_names: false,
            private_candidate: None,
            unsatisfied_predicates: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.inherent_candidates.clear();
        self.extension_candidates.clear();
        self.impl_dups.clear();
        self.static_candidates.clear();
        self.private_candidate = None;
    }

    ///////////////////////////////////////////////////////////////////////////
    // CANDIDATE ASSEMBLY

    fn push_candidate(&mut self,
                      candidate: Candidate<'tcx>,
                      is_inherent: bool)
    {
        let is_accessible = if let Some(name) = self.method_name {
            let item = candidate.item;
            let def_scope = self.tcx.adjust(name, item.container.id(), self.body_id).1;
            item.vis.is_accessible_from(def_scope, self.tcx)
        } else {
            true
        };
        if is_accessible {
            if is_inherent {
                self.inherent_candidates.push(candidate);
            } else {
                self.extension_candidates.push(candidate);
            }
        } else if self.private_candidate.is_none() {
            self.private_candidate = Some(candidate.item.def());
        }
    }

    fn assemble_inherent_candidates(&mut self) {
        let steps = self.steps.clone();
        for step in steps.iter() {
            self.assemble_probe(step.self_ty);
        }
    }

    fn assemble_probe(&mut self, self_ty: Ty<'tcx>) {
        debug!("assemble_probe: self_ty={:?}", self_ty);
        let lang_items = self.tcx.lang_items();

        match self_ty.sty {
            ty::TyDynamic(ref data, ..) => {
                if let Some(p) = data.principal() {
                    self.assemble_inherent_candidates_from_object(self_ty, p);
                    self.assemble_inherent_impl_candidates_for_type(p.def_id());
                }
            }
            ty::TyAdt(def, _) => {
                self.assemble_inherent_impl_candidates_for_type(def.did);
            }
            ty::TyForeign(did) => {
                self.assemble_inherent_impl_candidates_for_type(did);
            }
            ty::TyParam(p) => {
                self.assemble_inherent_candidates_from_param(self_ty, p);
            }
            ty::TyChar => {
                let lang_def_id = lang_items.char_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyStr => {
                let lang_def_id = lang_items.str_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TySlice(_) => {
                let lang_def_id = lang_items.slice_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);

                let lang_def_id = lang_items.slice_u8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                let lang_def_id = lang_items.const_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                let lang_def_id = lang_items.mut_ptr_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I8) => {
                let lang_def_id = lang_items.i8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I16) => {
                let lang_def_id = lang_items.i16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I32) => {
                let lang_def_id = lang_items.i32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I64) => {
                let lang_def_id = lang_items.i64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::I128) => {
                let lang_def_id = lang_items.i128_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyInt(ast::IntTy::Is) => {
                let lang_def_id = lang_items.isize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U8) => {
                let lang_def_id = lang_items.u8_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U16) => {
                let lang_def_id = lang_items.u16_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U32) => {
                let lang_def_id = lang_items.u32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U64) => {
                let lang_def_id = lang_items.u64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::U128) => {
                let lang_def_id = lang_items.u128_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyUint(ast::UintTy::Us) => {
                let lang_def_id = lang_items.usize_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::FloatTy::F32) => {
                let lang_def_id = lang_items.f32_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            ty::TyFloat(ast::FloatTy::F64) => {
                let lang_def_id = lang_items.f64_impl();
                self.assemble_inherent_impl_for_primitive(lang_def_id);
            }
            _ => {}
        }
    }

    fn assemble_inherent_impl_for_primitive(&mut self, lang_def_id: Option<DefId>) {
        if let Some(impl_def_id) = lang_def_id {
            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_candidates_for_type(&mut self, def_id: DefId) {
        let impl_def_ids = self.tcx.at(self.span).inherent_impls(def_id);
        for &impl_def_id in impl_def_ids.iter() {
            self.assemble_inherent_impl_probe(impl_def_id);
        }
    }

    fn assemble_inherent_impl_probe(&mut self, impl_def_id: DefId) {
        if !self.impl_dups.insert(impl_def_id) {
            return; // already visited
        }

        debug!("assemble_inherent_impl_probe {:?}", impl_def_id);

        for item in self.impl_or_trait_item(impl_def_id) {
            if !self.has_applicable_self(&item) {
                // No receiver declared. Not a candidate.
                self.record_static_candidate(ImplSource(impl_def_id));
                continue
            }

            let (impl_ty, impl_substs) = self.impl_ty_and_substs(impl_def_id);
            let impl_ty = impl_ty.subst(self.tcx, impl_substs);

            // Determine the receiver type that the method itself expects.
            let xform_tys = self.xform_self_ty(&item, impl_ty, impl_substs);

            // We can't use normalize_associated_types_in as it will pollute the
            // fcx's fulfillment context after this probe is over.
            let cause = traits::ObligationCause::misc(self.span, self.body_id);
            let selcx = &mut traits::SelectionContext::new(self.fcx);
            let traits::Normalized { value: (xform_self_ty, xform_ret_ty), obligations } =
                traits::normalize(selcx, self.param_env, cause, &xform_tys);
            debug!("assemble_inherent_impl_probe: xform_self_ty = {:?}/{:?}",
                   xform_self_ty, xform_ret_ty);

            self.push_candidate(Candidate {
                xform_self_ty, xform_ret_ty, item,
                kind: InherentImplCandidate(impl_substs, obligations),
                import_id: None
            }, true);
        }
    }

    fn assemble_inherent_candidates_from_object(&mut self,
                                                self_ty: Ty<'tcx>,
                                                principal: ty::PolyExistentialTraitRef<'tcx>) {
        debug!("assemble_inherent_candidates_from_object(self_ty={:?})",
               self_ty);

        // It is illegal to invoke a method on a trait instance that
        // refers to the `Self` type. An error will be reported by
        // `enforce_object_limitations()` if the method refers to the
        // `Self` type anywhere other than the receiver. Here, we use
        // a substitution that replaces `Self` with the object type
        // itself. Hence, a `&self` method will wind up with an
        // argument type like `&Trait`.
        let trait_ref = principal.with_self_ty(self.tcx, self_ty);
        self.elaborate_bounds(&[trait_ref], |this, new_trait_ref, item| {
            let new_trait_ref = this.erase_late_bound_regions(&new_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(&item, new_trait_ref.self_ty(), new_trait_ref.substs);
            this.push_candidate(Candidate {
                xform_self_ty, xform_ret_ty, item,
                kind: ObjectCandidate,
                import_id: None
            }, true);
        });
    }

    fn assemble_inherent_candidates_from_param(&mut self,
                                               _rcvr_ty: Ty<'tcx>,
                                               param_ty: ty::ParamTy) {
        // FIXME -- Do we want to commit to this behavior for param bounds?

        let bounds: Vec<_> = self.param_env
            .caller_bounds
            .iter()
            .filter_map(|predicate| {
                match *predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        match trait_predicate.0.trait_ref.self_ty().sty {
                            ty::TyParam(ref p) if *p == param_ty => {
                                Some(trait_predicate.to_poly_trait_ref())
                            }
                            _ => None,
                        }
                    }
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Subtype(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::TypeOutlives(..) |
                    ty::Predicate::ConstEvaluatable(..) => None,
                }
            })
            .collect();

        self.elaborate_bounds(&bounds, |this, poly_trait_ref, item| {
            let trait_ref = this.erase_late_bound_regions(&poly_trait_ref);

            let (xform_self_ty, xform_ret_ty) =
                this.xform_self_ty(&item, trait_ref.self_ty(), trait_ref.substs);

            // Because this trait derives from a where-clause, it
            // should not contain any inference variables or other
            // artifacts. This means it is safe to put into the
            // `WhereClauseCandidate` and (eventually) into the
            // `WhereClausePick`.
            assert!(!trait_ref.substs.needs_infer());

            this.push_candidate(Candidate {
                xform_self_ty, xform_ret_ty, item,
                kind: WhereClauseCandidate(poly_trait_ref),
                import_id: None
            }, true);
        });
    }

    // Do a search through a list of bounds, using a callback to actually
    // create the candidates.
    fn elaborate_bounds<F>(&mut self, bounds: &[ty::PolyTraitRef<'tcx>], mut mk_cand: F)
        where F: for<'b> FnMut(&mut ProbeContext<'b, 'gcx, 'tcx>,
                               ty::PolyTraitRef<'tcx>,
                               ty::AssociatedItem)
    {
        debug!("elaborate_bounds(bounds={:?})", bounds);

        let tcx = self.tcx;
        for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
            for item in self.impl_or_trait_item(bound_trait_ref.def_id()) {
                if !self.has_applicable_self(&item) {
                    self.record_static_candidate(TraitSource(bound_trait_ref.def_id()));
                } else {
                    mk_cand(self, bound_trait_ref, item);
                }
            }
        }
    }

    fn assemble_extension_candidates_for_traits_in_scope(&mut self,
                                                         expr_id: ast::NodeId)
                                                         -> Result<(), MethodError<'tcx>> {
        if expr_id == ast::DUMMY_NODE_ID {
            return Ok(())
        }
        let mut duplicates = FxHashSet();
        let expr_hir_id = self.tcx.hir.node_to_hir_id(expr_id);
        let opt_applicable_traits = self.tcx.in_scope_traits(expr_hir_id);
        if let Some(applicable_traits) = opt_applicable_traits {
            for trait_candidate in applicable_traits.iter() {
                let trait_did = trait_candidate.def_id;
                if duplicates.insert(trait_did) {
                    let import_id = trait_candidate.import_id;
                    let result = self.assemble_extension_candidates_for_trait(import_id, trait_did);
                    result?;
                }
            }
        }
        Ok(())
    }

    fn assemble_extension_candidates_for_all_traits(&mut self) -> Result<(), MethodError<'tcx>> {
        let mut duplicates = FxHashSet();
        for trait_info in suggest::all_traits(self.tcx) {
            if duplicates.insert(trait_info.def_id) {
                self.assemble_extension_candidates_for_trait(None, trait_info.def_id)?;
            }
        }
        Ok(())
    }

    pub fn matches_return_type(&self,
                               method: &ty::AssociatedItem,
                               self_ty: Option<Ty<'tcx>>,
                               expected: Ty<'tcx>) -> bool {
        match method.def() {
            Def::Method(def_id) => {
                let fty = self.tcx.fn_sig(def_id);
                self.probe(|_| {
                    let substs = self.fresh_substs_for_item(self.span, method.def_id);
                    let fty = fty.subst(self.tcx, substs);
                    let (fty, _) = self.replace_late_bound_regions_with_fresh_var(
                        self.span, infer::FnCall, &fty);

                    if let Some(self_ty) = self_ty {
                        if let Err(_) = self.at(&ObligationCause::dummy(), self.param_env)
                            .sup(fty.inputs()[0], self_ty)
                        {
                            return false
                        }
                    }
                    self.can_sub(self.param_env, fty.output(), expected).is_ok()
                })
            }
            _ => false,
        }
    }

    fn assemble_extension_candidates_for_trait(&mut self,
                                               import_id: Option<ast::NodeId>,
                                               trait_def_id: DefId)
                                               -> Result<(), MethodError<'tcx>> {
        debug!("assemble_extension_candidates_for_trait(trait_def_id={:?})",
               trait_def_id);
        let trait_substs = self.fresh_item_substs(trait_def_id);
        let trait_ref = ty::TraitRef::new(trait_def_id, trait_substs);

        for item in self.impl_or_trait_item(trait_def_id) {
            // Check whether `trait_def_id` defines a method with suitable name:
            if !self.has_applicable_self(&item) {
                debug!("method has inapplicable self");
                self.record_static_candidate(TraitSource(trait_def_id));
                continue;
            }

            let (xform_self_ty, xform_ret_ty) =
                self.xform_self_ty(&item, trait_ref.self_ty(), trait_substs);
            self.push_candidate(Candidate {
                xform_self_ty, xform_ret_ty, item, import_id,
                kind: TraitCandidate(trait_ref),
            }, false);
        }
        Ok(())
    }

    fn candidate_method_names(&self) -> Vec<ast::Name> {
        let mut set = FxHashSet();
        let mut names: Vec<_> = self.inherent_candidates
            .iter()
            .chain(&self.extension_candidates)
            .filter(|candidate| {
                if let Some(return_ty) = self.return_type {
                    self.matches_return_type(&candidate.item, None, return_ty)
                } else {
                    true
                }
            })
            .map(|candidate| candidate.item.name)
            .filter(|&name| set.insert(name))
            .collect();

        // sort them by the name so we have a stable result
        names.sort_by_key(|n| n.as_str());
        names
    }

    ///////////////////////////////////////////////////////////////////////////
    // THE ACTUAL SEARCH

    fn pick(mut self) -> PickResult<'tcx> {
        assert!(self.method_name.is_some());

        if let Some(r) = self.pick_core() {
            return r;
        }

        let static_candidates = mem::replace(&mut self.static_candidates, vec![]);
        let private_candidate = mem::replace(&mut self.private_candidate, None);
        let unsatisfied_predicates = mem::replace(&mut self.unsatisfied_predicates, vec![]);

        // things failed, so lets look at all traits, for diagnostic purposes now:
        self.reset();

        let span = self.span;
        let tcx = self.tcx;

        self.assemble_extension_candidates_for_all_traits()?;

        let out_of_scope_traits = match self.pick_core() {
            Some(Ok(p)) => vec![p.item.container.id()],
            //Some(Ok(p)) => p.iter().map(|p| p.item.container().id()).collect(),
            Some(Err(MethodError::Ambiguity(v))) => {
                v.into_iter()
                    .map(|source| {
                        match source {
                            TraitSource(id) => id,
                            ImplSource(impl_id) => {
                                match tcx.trait_id_of_impl(impl_id) {
                                    Some(id) => id,
                                    None => {
                                        span_bug!(span,
                                                  "found inherent method when looking at traits")
                                    }
                                }
                            }
                        }
                    })
                    .collect()
            }
            Some(Err(MethodError::NoMatch(NoMatchData { out_of_scope_traits: others, .. }))) => {
                assert!(others.is_empty());
                vec![]
            }
            _ => vec![],
        };

        if let Some(def) = private_candidate {
            return Err(MethodError::PrivateMatch(def, out_of_scope_traits));
        }
        let lev_candidate = self.probe_for_lev_candidate()?;

        Err(MethodError::NoMatch(NoMatchData::new(static_candidates,
                                                  unsatisfied_predicates,
                                                  out_of_scope_traits,
                                                  lev_candidate,
                                                  self.mode)))
    }

    fn pick_core(&mut self) -> Option<PickResult<'tcx>> {
        let steps = self.steps.clone();

        // find the first step that works
        steps
            .iter()
            .filter(|step| {
                debug!("pick_core: step={:?}", step);
                // skip types that are from a type error or that would require dereferencing
                // a raw pointer
                !step.self_ty.references_error() && !step.from_unsafe_deref
            }).flat_map(|step| {
                self.pick_by_value_method(step).or_else(|| {
                self.pick_autorefd_method(step, hir::MutImmutable).or_else(|| {
                self.pick_autorefd_method(step, hir::MutMutable)
            })})})
            .next()
    }

    fn pick_by_value_method(&mut self, step: &CandidateStep<'tcx>) -> Option<PickResult<'tcx>> {
        //! For each type `T` in the step list, this attempts to find a
        //! method where the (transformed) self type is exactly `T`. We
        //! do however do one transformation on the adjustment: if we
        //! are passing a region pointer in, we will potentially
        //! *reborrow* it to a shorter lifetime. This allows us to
        //! transparently pass `&mut` pointers, in particular, without
        //! consuming them for their entire lifetime.

        if step.unsize {
            return None;
        }

        self.pick_method(step.self_ty).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;

                // Insert a `&*` or `&mut *` if this is a reference type:
                if let ty::TyRef(_, mt) = step.self_ty.sty {
                    pick.autoderefs += 1;
                    pick.autoref = Some(mt.mutbl);
                }

                pick
            })
        })
    }

    fn pick_autorefd_method(&mut self, step: &CandidateStep<'tcx>, mutbl: hir::Mutability)
                            -> Option<PickResult<'tcx>> {
        let tcx = self.tcx;

        // In general, during probing we erase regions. See
        // `impl_self_ty()` for an explanation.
        let region = tcx.types.re_erased;

        let autoref_ty = tcx.mk_ref(region,
                                    ty::TypeAndMut {
                                        ty: step.self_ty, mutbl
                                    });
        self.pick_method(autoref_ty).map(|r| {
            r.map(|mut pick| {
                pick.autoderefs = step.autoderefs;
                pick.autoref = Some(mutbl);
                pick.unsize = if step.unsize {
                    Some(step.self_ty)
                } else {
                    None
                };
                pick
            })
        })
    }

    fn pick_method(&mut self, self_ty: Ty<'tcx>) -> Option<PickResult<'tcx>> {
        debug!("pick_method(self_ty={})", self.ty_to_string(self_ty));

        let mut possibly_unsatisfied_predicates = Vec::new();

        debug!("searching inherent candidates");
        if let Some(pick) = self.consider_candidates(self_ty,
                                                     &self.inherent_candidates,
                                                     &mut possibly_unsatisfied_predicates) {
            return Some(pick);
        }

        debug!("searching extension candidates");
        let res = self.consider_candidates(self_ty,
                                           &self.extension_candidates,
                                           &mut possibly_unsatisfied_predicates);
        if let None = res {
            self.unsatisfied_predicates.extend(possibly_unsatisfied_predicates);
        }
        res
    }

    fn consider_candidates(&self,
                           self_ty: Ty<'tcx>,
                           probes: &[Candidate<'tcx>],
                           possibly_unsatisfied_predicates: &mut Vec<TraitRef<'tcx>>)
                           -> Option<PickResult<'tcx>> {
        let mut applicable_candidates: Vec<_> = probes.iter()
            .map(|probe| {
                (probe, self.consider_probe(self_ty, probe, possibly_unsatisfied_predicates))
            })
            .filter(|&(_, status)| status != ProbeResult::NoMatch)
            .collect();

        debug!("applicable_candidates: {:?}", applicable_candidates);

        if applicable_candidates.len() > 1 {
            if let Some(pick) = self.collapse_candidates_to_trait_pick(&applicable_candidates[..]) {
                return Some(Ok(pick));
            }
        }

        if applicable_candidates.len() > 1 {
            let sources = probes.iter()
                .map(|p| self.candidate_source(p, self_ty))
                .collect();
            return Some(Err(MethodError::Ambiguity(sources)));
        }

        applicable_candidates.pop().map(|(probe, status)| {
            if status == ProbeResult::Match {
                Ok(probe.to_unadjusted_pick())
            } else {
                Err(MethodError::BadReturnType)
            }
        })
    }

    fn select_trait_candidate(&self, trait_ref: ty::TraitRef<'tcx>)
                              -> traits::SelectionResult<'tcx, traits::Selection<'tcx>>
    {
        let cause = traits::ObligationCause::misc(self.span, self.body_id);
        let predicate =
            trait_ref.to_poly_trait_ref().to_poly_trait_predicate();
        let obligation = traits::Obligation::new(cause, self.param_env, predicate);
        traits::SelectionContext::new(self).select(&obligation)
    }

    fn candidate_source(&self, candidate: &Candidate<'tcx>, self_ty: Ty<'tcx>)
                        -> CandidateSource
    {
        match candidate.kind {
            InherentImplCandidate(..) => ImplSource(candidate.item.container.id()),
            ObjectCandidate |
            WhereClauseCandidate(_) => TraitSource(candidate.item.container.id()),
            TraitCandidate(trait_ref) => self.probe(|_| {
                let _ = self.at(&ObligationCause::dummy(), self.param_env)
                    .sup(candidate.xform_self_ty, self_ty);
                match self.select_trait_candidate(trait_ref) {
                    Ok(Some(traits::Vtable::VtableImpl(ref impl_data))) => {
                        // If only a single impl matches, make the error message point
                        // to that impl.
                        ImplSource(impl_data.impl_def_id)
                    }
                    _ => {
                        TraitSource(candidate.item.container.id())
                    }
                }
            })
        }
    }

    fn consider_probe(&self,
                      self_ty: Ty<'tcx>,
                      probe: &Candidate<'tcx>,
                      possibly_unsatisfied_predicates: &mut Vec<TraitRef<'tcx>>)
                      -> ProbeResult {
        debug!("consider_probe: self_ty={:?} probe={:?}", self_ty, probe);

        self.probe(|_| {
            // First check that the self type can be related.
            let sub_obligations = match self.at(&ObligationCause::dummy(), self.param_env)
                                            .sup(probe.xform_self_ty, self_ty) {
                Ok(InferOk { obligations, value: () }) => obligations,
                Err(_) => {
                    debug!("--> cannot relate self-types");
                    return ProbeResult::NoMatch;
                }
            };

            let mut result = ProbeResult::Match;
            let selcx = &mut traits::SelectionContext::new(self);
            let cause = traits::ObligationCause::misc(self.span, self.body_id);

            // If so, impls may carry other conditions (e.g., where
            // clauses) that must be considered. Make sure that those
            // match as well (or at least may match, sometimes we
            // don't have enough information to fully evaluate).
            let candidate_obligations : Vec<_> = match probe.kind {
                InherentImplCandidate(ref substs, ref ref_obligations) => {
                    // Check whether the impl imposes obligations we have to worry about.
                    let impl_def_id = probe.item.container.id();
                    let impl_bounds = self.tcx.predicates_of(impl_def_id);
                    let impl_bounds = impl_bounds.instantiate(self.tcx, substs);
                    let traits::Normalized { value: impl_bounds, obligations: norm_obligations } =
                        traits::normalize(selcx, self.param_env, cause.clone(), &impl_bounds);

                    // Convert the bounds into obligations.
                    let impl_obligations = traits::predicates_for_generics(
                        cause.clone(), self.param_env, &impl_bounds);

                    debug!("impl_obligations={:?}", impl_obligations);
                    impl_obligations.into_iter()
                        .chain(norm_obligations.into_iter())
                        .chain(ref_obligations.iter().cloned())
                        .collect()
                }

                ObjectCandidate |
                WhereClauseCandidate(..) => {
                    // These have no additional conditions to check.
                    vec![]
                }

                TraitCandidate(trait_ref) => {
                    let predicate = trait_ref.to_predicate();
                    let obligation =
                        traits::Obligation::new(cause.clone(), self.param_env, predicate);
                    if !selcx.evaluate_obligation(&obligation) {
                        if self.probe(|_| self.select_trait_candidate(trait_ref).is_err()) {
                            // This candidate's primary obligation doesn't even
                            // select - don't bother registering anything in
                            // `potentially_unsatisfied_predicates`.
                            return ProbeResult::NoMatch;
                        } else {
                            // Some nested subobligation of this predicate
                            // failed.
                            //
                            // FIXME: try to find the exact nested subobligation
                            // and point at it rather than reporting the entire
                            // trait-ref?
                            result = ProbeResult::NoMatch;
                            let trait_ref = self.resolve_type_vars_if_possible(&trait_ref);
                            possibly_unsatisfied_predicates.push(trait_ref);
                        }
                    }
                    vec![]
                }
            };

            debug!("consider_probe - candidate_obligations={:?} sub_obligations={:?}",
                   candidate_obligations, sub_obligations);

            // Evaluate those obligations to see if they might possibly hold.
            for o in candidate_obligations.into_iter().chain(sub_obligations) {
                let o = self.resolve_type_vars_if_possible(&o);
                if !selcx.evaluate_obligation(&o) {
                    result = ProbeResult::NoMatch;
                    if let &ty::Predicate::Trait(ref pred) = &o.predicate {
                        possibly_unsatisfied_predicates.push(pred.0.trait_ref);
                    }
                }
            }

            if let ProbeResult::Match = result {
                if let (Some(return_ty), Some(xform_ret_ty)) =
                    (self.return_type, probe.xform_ret_ty)
                {
                    let xform_ret_ty = self.resolve_type_vars_if_possible(&xform_ret_ty);
                    debug!("comparing return_ty {:?} with xform ret ty {:?}",
                           return_ty,
                           probe.xform_ret_ty);
                    if self.at(&ObligationCause::dummy(), self.param_env)
                        .sup(return_ty, xform_ret_ty)
                        .is_err()
                    {
                        return ProbeResult::BadReturnType;
                    }
                }
            }

            result
        })
    }

    /// Sometimes we get in a situation where we have multiple probes that are all impls of the
    /// same trait, but we don't know which impl to use. In this case, since in all cases the
    /// external interface of the method can be determined from the trait, it's ok not to decide.
    /// We can basically just collapse all of the probes for various impls into one where-clause
    /// probe. This will result in a pending obligation so when more type-info is available we can
    /// make the final decision.
    ///
    /// Example (`src/test/run-pass/method-two-trait-defer-resolution-1.rs`):
    ///
    /// ```
    /// trait Foo { ... }
    /// impl Foo for Vec<int> { ... }
    /// impl Foo for Vec<usize> { ... }
    /// ```
    ///
    /// Now imagine the receiver is `Vec<_>`. It doesn't really matter at this time which impl we
    /// use, so it's ok to just commit to "using the method from the trait Foo".
    fn collapse_candidates_to_trait_pick(&self, probes: &[(&Candidate<'tcx>, ProbeResult)])
                                         -> Option<Pick<'tcx>>
    {
        // Do all probes correspond to the same trait?
        let container = probes[0].0.item.container;
        match container {
            ty::TraitContainer(_) => {}
            ty::ImplContainer(_) => return None,
        }
        if probes[1..].iter().any(|&(p, _)| p.item.container != container) {
            return None;
        }

        // FIXME: check the return type here somehow.
        // If so, just use this trait and call it a day.
        Some(Pick {
            item: probes[0].0.item.clone(),
            kind: TraitPick,
            import_id: probes[0].0.import_id,
            autoderefs: 0,
            autoref: None,
            unsize: None,
        })
    }

    /// Similarly to `probe_for_return_type`, this method attempts to find the best matching
    /// candidate method where the method name may have been misspelt. Similarly to other
    /// Levenshtein based suggestions, we provide at most one such suggestion.
    fn probe_for_lev_candidate(&mut self) -> Result<Option<ty::AssociatedItem>, MethodError<'tcx>> {
        debug!("Probing for method names similar to {:?}",
               self.method_name);

        let steps = self.steps.clone();
        self.probe(|_| {
            let mut pcx = ProbeContext::new(self.fcx, self.span, self.mode, self.method_name,
                                            self.return_type, steps);
            pcx.allow_similar_names = true;
            pcx.assemble_inherent_candidates();
            pcx.assemble_extension_candidates_for_traits_in_scope(ast::DUMMY_NODE_ID)?;

            let method_names = pcx.candidate_method_names();
            pcx.allow_similar_names = false;
            let applicable_close_candidates: Vec<ty::AssociatedItem> = method_names
                .iter()
                .filter_map(|&method_name| {
                    pcx.reset();
                    pcx.method_name = Some(method_name);
                    pcx.assemble_inherent_candidates();
                    pcx.assemble_extension_candidates_for_traits_in_scope(ast::DUMMY_NODE_ID)
                        .ok().map_or(None, |_| {
                            pcx.pick_core()
                                .and_then(|pick| pick.ok())
                                .and_then(|pick| Some(pick.item))
                        })
                })
               .collect();

            if applicable_close_candidates.is_empty() {
                Ok(None)
            } else {
                let best_name = {
                    let names = applicable_close_candidates.iter().map(|cand| &cand.name);
                    find_best_match_for_name(names,
                                             &self.method_name.unwrap().as_str(),
                                             None)
                }.unwrap();
                Ok(applicable_close_candidates
                   .into_iter()
                   .find(|method| method.name == best_name))
            }
        })
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY
    fn has_applicable_self(&self, item: &ty::AssociatedItem) -> bool {
        // "Fast track" -- check for usage of sugar when in method call
        // mode.
        //
        // In Path mode (i.e., resolving a value like `T::next`), consider any
        // associated value (i.e., methods, constants) but not types.
        match self.mode {
            Mode::MethodCall => item.method_has_self_argument,
            Mode::Path => match item.kind {
                ty::AssociatedKind::Type => false,
                ty::AssociatedKind::Method | ty::AssociatedKind::Const => true
            },
        }
        // FIXME -- check for types that deref to `Self`,
        // like `Rc<Self>` and so on.
        //
        // Note also that the current code will break if this type
        // includes any of the type parameters defined on the method
        // -- but this could be overcome.
    }

    fn record_static_candidate(&mut self, source: CandidateSource) {
        self.static_candidates.push(source);
    }

    fn xform_self_ty(&self,
                     item: &ty::AssociatedItem,
                     impl_ty: Ty<'tcx>,
                     substs: &Substs<'tcx>)
                     -> (Ty<'tcx>, Option<Ty<'tcx>>) {
        if item.kind == ty::AssociatedKind::Method && self.mode == Mode::MethodCall {
            let sig = self.xform_method_sig(item.def_id, substs);
            (sig.inputs()[0], Some(sig.output()))
        } else {
            (impl_ty, None)
        }
    }

    fn xform_method_sig(&self,
                        method: DefId,
                        substs: &Substs<'tcx>)
                        -> ty::FnSig<'tcx>
    {
        let fn_sig = self.tcx.fn_sig(method);
        debug!("xform_self_ty(fn_sig={:?}, substs={:?})",
               fn_sig,
               substs);

        assert!(!substs.has_escaping_regions());

        // It is possible for type parameters or early-bound lifetimes
        // to appear in the signature of `self`. The substitutions we
        // are given do not include type/lifetime parameters for the
        // method yet. So create fresh variables here for those too,
        // if there are any.
        let generics = self.tcx.generics_of(method);
        assert_eq!(substs.types().count(), generics.parent_types as usize);
        assert_eq!(substs.regions().count(), generics.parent_regions as usize);

        // Erase any late-bound regions from the method and substitute
        // in the values from the substitution.
        let xform_fn_sig = self.erase_late_bound_regions(&fn_sig);

        if generics.types.is_empty() && generics.regions.is_empty() {
            xform_fn_sig.subst(self.tcx, substs)
        } else {
            let substs = Substs::for_item(self.tcx, method, |def, _| {
                let i = def.index as usize;
                if i < substs.len() {
                    substs.region_at(i)
                } else {
                    // In general, during probe we erase regions. See
                    // `impl_self_ty()` for an explanation.
                    self.tcx.types.re_erased
                }
            }, |def, cur_substs| {
                let i = def.index as usize;
                if i < substs.len() {
                    substs.type_at(i)
                } else {
                    self.type_var_for_def(self.span, def, cur_substs)
                }
            });
            xform_fn_sig.subst(self.tcx, substs)
        }
    }

    /// Get the type of an impl and generate substitutions with placeholders.
    fn impl_ty_and_substs(&self, impl_def_id: DefId) -> (Ty<'tcx>, &'tcx Substs<'tcx>) {
        (self.tcx.type_of(impl_def_id), self.fresh_item_substs(impl_def_id))
    }

    fn fresh_item_substs(&self, def_id: DefId) -> &'tcx Substs<'tcx> {
        Substs::for_item(self.tcx,
                         def_id,
                         |_, _| self.tcx.types.re_erased,
                         |_, _| self.next_ty_var(
                             TypeVariableOrigin::SubstitutionPlaceholder(
                                 self.tcx.def_span(def_id))))
    }

    /// Replace late-bound-regions bound by `value` with `'static` using
    /// `ty::erase_late_bound_regions`.
    ///
    /// This is only a reasonable thing to do during the *probe* phase, not the *confirm* phase, of
    /// method matching. It is reasonable during the probe phase because we don't consider region
    /// relationships at all. Therefore, we can just replace all the region variables with 'static
    /// rather than creating fresh region variables. This is nice for two reasons:
    ///
    /// 1. Because the numbers of the region variables would otherwise be fairly unique to this
    ///    particular method call, it winds up creating fewer types overall, which helps for memory
    ///    usage. (Admittedly, this is a rather small effect, though measureable.)
    ///
    /// 2. It makes it easier to deal with higher-ranked trait bounds, because we can replace any
    ///    late-bound regions with 'static. Otherwise, if we were going to replace late-bound
    ///    regions with actual region variables as is proper, we'd have to ensure that the same
    ///    region got replaced with the same variable, which requires a bit more coordination
    ///    and/or tracking the substitution and
    ///    so forth.
    fn erase_late_bound_regions<T>(&self, value: &ty::Binder<T>) -> T
        where T: TypeFoldable<'tcx>
    {
        self.tcx.erase_late_bound_regions(value)
    }

    /// Find the method with the appropriate name (or return type, as the case may be). If
    /// `allow_similar_names` is set, find methods with close-matching names.
    fn impl_or_trait_item(&self, def_id: DefId) -> Vec<ty::AssociatedItem> {
        if let Some(name) = self.method_name {
            if self.allow_similar_names {
                let max_dist = max(name.as_str().len(), 3) / 3;
                self.tcx.associated_items(def_id)
                    .filter(|x| {
                        let dist = lev_distance(&*name.as_str(), &x.name.as_str());
                        Namespace::from(x.kind) == Namespace::Value && dist > 0
                        && dist <= max_dist
                    })
                    .collect()
            } else {
                self.fcx
                    .associated_item(def_id, name, Namespace::Value)
                    .map_or(Vec::new(), |x| vec![x])
            }
        } else {
            self.tcx.associated_items(def_id).collect()
        }
    }
}

impl<'tcx> Candidate<'tcx> {
    fn to_unadjusted_pick(&self) -> Pick<'tcx> {
        Pick {
            item: self.item.clone(),
            kind: match self.kind {
                InherentImplCandidate(..) => InherentImplPick,
                ObjectCandidate => ObjectPick,
                TraitCandidate(_) => TraitPick,
                WhereClauseCandidate(ref trait_ref) => {
                    // Only trait derived from where-clauses should
                    // appear here, so they should not contain any
                    // inference variables or other artifacts. This
                    // means they are safe to put into the
                    // `WhereClausePick`.
                    assert!(!trait_ref.substs().needs_infer());

                    WhereClausePick(trait_ref.clone())
                }
            },
            import_id: self.import_id,
            autoderefs: 0,
            autoref: None,
            unsize: None,
        }
    }
}
