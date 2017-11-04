use middle::free_region::FreeRegionMap;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::infer::{InferCtxt, GenericKind};
use rustc::traits::FulfillmentContext;
use rustc::ty::outlives::Component;
use rustc::ty::wf;

use syntax::ast;
use syntax_pos::Span;

#[derive(Clone)]
pub struct OutlivesEnvironment<'tcx> {
    param_env: ty::ParamEnv<'tcx>,
    free_region_map: FreeRegionMap<'tcx>,
    region_bound_pairs: Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>,
}

/// Implied bounds are region relationships that we deduce
/// automatically.  The idea is that (e.g.) a caller must check that a
/// function's argument types are well-formed immediately before
/// calling that fn, and hence the *callee* can assume that its
/// argument types are well-formed. This may imply certain relationships
/// between generic parameters. For example:
///
///     fn foo<'a,T>(x: &'a T)
///
/// can only be called with a `'a` and `T` such that `&'a T` is WF.
/// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
#[derive(Debug)]
enum ImpliedBound<'tcx> {
    RegionSubRegion(ty::Region<'tcx>, ty::Region<'tcx>),
    RegionSubParam(ty::Region<'tcx>, ty::ParamTy),
    RegionSubProjection(ty::Region<'tcx>, ty::ProjectionTy<'tcx>),
}

impl<'a, 'gcx: 'tcx, 'tcx: 'a> OutlivesEnvironment<'tcx> {
    pub fn new(param_env: ty::ParamEnv<'tcx>) -> Self {
        let mut free_region_map = FreeRegionMap::new();
        free_region_map.relate_free_regions_from_predicates(&param_env.caller_bounds);

        OutlivesEnvironment {
            param_env,
            free_region_map,
            region_bound_pairs: vec![],
        }
    }

    /// Borrows current value of the `free_region_map`.
    pub fn free_region_map(&self) -> &FreeRegionMap<'tcx> {
        &self.free_region_map
    }

    /// Borrows current value of the `region_bound_pairs`.
    pub fn region_bound_pairs(&self) -> &[(ty::Region<'tcx>, GenericKind<'tcx>)] {
        &self.region_bound_pairs
    }

    /// Returns ownership of the `free_region_map`.
    pub fn into_free_region_map(self) -> FreeRegionMap<'tcx> {
        self.free_region_map
    }

    /// This method adds "implied bounds" into the outlives environment.
    /// Implied bounds are outlives relationships that we can deduce
    /// on the basis that certain types must be well-formed -- these are
    /// either the types that appear in the function signature or else
    /// the input types to an impl. For example, if you have a function
    /// like
    ///
    /// ```
    /// fn foo<'a, 'b, T>(x: &'a &'b [T]) { }
    /// ```
    ///
    /// we can assume in the caller's body that `'b: 'a` and that `T:
    /// 'b` (and hence, transitively, that `T: 'a`). This method would
    /// add those assumptions into the outlives-environment.
    ///
    /// Tests: `src/test/compile-fail/regions-free-region-ordering-*.rs`
    pub fn add_implied_bounds(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        fn_sig_tys: &[Ty<'tcx>],
        body_id: ast::NodeId,
        span: Span,
    ) {
        debug!("add_implied_bounds()");

        for &ty in fn_sig_tys {
            let ty = infcx.resolve_type_vars_if_possible(&ty);
            debug!("add_implied_bounds: ty = {}", ty);
            let implied_bounds = self.implied_bounds(infcx, body_id, ty, span);

            // But also record other relationships, such as `T:'x`,
            // that don't go into the free-region-map but which we use
            // here.
            for implication in implied_bounds {
                debug!("add_implied_bounds: implication={:?}", implication);
                match implication {
                    ImpliedBound::RegionSubRegion(
                        r_a @ &ty::ReEarlyBound(_),
                        &ty::ReVar(vid_b),
                    ) |
                    ImpliedBound::RegionSubRegion(r_a @ &ty::ReFree(_), &ty::ReVar(vid_b)) => {
                        infcx.add_given(r_a, vid_b);
                    }
                    ImpliedBound::RegionSubParam(r_a, param_b) => {
                        self.region_bound_pairs
                            .push((r_a, GenericKind::Param(param_b)));
                    }
                    ImpliedBound::RegionSubProjection(r_a, projection_b) => {
                        self.region_bound_pairs
                            .push((r_a, GenericKind::Projection(projection_b)));
                    }
                    ImpliedBound::RegionSubRegion(r_a, r_b) => {
                        // In principle, we could record (and take
                        // advantage of) every relationship here, but
                        // we are also free not to -- it simply means
                        // strictly less that we can successfully type
                        // check. Right now we only look for things
                        // relationships between free regions. (It may
                        // also be that we should revise our inference
                        // system to be more general and to make use
                        // of *every* relationship that arises here,
                        // but presently we do not.)
                        self.free_region_map.relate_regions(r_a, r_b);
                    }
                }
            }
        }
    }

    /// Compute the implied bounds that a callee/impl can assume based on
    /// the fact that caller/projector has ensured that `ty` is WF.  See
    /// the `ImpliedBound` type for more details.
    fn implied_bounds(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        body_id: ast::NodeId,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Vec<ImpliedBound<'tcx>> {
        let tcx = infcx.tcx;

        // Sometimes when we ask what it takes for T: WF, we get back that
        // U: WF is required; in that case, we push U onto this stack and
        // process it next. Currently (at least) these resulting
        // predicates are always guaranteed to be a subset of the original
        // type, so we need not fear non-termination.
        let mut wf_types = vec![ty];

        let mut implied_bounds = vec![];

        let mut fulfill_cx = FulfillmentContext::new();

        while let Some(ty) = wf_types.pop() {
            // Compute the obligations for `ty` to be well-formed. If `ty` is
            // an unresolved inference variable, just substituted an empty set
            // -- because the return type here is going to be things we *add*
            // to the environment, it's always ok for this set to be smaller
            // than the ultimate set. (Note: normally there won't be
            // unresolved inference variables here anyway, but there might be
            // during typeck under some circumstances.)
            let obligations =
                wf::obligations(infcx, self.param_env, body_id, ty, span).unwrap_or(vec![]);

            // NB: All of these predicates *ought* to be easily proven
            // true. In fact, their correctness is (mostly) implied by
            // other parts of the program. However, in #42552, we had
            // an annoying scenario where:
            //
            // - Some `T::Foo` gets normalized, resulting in a
            //   variable `_1` and a `T: Trait<Foo=_1>` constraint
            //   (not sure why it couldn't immediately get
            //   solved). This result of `_1` got cached.
            // - These obligations were dropped on the floor here,
            //   rather than being registered.
            // - Then later we would get a request to normalize
            //   `T::Foo` which would result in `_1` being used from
            //   the cache, but hence without the `T: Trait<Foo=_1>`
            //   constraint. As a result, `_1` never gets resolved,
            //   and we get an ICE (in dropck).
            //
            // Therefore, we register any predicates involving
            // inference variables. We restrict ourselves to those
            // involving inference variables both for efficiency and
            // to avoids duplicate errors that otherwise show up.
            fulfill_cx.register_predicate_obligations(
                infcx,
                obligations
                    .iter()
                    .filter(|o| o.predicate.has_infer_types())
                    .cloned());

            // From the full set of obligations, just filter down to the
            // region relationships.
            implied_bounds.extend(obligations.into_iter().flat_map(|obligation| {
                assert!(!obligation.has_escaping_regions());
                match obligation.predicate {
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Subtype(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ConstEvaluatable(..) => vec![],

                    ty::Predicate::WellFormed(subty) => {
                        wf_types.push(subty);
                        vec![]
                    }

                    ty::Predicate::RegionOutlives(ref data) => {
                        match tcx.no_late_bound_regions(data) {
                            None => vec![],
                            Some(ty::OutlivesPredicate(r_a, r_b)) => {
                                vec![ImpliedBound::RegionSubRegion(r_b, r_a)]
                            }
                        }
                    }

                    ty::Predicate::TypeOutlives(ref data) => {
                        match tcx.no_late_bound_regions(data) {
                            None => vec![],
                            Some(ty::OutlivesPredicate(ty_a, r_b)) => {
                                let ty_a = infcx.resolve_type_vars_if_possible(&ty_a);
                                let components = tcx.outlives_components(ty_a);
                                self.implied_bounds_from_components(r_b, components)
                            }
                        }
                    }
                }
            }));
        }

        // Ensure that those obligations that we had to solve
        // get solved *here*.
        match fulfill_cx.select_all_or_error(infcx) {
            Ok(()) => (),
            Err(errors) => infcx.report_fulfillment_errors(&errors, None),
        }

        implied_bounds
    }

    /// When we have an implied bound that `T: 'a`, we can further break
    /// this down to determine what relationships would have to hold for
    /// `T: 'a` to hold. We get to assume that the caller has validated
    /// those relationships.
    fn implied_bounds_from_components(
        &self,
        sub_region: ty::Region<'tcx>,
        sup_components: Vec<Component<'tcx>>,
    ) -> Vec<ImpliedBound<'tcx>> {
        sup_components
            .into_iter()
            .flat_map(|component| {
                match component {
                    Component::Region(r) =>
                        vec![ImpliedBound::RegionSubRegion(sub_region, r)],
                    Component::Param(p) =>
                        vec![ImpliedBound::RegionSubParam(sub_region, p)],
                    Component::Projection(p) =>
                        vec![ImpliedBound::RegionSubProjection(sub_region, p)],
                    Component::EscapingProjection(_) =>
                    // If the projection has escaping regions, don't
                    // try to infer any implied bounds even for its
                    // free components. This is conservative, because
                    // the caller will still have to prove that those
                    // free components outlive `sub_region`. But the
                    // idea is that the WAY that the caller proves
                    // that may change in the future and we want to
                    // give ourselves room to get smarter here.
                        vec![],
                    Component::UnresolvedInferenceVariable(..) =>
                        vec![],
                }
            })
            .collect()
    }
}
