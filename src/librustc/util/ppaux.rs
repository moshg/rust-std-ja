// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use ast_map;
use middle::def;
use middle::region;
use middle::subst::{VecPerParamSpace,Subst};
use middle::subst;
use middle::ty::{BoundRegion, BrAnon, BrNamed};
use middle::ty::{ReEarlyBound, BrFresh, ctxt};
use middle::ty::{ReFree, ReScope, ReInfer, ReStatic, Region, ReEmpty};
use middle::ty::{ReSkolemized, ReVar, BrEnv};
use middle::ty::{mt, Ty, ParamTy};
use middle::ty::{TyBool, TyChar, TyStruct, TyEnum};
use middle::ty::{TyError, TyStr, TyArray, TySlice, TyFloat, TyBareFn};
use middle::ty::{TyParam, TyRawPtr, TyRef, TyTuple};
use middle::ty::TyClosure;
use middle::ty::{TyBox, TyTrait, TyInt, TyUint, TyInfer};
use middle::ty;
use middle::ty_fold::{self, TypeFoldable};

use std::collections::HashMap;
use std::collections::hash_state::HashState;
use std::hash::Hash;
use std::rc::Rc;
use syntax::abi;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::{ast, ast_util};
use syntax::owned_slice::OwnedSlice;

/// Produces a string suitable for debugging output.
pub trait Repr<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String;
}

/// Produces a string suitable for showing to the user.
pub trait UserString<'tcx> : Repr<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String;
}

fn parameterized<'tcx, GG>(cx: &ctxt<'tcx>,
                           substs: &subst::Substs<'tcx>,
                           did: ast::DefId,
                           projections: &[ty::ProjectionPredicate<'tcx>],
                           get_generics: GG)
                           -> String
    where GG : FnOnce() -> ty::Generics<'tcx>
{
    let base = ty::item_path_str(cx, did);
    if cx.sess.verbose() {
        let mut strings = vec![];
        match substs.regions {
            subst::ErasedRegions => {
                strings.push(format!(".."));
            }
            subst::NonerasedRegions(ref regions) => {
                for region in regions {
                    strings.push(region.repr(cx));
                }
            }
        }
        for ty in &substs.types {
            strings.push(ty.repr(cx));
        }
        for projection in projections {
            strings.push(format!("{}={}",
                                 projection.projection_ty.item_name.user_string(cx),
                                 projection.ty.user_string(cx)));
        }
        return if strings.is_empty() {
            format!("{}", base)
        } else {
            format!("{}<{}>", base, strings.connect(","))
        };
    }

    let mut strs = Vec::new();

    match substs.regions {
        subst::ErasedRegions => { }
        subst::NonerasedRegions(ref regions) => {
            for &r in regions {
                let s = r.user_string(cx);
                if s.is_empty() {
                    // This happens when the value of the region
                    // parameter is not easily serialized. This may be
                    // because the user omitted it in the first place,
                    // or because it refers to some block in the code,
                    // etc. I'm not sure how best to serialize this.
                    strs.push(format!("'_"));
                } else {
                    strs.push(s)
                }
            }
        }
    }

    // It is important to execute this conditionally, only if -Z
    // verbose is false. Otherwise, debug logs can sometimes cause
    // ICEs trying to fetch the generics early in the pipeline. This
    // is kind of a hacky workaround in that -Z verbose is required to
    // avoid those ICEs.
    let generics = get_generics();

    let has_self = substs.self_ty().is_some();
    let tps = substs.types.get_slice(subst::TypeSpace);
    let ty_params = generics.types.get_slice(subst::TypeSpace);
    let has_defaults = ty_params.last().map_or(false, |def| def.default.is_some());
    let num_defaults = if has_defaults {
        ty_params.iter().zip(tps).rev().take_while(|&(def, &actual)| {
            match def.default {
                Some(default) => {
                    if !has_self && ty::type_has_self(default) {
                        // In an object type, there is no `Self`, and
                        // thus if the default value references Self,
                        // the user will be required to give an
                        // explicit value. We can't even do the
                        // substitution below to check without causing
                        // an ICE. (#18956).
                        false
                    } else {
                        default.subst(cx, substs) == actual
                    }
                }
                None => false
            }
        }).count()
    } else {
        0
    };

    for t in &tps[..tps.len() - num_defaults] {
        strs.push(t.user_string(cx))
    }

    for projection in projections {
        strs.push(format!("{}={}",
                          projection.projection_ty.item_name.user_string(cx),
                          projection.ty.user_string(cx)));
    }

    if cx.lang_items.fn_trait_kind(did).is_some() && projections.len() == 1 {
        let projection_ty = projections[0].ty;
        let tail =
            if ty::type_is_nil(projection_ty) {
                format!("")
            } else {
                format!(" -> {}", projection_ty.user_string(cx))
            };
        format!("{}({}){}",
                base,
                if strs[0].starts_with("(") && strs[0].ends_with(",)") {
                    &strs[0][1 .. strs[0].len() - 2] // Remove '(' and ',)'
                } else if strs[0].starts_with("(") && strs[0].ends_with(")") {
                    &strs[0][1 .. strs[0].len() - 1] // Remove '(' and ')'
                } else {
                    &strs[0][..]
                },
                tail)
    } else if !strs.is_empty() {
        format!("{}<{}>", base, strs.connect(", "))
    } else {
        format!("{}", base)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Option<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &None => "None".to_string(),
            &Some(ref t) => t.repr(tcx),
        }
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for P<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (**self).repr(tcx)
    }
}

impl<'tcx,T:Repr<'tcx>,U:Repr<'tcx>> Repr<'tcx> for Result<T,U> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &Ok(ref t) => t.repr(tcx),
            &Err(ref u) => format!("Err({})", u.repr(tcx))
        }
    }
}

impl<'tcx> Repr<'tcx> for () {
    fn repr(&self, _tcx: &ctxt) -> String {
        "()".to_string()
    }
}

impl<'a, 'tcx, T: ?Sized +Repr<'tcx>> Repr<'tcx> for &'a T {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        Repr::repr(*self, tcx)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Rc<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (&**self).repr(tcx)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Box<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        (&**self).repr(tcx)
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for [T] {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("[{}]", self.iter().map(|t| t.repr(tcx)).collect::<Vec<_>>().connect(", "))
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for OwnedSlice<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        self[..].repr(tcx)
    }
}

// This is necessary to handle types like Option<Vec<T>>, for which
// autoderef cannot convert the &[T] handler
impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for Vec<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        self[..].repr(tcx)
    }
}

impl<'a, 'tcx, T: ?Sized +UserString<'tcx>> UserString<'tcx> for &'a T {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        UserString::user_string(*self, tcx)
    }
}

impl<'tcx, T:UserString<'tcx>> UserString<'tcx> for Vec<T> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let strs: Vec<String> =
            self.iter().map(|t| t.user_string(tcx)).collect();
        strs.connect(", ")
    }
}

impl<'tcx> Repr<'tcx> for def::Def {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

/// This curious type is here to help pretty-print trait objects. In
/// a trait object, the projections are stored separately from the
/// main trait bound, but in fact we want to package them together
/// when printing out; they also have separate binders, but we want
/// them to share a binder when we print them out. (And the binder
/// pretty-printing logic is kind of clever and we don't want to
/// reproduce it.) So we just repackage up the structure somewhat.
///
/// Right now there is only one trait in an object that can have
/// projection bounds, so we just stuff them altogether. But in
/// reality we should eventually sort things out better.
type TraitAndProjections<'tcx> =
    (ty::TraitRef<'tcx>, Vec<ty::ProjectionPredicate<'tcx>>);

impl<'tcx> UserString<'tcx> for TraitAndProjections<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let &(ref trait_ref, ref projection_bounds) = self;
        parameterized(tcx,
                      trait_ref.substs,
                      trait_ref.def_id,
                      &projection_bounds[..],
                      || ty::lookup_trait_def(tcx, trait_ref.def_id).generics.clone())
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitTy<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let &ty::TraitTy { ref principal, ref bounds } = self;

        let mut components = vec![];

        let tap: ty::Binder<TraitAndProjections<'tcx>> =
            ty::Binder((principal.0.clone(),
                        bounds.projection_bounds.iter().map(|x| x.0.clone()).collect()));

        // Generate the main trait ref, including associated types.
        components.push(tap.user_string(tcx));

        // Builtin bounds.
        for bound in &bounds.builtin_bounds {
            components.push(bound.user_string(tcx));
        }

        // Region, if not obviously implied by builtin bounds.
        if bounds.region_bound != ty::ReStatic {
            // Region bound is implied by builtin bounds:
            components.push(bounds.region_bound.user_string(tcx));
        }

        components.retain(|s| !s.is_empty());

        components.connect(" + ")
    }
}

impl<'tcx> Repr<'tcx> for ty::TypeParameterDef<'tcx> {
    fn repr(&self, _tcx: &ctxt<'tcx>) -> String {
        format!("TypeParameterDef({:?}, {:?}/{})",
                self.def_id,
                self.space,
                self.index)
    }
}

impl<'tcx> Repr<'tcx> for ty::RegionParameterDef {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("RegionParameterDef(name={}, def_id={}, bounds={})",
                token::get_name(self.name),
                self.def_id.repr(tcx),
                self.bounds.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::TyS<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        self.user_string(tcx)
    }
}

impl<'tcx> Repr<'tcx> for ty::mt<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{}{}",
            if self.mutbl == ast::MutMutable { "mut " } else { "" },
            self.ty.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for subst::Substs<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Substs[types={}, regions={}]",
                       self.types.repr(tcx),
                       self.regions.repr(tcx))
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for subst::VecPerParamSpace<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("[{};{};{}]",
                self.get_slice(subst::TypeSpace).repr(tcx),
                self.get_slice(subst::SelfSpace).repr(tcx),
                self.get_slice(subst::FnSpace).repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ItemSubsts<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ItemSubsts({})", self.substs.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for subst::RegionSubsts {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            subst::ErasedRegions => "erased".to_string(),
            subst::NonerasedRegions(ref regions) => regions.repr(tcx)
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::BuiltinBounds {
    fn repr(&self, _tcx: &ctxt) -> String {
        let mut res = Vec::new();
        for b in self {
            res.push(match b {
                ty::BoundSend => "Send".to_string(),
                ty::BoundSized => "Sized".to_string(),
                ty::BoundCopy => "Copy".to_string(),
                ty::BoundSync => "Sync".to_string(),
            });
        }
        res.connect("+")
    }
}

impl<'tcx> Repr<'tcx> for ty::ParamBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let mut res = Vec::new();
        res.push(self.builtin_bounds.repr(tcx));
        for t in &self.trait_bounds {
            res.push(t.repr(tcx));
        }
        res.connect("+")
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitRef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        // when printing out the debug representation, we don't need
        // to enumerate the `for<...>` etc because the debruijn index
        // tells you everything you need to know.
        let result = self.user_string(tcx);
        match self.substs.self_ty() {
            None => result,
            Some(sty) => format!("<{} as {}>", sty.repr(tcx), result)
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitDef<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TraitDef(generics={}, trait_ref={})",
                self.generics.repr(tcx),
                self.trait_ref.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ast::TraitItem {
    fn repr(&self, _tcx: &ctxt) -> String {
        let kind = match self.node {
            ast::ConstTraitItem(..) => "ConstTraitItem",
            ast::MethodTraitItem(..) => "MethodTraitItem",
            ast::TypeTraitItem(..) => "TypeTraitItem",
        };
        format!("{}({}, id={})", kind, self.ident, self.id)
    }
}

impl<'tcx> Repr<'tcx> for ast::Expr {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("expr({}: {})", self.id, pprust::expr_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Path {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("path({})", pprust::path_to_string(self))
    }
}

impl<'tcx> UserString<'tcx> for ast::Path {
    fn user_string(&self, _tcx: &ctxt) -> String {
        pprust::path_to_string(self)
    }
}

impl<'tcx> Repr<'tcx> for ast::Ty {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("type({})", pprust::ty_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Item {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("item({})", tcx.map.node_to_string(self.id))
    }
}

impl<'tcx> Repr<'tcx> for ast::Lifetime {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("lifetime({}: {})", self.id, pprust::lifetime_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Stmt {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("stmt({}: {})",
                ast_util::stmt_id(self),
                pprust::stmt_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ast::Pat {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("pat({}: {})", self.id, pprust::pat_to_string(self))
    }
}

impl<'tcx> Repr<'tcx> for ty::BoundRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::BrAnon(id) => format!("BrAnon({})", id),
            ty::BrNamed(id, name) => {
                format!("BrNamed({}, {})", id.repr(tcx), token::get_name(name))
            }
            ty::BrFresh(id) => format!("BrFresh({})", id),
            ty::BrEnv => "BrEnv".to_string()
        }
    }
}

impl<'tcx> UserString<'tcx> for ty::BoundRegion {
    fn user_string(&self, tcx: &ctxt) -> String {
        if tcx.sess.verbose() {
            return self.repr(tcx);
        }

        match *self {
            BrNamed(_, name) => token::get_name(name).to_string(),
            BrAnon(_) | BrFresh(_) | BrEnv => String::new()
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::Region {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::ReEarlyBound(ref data) => {
                format!("ReEarlyBound({}, {:?}, {}, {})",
                        data.param_id,
                        data.space,
                        data.index,
                        token::get_name(data.name))
            }

            ty::ReLateBound(binder_id, ref bound_region) => {
                format!("ReLateBound({:?}, {})",
                        binder_id,
                        bound_region.repr(tcx))
            }

            ty::ReFree(ref fr) => fr.repr(tcx),

            ty::ReScope(id) => {
                format!("ReScope({:?})", id)
            }

            ty::ReStatic => {
                "ReStatic".to_string()
            }

            ty::ReInfer(ReVar(ref vid)) => {
                format!("{:?}", vid)
            }

            ty::ReInfer(ReSkolemized(id, ref bound_region)) => {
                format!("re_skolemized({}, {})", id, bound_region.repr(tcx))
            }

            ty::ReEmpty => {
                "ReEmpty".to_string()
            }
        }
    }
}

impl<'tcx> UserString<'tcx> for ty::Region {
    fn user_string(&self, tcx: &ctxt) -> String {
        if tcx.sess.verbose() {
            return self.repr(tcx);
        }

        // These printouts are concise.  They do not contain all the information
        // the user might want to diagnose an error, but there is basically no way
        // to fit that into a short string.  Hence the recommendation to use
        // `explain_region()` or `note_and_explain_region()`.
        match *self {
            ty::ReEarlyBound(ref data) => {
                token::get_name(data.name).to_string()
            }
            ty::ReLateBound(_, br) |
            ty::ReFree(ty::FreeRegion { bound_region: br, .. }) |
            ty::ReInfer(ReSkolemized(_, br)) => {
                br.user_string(tcx)
            }
            ty::ReScope(_) |
            ty::ReInfer(ReVar(_)) => String::new(),
            ty::ReStatic => "'static".to_owned(),
            ty::ReEmpty => "'<empty>".to_owned(),
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::FreeRegion {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ReFree({}, {})",
                self.scope.repr(tcx),
                self.bound_region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for region::CodeExtent {
    fn repr(&self, _tcx: &ctxt) -> String {
        match *self {
            region::CodeExtent::ParameterScope { fn_id, body_id } =>
                format!("ParameterScope({}, {})", fn_id, body_id),
            region::CodeExtent::Misc(node_id) =>
                format!("Misc({})", node_id),
            region::CodeExtent::DestructionScope(node_id) =>
                format!("DestructionScope({})", node_id),
            region::CodeExtent::Remainder(rem) =>
                format!("Remainder({}, {})", rem.block, rem.first_statement_index),
        }
    }
}

impl<'tcx> Repr<'tcx> for region::DestructionScopeData {
    fn repr(&self, _tcx: &ctxt) -> String {
        match *self {
            region::DestructionScopeData{ node_id } =>
                format!("DestructionScopeData {{ node_id: {} }}", node_id),
        }
    }
}

impl<'tcx> Repr<'tcx> for ast::DefId {
    fn repr(&self, tcx: &ctxt) -> String {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.krate == ast::LOCAL_CRATE {
            match tcx.map.find(self.node) {
                Some(ast_map::NodeItem(..)) |
                Some(ast_map::NodeForeignItem(..)) |
                Some(ast_map::NodeImplItem(..)) |
                Some(ast_map::NodeTraitItem(..)) |
                Some(ast_map::NodeVariant(..)) |
                Some(ast_map::NodeStructCtor(..)) => {
                    return format!(
                                "{:?}:{}",
                                *self,
                                ty::item_path_str(tcx, *self))
                }
                _ => {}
            }
        }
        return format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TypeScheme<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TypeScheme {{generics: {}, ty: {}}}",
                self.generics.repr(tcx),
                self.ty.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Generics<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Generics(types: {}, regions: {})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::GenericPredicates<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("GenericPredicates(predicates: {})",
                self.predicates.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::InstantiatedPredicates<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("InstantiatedPredicates({})",
                self.predicates.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ItemVariances {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("ItemVariances(types={}, \
                regions={})",
                self.types.repr(tcx),
                self.regions.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Variance {
    fn repr(&self, _: &ctxt) -> String {
        // The first `.to_string()` returns a &'static str (it is not an implementation
        // of the ToString trait). Because of that, we need to call `.to_string()` again
        // if we want to have a `String`.
        let result: &'static str = (*self).to_string();
        result.to_string()
    }
}

impl<'tcx> Repr<'tcx> for ty::ImplOrTraitItem<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("ImplOrTraitItem({})",
                match *self {
                    ty::ImplOrTraitItem::MethodTraitItem(ref i) => i.repr(tcx),
                    ty::ImplOrTraitItem::ConstTraitItem(ref i) => i.repr(tcx),
                    ty::ImplOrTraitItem::TypeTraitItem(ref i) => i.repr(tcx),
                })
    }
}

impl<'tcx> Repr<'tcx> for ty::AssociatedConst<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("AssociatedConst(name: {}, ty: {}, vis: {}, def_id: {})",
                self.name.repr(tcx),
                self.ty.repr(tcx),
                self.vis.repr(tcx),
                self.def_id.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::AssociatedType<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("AssociatedType(name: {}, vis: {}, def_id: {})",
                self.name.repr(tcx),
                self.vis.repr(tcx),
                self.def_id.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::Method<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Method(name: {}, generics: {}, predicates: {}, fty: {}, \
                 explicit_self: {}, vis: {}, def_id: {})",
                self.name.repr(tcx),
                self.generics.repr(tcx),
                self.predicates.repr(tcx),
                self.fty.repr(tcx),
                self.explicit_self.repr(tcx),
                self.vis.repr(tcx),
                self.def_id.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ast::Name {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).to_string()
    }
}

impl<'tcx> UserString<'tcx> for ast::Name {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(*self).to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::Ident {
    fn repr(&self, _tcx: &ctxt) -> String {
        token::get_ident(*self).to_string()
    }
}

impl<'tcx> Repr<'tcx> for ast::ExplicitSelf_ {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::Visibility {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BareFnTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("BareFnTy {{unsafety: {}, abi: {}, sig: {}}}",
                self.unsafety,
                self.abi.to_string(),
                self.sig.repr(tcx))
    }
}


impl<'tcx> Repr<'tcx> for ty::FnSig<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("fn{} -> {}", self.inputs.repr(tcx), self.output.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::FnOutput<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            ty::FnConverging(ty) =>
                format!("FnConverging({0})", ty.repr(tcx)),
            ty::FnDiverging =>
                "FnDiverging".to_string()
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodCallee<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodCallee {{origin: {}, ty: {}, {}}}",
                self.origin.repr(tcx),
                self.ty.repr(tcx),
                self.substs.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodOrigin<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        match self {
            &ty::MethodStatic(def_id) => {
                format!("MethodStatic({})", def_id.repr(tcx))
            }
            &ty::MethodStaticClosure(def_id) => {
                format!("MethodStaticClosure({})", def_id.repr(tcx))
            }
            &ty::MethodTypeParam(ref p) => {
                p.repr(tcx)
            }
            &ty::MethodTraitObject(ref p) => {
                p.repr(tcx)
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodParam<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodParam({},{})",
                self.trait_ref.repr(tcx),
                self.method_num)
    }
}

impl<'tcx> Repr<'tcx> for ty::MethodObject<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("MethodObject({},{},{})",
                self.trait_ref.repr(tcx),
                self.method_num,
                self.vtable_index)
    }
}

impl<'tcx> Repr<'tcx> for ty::BuiltinBound {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> UserString<'tcx> for ty::BuiltinBound {
    fn user_string(&self, _tcx: &ctxt) -> String {
        match *self {
            ty::BoundSend => "Send".to_string(),
            ty::BoundSized => "Sized".to_string(),
            ty::BoundCopy => "Copy".to_string(),
            ty::BoundSync => "Sync".to_string(),
        }
    }
}

impl<'tcx> Repr<'tcx> for Span {
    fn repr(&self, tcx: &ctxt) -> String {
        tcx.sess.codemap().span_to_string(*self).to_string()
    }
}

impl<'tcx, A:UserString<'tcx>> UserString<'tcx> for Rc<A> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let this: &A = &**self;
        this.user_string(tcx)
    }
}

impl<'tcx> UserString<'tcx> for ty::ParamBounds<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        let mut result = Vec::new();
        let s = self.builtin_bounds.user_string(tcx);
        if !s.is_empty() {
            result.push(s);
        }
        for n in &self.trait_bounds {
            result.push(n.user_string(tcx));
        }
        result.connect(" + ")
    }
}

impl<'tcx> Repr<'tcx> for ty::ExistentialBounds<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let mut res = Vec::new();

        let region_str = self.region_bound.repr(tcx);
        if !region_str.is_empty() {
            res.push(region_str);
        }

        for bound in &self.builtin_bounds {
            res.push(bound.repr(tcx));
        }

        for projection_bound in &self.projection_bounds {
            res.push(projection_bound.repr(tcx));
        }

        res.connect("+")
    }
}

impl<'tcx> UserString<'tcx> for ty::BuiltinBounds {
    fn user_string(&self, tcx: &ctxt) -> String {
        self.iter()
            .map(|bb| bb.user_string(tcx))
            .collect::<Vec<String>>()
            .connect("+")
            .to_string()
    }
}

impl<'tcx, T> UserString<'tcx> for ty::Binder<T>
    where T : UserString<'tcx> + TypeFoldable<'tcx>
{
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        // Replace any anonymous late-bound regions with named
        // variants, using gensym'd identifiers, so that we can
        // clearly differentiate between named and unnamed regions in
        // the output. We'll probably want to tweak this over time to
        // decide just how much information to give.
        let mut names = Vec::new();
        let (unbound_value, _) = ty_fold::replace_late_bound_regions(tcx, self, |br| {
            ty::ReLateBound(ty::DebruijnIndex::new(1), match br {
                ty::BrNamed(_, name) => {
                    names.push(token::get_name(name));
                    br
                }
                ty::BrAnon(_) |
                ty::BrFresh(_) |
                ty::BrEnv => {
                    let name = token::gensym("'r");
                    names.push(token::get_name(name));
                    ty::BrNamed(ast_util::local_def(ast::DUMMY_NODE_ID), name)
                }
            })
        });
        let names: Vec<_> = names.iter().map(|s| &s[..]).collect();

        let value_str = unbound_value.user_string(tcx);
        if names.is_empty() {
            value_str
        } else {
            format!("for<{}> {}", names.connect(","), value_str)
        }
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitRef<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        parameterized(tcx, self.substs, self.def_id, &[],
                      || ty::lookup_trait_def(tcx, self.def_id).generics.clone())
    }
}

impl<'tcx> UserString<'tcx> for ty::TyS<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        fn bare_fn_to_string<'tcx>(cx: &ctxt<'tcx>,
                                   opt_def_id: Option<ast::DefId>,
                                   unsafety: ast::Unsafety,
                                   abi: abi::Abi,
                                   ident: Option<ast::Ident>,
                                   sig: &ty::PolyFnSig<'tcx>)
                                   -> String {
            let mut s = String::new();

            match unsafety {
                ast::Unsafety::Normal => {}
                ast::Unsafety::Unsafe => {
                    s.push_str(&unsafety.to_string());
                    s.push(' ');
                }
            };

            if abi != abi::Rust {
                s.push_str(&format!("extern {} ", abi.to_string()));
            };

            s.push_str("fn");

            match ident {
                Some(i) => {
                    s.push(' ');
                    s.push_str(&token::get_ident(i));
                }
                _ => { }
            }

            push_sig_to_string(cx, &mut s, '(', ')', sig);

            match opt_def_id {
                Some(def_id) => {
                    s.push_str(" {");
                    let path_str = ty::item_path_str(cx, def_id);
                    s.push_str(&path_str[..]);
                    s.push_str("}");
                }
                None => { }
            }

            s
        }

        fn closure_to_string<'tcx>(cx: &ctxt<'tcx>,
                                   cty: &ty::ClosureTy<'tcx>,
                                   did: &ast::DefId)
                                   -> String {
            let mut s = String::new();
            s.push_str("[closure");
            push_sig_to_string(cx, &mut s, '(', ')', &cty.sig);
            if cx.sess.verbose() {
                s.push_str(&format!(" id={:?}]", did));
            } else {
                s.push(']');
            }
            s
        }

        fn push_sig_to_string<'tcx>(cx: &ctxt<'tcx>,
                                    s: &mut String,
                                    bra: char,
                                    ket: char,
                                    sig: &ty::PolyFnSig<'tcx>) {
            s.push(bra);
            let strs = sig.0.inputs
                .iter()
                .map(|a| a.user_string(cx))
                .collect::<Vec<_>>();
            s.push_str(&strs.connect(", "));
            if sig.0.variadic {
                s.push_str(", ...");
            }
            s.push(ket);

            match sig.0.output {
                ty::FnConverging(t) => {
                    if !ty::type_is_nil(t) {
                        s.push_str(" -> ");
                        s.push_str(& t.user_string(cx));
                    }
                }
                ty::FnDiverging => {
                    s.push_str(" -> !");
                }
            }
        }

        fn infer_ty_to_string(cx: &ctxt, ty: ty::InferTy) -> String {
            let print_var_ids = cx.sess.verbose();
            match ty {
                ty::TyVar(ref vid) if print_var_ids => vid.repr(cx),
                ty::IntVar(ref vid) if print_var_ids => vid.repr(cx),
                ty::FloatVar(ref vid) if print_var_ids => vid.repr(cx),
                ty::TyVar(_) | ty::IntVar(_) | ty::FloatVar(_) => format!("_"),
                ty::FreshTy(v) => format!("FreshTy({})", v),
                ty::FreshIntTy(v) => format!("FreshIntTy({})", v),
                ty::FreshFloatTy(v) => format!("FreshFloatTy({})", v)
            }
        }

        // pretty print the structural type representation:
        match self.sty {
            TyBool => "bool".to_string(),
            TyChar => "char".to_string(),
            TyInt(t) => ast_util::int_ty_to_string(t, None).to_string(),
            TyUint(t) => ast_util::uint_ty_to_string(t, None).to_string(),
            TyFloat(t) => ast_util::float_ty_to_string(t).to_string(),
            TyBox(typ) => format!("Box<{}>",  typ.user_string(tcx)),
            TyRawPtr(ref tm) => {
                format!("*{} {}", match tm.mutbl {
                    ast::MutMutable => "mut",
                    ast::MutImmutable => "const",
                },  tm.ty.user_string(tcx))
            }
            TyRef(r, ref tm) => {
                let mut buf = "&".to_owned();
                buf.push_str(&r.user_string(tcx));
                if !buf.is_empty() {
                    buf.push_str(" ");
                }
                buf.push_str(&tm.repr(tcx));
                buf
            }
            TyTuple(ref elems) => {
                let strs = elems
                    .iter()
                    .map(|elem| elem.user_string(tcx))
                    .collect::<Vec<_>>();
                match &strs[..] {
                    [ref string] => format!("({},)", string),
                    strs => format!("({})", strs.connect(", "))
                }
            }
            TyBareFn(opt_def_id, ref f) => {
                bare_fn_to_string(tcx, opt_def_id, f.unsafety, f.abi, None, &f.sig)
            }
            TyInfer(infer_ty) => infer_ty_to_string(tcx, infer_ty),
            TyError => "[type error]".to_string(),
            TyParam(ref param_ty) => param_ty.user_string(tcx),
            TyEnum(did, substs) | TyStruct(did, substs) => {
                parameterized(tcx, substs, did, &[],
                              || ty::lookup_item_type(tcx, did).generics)
            }
            TyTrait(ref data) => {
                data.user_string(tcx)
            }
            ty::TyProjection(ref data) => {
                format!("<{} as {}>::{}",
                        data.trait_ref.self_ty().user_string(tcx),
                        data.trait_ref.user_string(tcx),
                        data.item_name.user_string(tcx))
            }
            TyStr => "str".to_string(),
            TyClosure(ref did, substs) => {
                let closure_tys = tcx.closure_tys.borrow();
                closure_tys.get(did).map(|closure_type| {
                    closure_to_string(tcx, &closure_type.subst(tcx, substs), did)
                }).unwrap_or_else(|| {
                    let id_str = if tcx.sess.verbose() {
                        format!(" id={:?}", did)
                    } else {
                        "".to_owned()
                    };


                    if did.krate == ast::LOCAL_CRATE {
                        let span = tcx.map.span(did.node);
                        format!("[closure {}{}]", span.repr(tcx), id_str)
                    } else {
                        format!("[closure{}]", id_str)
                    }
                })
            }
            TyArray(t, sz) => {
                format!("[{}; {}]",  t.user_string(tcx), sz)
            }
            TySlice(t) => {
                format!("[{}]",  t.user_string(tcx))
            }
        }
    }
}

impl<'tcx> UserString<'tcx> for ast::Ident {
    fn user_string(&self, _tcx: &ctxt) -> String {
        token::get_name(self.name).to_string()
    }
}

impl<'tcx> Repr<'tcx> for abi::Abi {
    fn repr(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl<'tcx> UserString<'tcx> for abi::Abi {
    fn user_string(&self, _tcx: &ctxt) -> String {
        self.to_string()
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarId {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarId({};`{}`;{})",
                self.var_id,
                ty::local_var_name_str(tcx, self.var_id),
                self.closure_expr_id)
    }
}

impl<'tcx> Repr<'tcx> for ast::Mutability {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::BorrowKind {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarBorrow {
    fn repr(&self, tcx: &ctxt) -> String {
        format!("UpvarBorrow({}, {})",
                self.kind.repr(tcx),
                self.region.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::UpvarCapture {
    fn repr(&self, tcx: &ctxt) -> String {
        match *self {
            ty::UpvarCapture::ByValue => format!("ByValue"),
            ty::UpvarCapture::ByRef(ref data) => format!("ByRef({})", data.repr(tcx)),
        }
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::FloatVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::RegionVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::TyVid {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", self)
    }
}

impl<'tcx> Repr<'tcx> for ty::IntVarValue {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::IntTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::UintTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ast::FloatTy {
    fn repr(&self, _tcx: &ctxt) -> String {
        format!("{:?}", *self)
    }
}

impl<'tcx> Repr<'tcx> for ty::ExplicitSelfCategory {
    fn repr(&self, _: &ctxt) -> String {
        match *self {
            ty::StaticExplicitSelfCategory => "static",
            ty::ByValueExplicitSelfCategory => "self",
            ty::ByReferenceExplicitSelfCategory(_, ast::MutMutable) => {
                "&mut self"
            }
            ty::ByReferenceExplicitSelfCategory(_, ast::MutImmutable) => "&self",
            ty::ByBoxExplicitSelfCategory => "Box<self>",
        }.to_owned()
    }
}

impl<'tcx> UserString<'tcx> for ParamTy {
    fn user_string(&self, _tcx: &ctxt) -> String {
        format!("{}", token::get_name(self.name))
    }
}

impl<'tcx> Repr<'tcx> for ParamTy {
    fn repr(&self, tcx: &ctxt) -> String {
        let ident = self.user_string(tcx);
        format!("{}/{:?}.{}", ident, self.space, self.idx)
    }
}

impl<'tcx, A:Repr<'tcx>, B:Repr<'tcx>> Repr<'tcx> for (A,B) {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        let &(ref a, ref b) = self;
        format!("({},{})", a.repr(tcx), b.repr(tcx))
    }
}

impl<'tcx, T:Repr<'tcx>> Repr<'tcx> for ty::Binder<T> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("Binder({})", self.0.repr(tcx))
    }
}

impl<'tcx, S, K, V> Repr<'tcx> for HashMap<K, V, S>
    where K: Hash + Eq + Repr<'tcx>,
          V: Repr<'tcx>,
          S: HashState,
{
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("HashMap({})",
                self.iter()
                    .map(|(k,v)| format!("{} => {}", k.repr(tcx), v.repr(tcx)))
                    .collect::<Vec<String>>()
                    .connect(", "))
    }
}

impl<'tcx, T, U> Repr<'tcx> for ty::OutlivesPredicate<T,U>
    where T : Repr<'tcx> + TypeFoldable<'tcx>,
          U : Repr<'tcx> + TypeFoldable<'tcx>,
{
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("OutlivesPredicate({}, {})",
                self.0.repr(tcx),
                self.1.repr(tcx))
    }
}

impl<'tcx, T, U> UserString<'tcx> for ty::OutlivesPredicate<T,U>
    where T : UserString<'tcx> + TypeFoldable<'tcx>,
          U : UserString<'tcx> + TypeFoldable<'tcx>,
{
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} : {}",
                self.0.user_string(tcx),
                self.1.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::EquatePredicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("EquatePredicate({}, {})",
                self.0.repr(tcx),
                self.1.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::EquatePredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} == {}",
                self.0.user_string(tcx),
                self.1.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::TraitPredicate<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("TraitPredicate({})",
                self.trait_ref.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::TraitPredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} : {}",
                self.trait_ref.self_ty().user_string(tcx),
                self.trait_ref.user_string(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{} == {}",
                self.projection_ty.user_string(tcx),
                self.ty.user_string(tcx))
    }
}

impl<'tcx> Repr<'tcx> for ty::ProjectionTy<'tcx> {
    fn repr(&self, tcx: &ctxt<'tcx>) -> String {
        format!("{}::{}",
                self.trait_ref.repr(tcx),
                self.item_name.repr(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::ProjectionTy<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        format!("<{} as {}>::{}",
                self.trait_ref.self_ty().user_string(tcx),
                self.trait_ref.user_string(tcx),
                self.item_name.user_string(tcx))
    }
}

impl<'tcx> UserString<'tcx> for ty::Predicate<'tcx> {
    fn user_string(&self, tcx: &ctxt<'tcx>) -> String {
        match *self {
            ty::Predicate::Trait(ref data) => data.user_string(tcx),
            ty::Predicate::Equate(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::RegionOutlives(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::TypeOutlives(ref predicate) => predicate.user_string(tcx),
            ty::Predicate::Projection(ref predicate) => predicate.user_string(tcx),
        }
    }
}

impl<'tcx> Repr<'tcx> for ast::Unsafety {
    fn repr(&self, _: &ctxt<'tcx>) -> String {
        format!("{:?}", *self)
    }
}
