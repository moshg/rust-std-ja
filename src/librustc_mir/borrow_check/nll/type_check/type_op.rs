// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::type_check::TypeChecker;
use rustc::infer::InferResult;
use rustc::traits::ObligationCause;
use rustc::ty::Ty;

pub(super) trait TypeOp<'gcx, 'tcx> {
    type Output;

    fn perform(
        self,
        type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>,
    ) -> InferResult<'tcx, Self::Output>;
}

pub(super) struct CustomTypeOp<F> {
    closure: F,
}

impl<F> CustomTypeOp<F> {
    pub(super) fn new<'gcx, 'tcx, R>(closure: F) -> Self
    where
        F: FnOnce(&mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
    {
        CustomTypeOp { closure }
    }
}

impl<'gcx, 'tcx, F, R> TypeOp<'gcx, 'tcx> for CustomTypeOp<F>
where
    F: FnOnce(&mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
{
    type Output = R;

    fn perform(self, type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R> {
        (self.closure)(type_checker)
    }
}

pub(super) struct Subtype<'tcx> {
    sub: Ty<'tcx>,
    sup: Ty<'tcx>,
}

impl<'tcx> Subtype<'tcx> {
    pub(super) fn new(sub: Ty<'tcx>, sup: Ty<'tcx>) -> Self {
        Self { sub, sup }
    }
}

impl<'gcx, 'tcx> TypeOp<'gcx, 'tcx> for Subtype<'tcx> {
    type Output = ();

    fn perform(self, type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        type_checker.infcx
            .at(&ObligationCause::dummy(), type_checker.param_env)
            .sup(self.sup, self.sub)
    }
}

