// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Propagates constants for early reporting of statically known
//! assertion failures



use rustc::mir::{Constant, Literal, Location, Place, Mir, Operand, Rvalue, Local};
use rustc::mir::{NullOp, StatementKind, Statement, BasicBlock, LocalKind};
use rustc::mir::TerminatorKind;
use rustc::mir::visit::{MutVisitor, Visitor};
use rustc::middle::const_val::ConstVal;
use rustc::ty::{TyCtxt, self, Instance};
use rustc::mir::interpret::{Value, PrimVal, GlobalId};
use interpret::{eval_body_with_mir, eval_body, mk_borrowck_eval_cx, unary_op, ValTy};
use rustc::util::nodemap::FxHashMap;
use transform::{MirPass, MirSource};
use syntax::codemap::Span;
use rustc::ty::subst::Substs;

pub struct ConstProp;

impl MirPass for ConstProp {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        trace!("ConstProp starting for {:?}", source.def_id);

        // First, find optimization opportunities. This is done in a pre-pass to keep the MIR
        // read-only so that we can do global analyses on the MIR in the process (e.g.
        // `Place::ty()`).
        let optimizations = {
            let mut optimization_finder = OptimizationFinder::new(mir, tcx, source);
            optimization_finder.visit_mir(mir);
            optimization_finder.optimizations
        };

        // Then carry out those optimizations.
        MutVisitor::visit_mir(&mut ConstPropVisitor { optimizations, tcx }, mir);
        trace!("ConstProp done for {:?}", source.def_id);
    }
}

type Const<'tcx> = (Value, ty::Ty<'tcx>, Span);

pub struct ConstPropVisitor<'a, 'tcx: 'a> {
    optimizations: OptimizationList<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for ConstPropVisitor<'a, 'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if let Some((value, ty, span)) = self.optimizations.const_prop.remove(&location) {
            let value = self.tcx.mk_const(ty::Const {
                val: ConstVal::Value(value),
                ty,
            });
            debug!("Replacing `{:?}` with {:?}", rvalue, value);
            let constant = Constant {
                ty,
                literal: Literal::Value { value },
                span,
            };
            *rvalue = Rvalue::Use(Operand::Constant(box constant));
        }

        self.super_rvalue(rvalue, location)
    }

    fn visit_constant(
        &mut self,
        constant: &mut Constant<'tcx>,
        location: Location,
    ) {
        self.super_constant(constant, location);
        if let Some(&(val, ty, _)) = self.optimizations.constants.get(constant) {
            debug!("Replacing `{:?}` with {:?}:{:?}", constant.literal, val, ty);
            constant.literal = Literal::Value {
                value: self.tcx.mk_const(ty::Const {
                    val: ConstVal::Value(val),
                    ty,
                }),
            };
        }
    }

    fn visit_operand(
        &mut self,
        operand: &mut Operand<'tcx>,
        location: Location,
    ) {
        self.super_operand(operand, location);
        let new = match operand {
            Operand::Move(Place::Local(local)) |
            Operand::Copy(Place::Local(local)) => {
                trace!("trying to read {:?}", local);
                self.optimizations.places.get(&local).cloned()
            },
            _ => return,
        };
        if let Some((value, ty, span)) = new {
            let value = self.tcx.mk_const(ty::Const {
                val: ConstVal::Value(value),
                ty,
            });
            debug!("Replacing `{:?}` with {:?}", operand, value);
            let constant = Constant {
                ty,
                literal: Literal::Value { value },
                span,
            };
            *operand = Operand::Constant(box constant);
        }
    }

    fn visit_terminator_kind(
        &mut self,
        block: BasicBlock,
        kind: &mut TerminatorKind<'tcx>,
        location: Location,
    ) {
        match kind {
            TerminatorKind::SwitchInt { discr: value, .. } |
            TerminatorKind::Yield { value, .. } |
            TerminatorKind::Assert { cond: value, .. } => {
                if let Some((new, ty, span)) = self.optimizations.terminators.remove(&block) {
                    let new = self.tcx.mk_const(ty::Const {
                        val: ConstVal::Value(new),
                        ty,
                    });
                    debug!("Replacing `{:?}` with {:?}", value, new);
                    let constant = Constant {
                        ty,
                        literal: Literal::Value { value: new },
                        span,
                    };
                    *value = Operand::Constant(box constant);
                }
            }
            // FIXME: do this optimization for function calls
            _ => {},
        }
        self.super_terminator_kind(block, kind, location)
    }
}

/// Finds optimization opportunities on the MIR.
struct OptimizationFinder<'b, 'a, 'tcx:'a+'b> {
    mir: &'b Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    source: MirSource,
    optimizations: OptimizationList<'tcx>,
}

impl<'b, 'a, 'tcx:'b> OptimizationFinder<'b, 'a, 'tcx> {
    fn new(
        mir: &'b Mir<'tcx>,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        source: MirSource,
    ) -> OptimizationFinder<'b, 'a, 'tcx> {
        OptimizationFinder {
            mir,
            tcx,
            source,
            optimizations: OptimizationList::default(),
        }
    }

    fn eval_constant(&mut self, c: &Constant<'tcx>) -> Option<Const<'tcx>> {
        if let Some(&val) = self.optimizations.constants.get(c) {
            return Some(val);
        }
        match c.literal {
            Literal::Value { value } => match value.val {
                ConstVal::Value(v) => Some((v, value.ty, c.span)),
                ConstVal::Unevaluated(did, substs) => {
                    let param_env = self.tcx.param_env(self.source.def_id);
                    let instance = Instance::resolve(
                        self.tcx,
                        param_env,
                        did,
                        substs,
                    )?;
                    let cid = GlobalId {
                        instance,
                        promoted: None,
                    };
                    let (value, _, ty) = eval_body(self.tcx, cid, param_env)?;
                    let val = (value, ty, c.span);
                    trace!("evaluated {:?} to {:?}", c, val);
                    self.optimizations.constants.insert(c.clone(), val);
                    Some(val)
                },
            },
            // evaluate the promoted and replace the constant with the evaluated result
            Literal::Promoted { index } => {
                let generics = self.tcx.generics_of(self.source.def_id);
                if generics.parent_types as usize + generics.types.len() > 0 {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let cid = GlobalId {
                    instance,
                    promoted: Some(index),
                };
                let param_env = self.tcx.param_env(self.source.def_id);
                let (value, _, ty) = eval_body_with_mir(self.tcx, cid, self.mir, param_env)?;
                let val = (value, ty, c.span);
                trace!("evaluated {:?} to {:?}", c, val);
                self.optimizations.constants.insert(c.clone(), val);
                Some(val)
            }
        }
    }

    fn eval_operand(&mut self, op: &Operand<'tcx>) -> Option<Const<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c),
            Operand::Move(ref place) | Operand::Copy(ref place) => match *place {
                Place::Local(loc) => self.optimizations.places.get(&loc).cloned(),
                // FIXME(oli-obk): field and index projections
                Place::Projection(_) => None,
                _ => None,
            },
        }
    }

    fn simplify_operand(&mut self, op: &Operand<'tcx>) -> Option<Const<'tcx>> {
        match *op {
            Operand::Constant(ref c) => match c.literal {
                Literal::Value { .. } => None,
                _ => self.eval_operand(op),
            },
            _ => self.eval_operand(op),
        }
    }

    fn const_prop(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place_ty: ty::Ty<'tcx>,
        span: Span,
    ) -> Option<Const<'tcx>> {
        match *rvalue {
            // No need to overwrite an already evaluated constant
            Rvalue::Use(Operand::Constant(box Constant {
                literal: Literal::Value {
                    value: &ty::Const {
                        val: ConstVal::Value(_),
                        ..
                    },
                },
                ..
            })) => None,
            // This branch exists for the sanity type check
            Rvalue::Use(Operand::Constant(ref c)) => {
                assert_eq!(c.ty, place_ty);
                self.eval_constant(c)
            },
            Rvalue::Use(ref op) => {
                self.eval_operand(op)
            },
            Rvalue::Repeat(..) |
            Rvalue::Ref(..) |
            Rvalue::Cast(..) |
            Rvalue::Aggregate(..) |
            Rvalue::NullaryOp(NullOp::Box, _) |
            Rvalue::Discriminant(..) => None,
            // FIXME(oli-obk): evaluate static/constant slice lengths
            Rvalue::Len(_) => None,
            Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                let param_env = self.tcx.param_env(self.source.def_id);
                type_size_of(self.tcx, param_env, ty).map(|n| (
                    Value::ByVal(PrimVal::Bytes(n as u128)),
                    self.tcx.types.usize,
                    span,
                ))
            }
            Rvalue::UnaryOp(op, ref arg) => {
                let def_id = if self.tcx.is_closure(self.source.def_id) {
                    self.tcx.closure_base_def_id(self.source.def_id)
                } else {
                    self.source.def_id
                };
                let generics = self.tcx.generics_of(def_id);
                if generics.parent_types as usize + generics.types.len() > 0 {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let ecx = mk_borrowck_eval_cx(self.tcx, instance, self.mir, span).unwrap();

                let val = self.eval_operand(arg)?;
                let prim = ecx.value_to_primval(ValTy { value: val.0, ty: val.1 }).ok()?;
                let kind = ecx.ty_to_primval_kind(val.1).ok()?;
                match unary_op(op, prim, kind) {
                    Ok(val) => Some((Value::ByVal(val), place_ty, span)),
                    Err(mut err) => {
                        ecx.report(&mut err, false, Some(span));
                        None
                    },
                }
            }
            Rvalue::CheckedBinaryOp(op, ref left, ref right) |
            Rvalue::BinaryOp(op, ref left, ref right) => {
                trace!("rvalue binop {:?} for {:?} and {:?}", op, left, right);
                let left = self.eval_operand(left)?;
                let right = self.eval_operand(right)?;
                let def_id = if self.tcx.is_closure(self.source.def_id) {
                    self.tcx.closure_base_def_id(self.source.def_id)
                } else {
                    self.source.def_id
                };
                let generics = self.tcx.generics_of(def_id);
                let has_generics = generics.parent_types as usize + generics.types.len() > 0;
                if has_generics {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let ecx = mk_borrowck_eval_cx(self.tcx, instance, self.mir, span).unwrap();

                let l = ecx.value_to_primval(ValTy { value: left.0, ty: left.1 }).ok()?;
                let r = ecx.value_to_primval(ValTy { value: right.0, ty: right.1 }).ok()?;
                trace!("const evaluating {:?} for {:?} and {:?}", op, left, right);
                match ecx.binary_op(op, l, left.1, r, right.1) {
                    Ok((val, overflow)) => {
                        let val = if let Rvalue::CheckedBinaryOp(..) = *rvalue {
                            Value::ByValPair(
                                val,
                                PrimVal::from_bool(overflow),
                            )
                        } else {
                            if overflow {
                                use rustc::mir::interpret::EvalErrorKind;
                                let mut err = EvalErrorKind::OverflowingMath.into();
                                ecx.report(&mut err, false, Some(span));
                                return None;
                            }
                            Value::ByVal(val)
                        };
                        Some((val, place_ty, span))
                    },
                    Err(mut err) => {
                        ecx.report(&mut err, false, Some(span));
                        None
                    },
                }
            },
        }
    }
}

fn type_size_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_env: ty::ParamEnv<'tcx>,
                          ty: ty::Ty<'tcx>) -> Option<u64> {
    use rustc::ty::layout::LayoutOf;
    (tcx, param_env).layout_of(ty).ok().map(|layout| layout.size.bytes())
}

struct CanConstProp {
    local: Local,
    can_const_prop: bool,
    // false at the beginning, once set, there are not allowed to be any more assignments
    found_assignment: bool,
}

impl CanConstProp {
    /// returns true if `local` can be propagated
    fn check<'tcx>(local: Local, mir: &Mir<'tcx>) -> bool {
        let mut cpv = CanConstProp {
            local,
            can_const_prop: true,
            found_assignment: false,
        };
        cpv.visit_mir(mir);
        cpv.can_const_prop
    }

    fn is_our_local(&mut self, mut place: &Place) -> bool {
        while let Place::Projection(ref proj) = place {
            place = &proj.base;
        }
        if let Place::Local(local) = *place {
            local == self.local
        } else {
            false
        }
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        self.super_statement(block, statement, location);
        match statement.kind {
            StatementKind::SetDiscriminant { ref place, .. } |
            StatementKind::Assign(ref place, _) => {
                if self.is_our_local(place) {
                    if self.found_assignment {
                        self.can_const_prop = false;
                    } else {
                        self.found_assignment = true
                    }
                }
            },
            StatementKind::InlineAsm { ref outputs, .. } => {
                for place in outputs {
                    if self.is_our_local(place) {
                        if self.found_assignment {
                            self.can_const_prop = false;
                        } else {
                            self.found_assignment = true
                        }
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        if let Rvalue::Ref(_, _, ref place) = *rvalue {
            if self.is_our_local(place) {
                self.can_const_prop = false;
            }
        }
    }
}

impl<'b, 'a, 'tcx> Visitor<'tcx> for OptimizationFinder<'b, 'a, 'tcx> {
    fn visit_constant(
        &mut self,
        constant: &Constant<'tcx>,
        location: Location,
    ) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant);
    }

    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        trace!("visit_statement: {:?}", statement);
        if let StatementKind::Assign(ref place, ref rval) = statement.kind {
            let place_ty = place
                .ty(&self.mir.local_decls, self.tcx)
                .to_ty(self.tcx);
            let span = statement.source_info.span;
            if let Some(value) = self.const_prop(rval, place_ty, span) {
                self.optimizations.const_prop.insert(location, value);
                if let Place::Local(local) = *place {
                    if self.mir.local_kind(local) == LocalKind::Temp
                        && CanConstProp::check(local, self.mir) {
                        trace!("storing {:?} to {:?}", value, local);
                        assert!(self.optimizations.places.insert(local, value).is_none());
                    }
                }
            }
        }
        self.super_statement(block, statement, location);
    }

    fn visit_terminator_kind(
        &mut self,
        block: BasicBlock,
        kind: &TerminatorKind<'tcx>,
        _location: Location,
    ) {
        match kind {
            TerminatorKind::SwitchInt { discr: value, .. } |
            TerminatorKind::Yield { value, .. } |
            TerminatorKind::Assert { cond: value, .. } => {
                if let Some(value) = self.simplify_operand(value) {
                    self.optimizations.terminators.insert(block, value);
                }
            }
            // FIXME: do this optimization for function calls
            _ => {},
        }
    }
}

#[derive(Default)]
struct OptimizationList<'tcx> {
    const_prop: FxHashMap<Location, Const<'tcx>>,
    /// Terminators that get their Operand(s) turned into constants.
    terminators: FxHashMap<BasicBlock, Const<'tcx>>,
    places: FxHashMap<Local, Const<'tcx>>,
    constants: FxHashMap<Constant<'tcx>, Const<'tcx>>,
}
