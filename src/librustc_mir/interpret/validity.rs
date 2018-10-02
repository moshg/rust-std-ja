// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Write;

use syntax_pos::symbol::Symbol;
use rustc::ty::layout::{self, Size};
use rustc::ty::{self, Ty};
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::interpret::{
    Scalar, AllocType, EvalResult, EvalErrorKind, PointerArithmetic
};

use super::{
    OpTy, MPlaceTy, Machine, EvalContext, ScalarMaybeUndef
};

macro_rules! validation_failure {
    ($what:expr, $where:expr, $details:expr) => {{
        let where_ = path_format($where);
        let where_ = if where_.is_empty() {
            String::new()
        } else {
            format!(" at {}", where_)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}, but expected {}",
            $what, where_, $details,
        )))
    }};
    ($what:expr, $where:expr) => {{
        let where_ = path_format($where);
        let where_ = if where_.is_empty() {
            String::new()
        } else {
            format!(" at {}", where_)
        };
        err!(ValidationFailure(format!(
            "encountered {}{}",
            $what, where_,
        )))
    }};
}

macro_rules! try_validation {
    ($e:expr, $what:expr, $where:expr) => {{
        match $e {
            Ok(x) => x,
            Err(_) => return validation_failure!($what, $where),
        }
    }}
}

/// We want to show a nice path to the invalid field for diagnotsics,
/// but avoid string operations in the happy case where no error happens.
/// So we track a `Vec<PathElem>` where `PathElem` contains all the data we
/// need to later print something for the user.
#[derive(Copy, Clone, Debug)]
pub enum PathElem {
    Field(Symbol),
    ClosureVar(Symbol),
    ArrayElem(usize),
    TupleElem(usize),
    Deref,
    Tag,
}

/// State for tracking recursive validation of references
pub struct RefTracking<'tcx> {
    pub seen: FxHashSet<(OpTy<'tcx>)>,
    pub todo: Vec<(OpTy<'tcx>, Vec<PathElem>)>,
}

impl<'tcx> RefTracking<'tcx> {
    pub fn new(op: OpTy<'tcx>) -> Self {
        let mut ref_tracking = RefTracking {
            seen: FxHashSet(),
            todo: vec![(op, Vec::new())],
        };
        ref_tracking.seen.insert(op);
        ref_tracking
    }
}

// Adding a Deref and making a copy of the path to be put into the queue
// always go together.  This one does it with only new allocation.
fn path_clone_and_deref(path: &Vec<PathElem>) -> Vec<PathElem> {
    let mut new_path = Vec::with_capacity(path.len()+1);
    new_path.clone_from(path);
    new_path.push(PathElem::Deref);
    new_path
}

/// Format a path
fn path_format(path: &Vec<PathElem>) -> String {
    use self::PathElem::*;

    let mut out = String::new();
    for elem in path.iter() {
        match elem {
            Field(name) => write!(out, ".{}", name).unwrap(),
            ClosureVar(name) => write!(out, ".<closure-var({})>", name).unwrap(),
            TupleElem(idx) => write!(out, ".{}", idx).unwrap(),
            ArrayElem(idx) => write!(out, "[{}]", idx).unwrap(),
            Deref =>
                // This does not match Rust syntax, but it is more readable for long paths -- and
                // some of the other items here also are not Rust syntax.  Actually we can't
                // even use the usual syntax because we are just showing the projections,
                // not the root.
                write!(out, ".<deref>").unwrap(),
            Tag => write!(out, ".<enum-tag>").unwrap(),
        }
    }
    out
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    fn validate_scalar(
        &self,
        value: ScalarMaybeUndef,
        size: Size,
        scalar: &layout::Scalar,
        path: &Vec<PathElem>,
        ty: Ty,
    ) -> EvalResult<'tcx> {
        trace!("validate scalar: {:#?}, {:#?}, {:#?}, {}", value, size, scalar, ty);
        let (lo, hi) = scalar.valid_range.clone().into_inner();

        let value = match value {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return validation_failure!("undefined bytes", path),
        };

        let bits = match value {
            Scalar::Bits { bits, size: value_size } => {
                assert_eq!(value_size as u64, size.bytes());
                bits
            },
            Scalar::Ptr(_) => {
                match ty.sty {
                    ty::Bool |
                    ty::Char |
                    ty::Float(_) |
                    ty::Int(_) |
                    ty::Uint(_) => {
                        return validation_failure!(
                                "a pointer",
                                path,
                                format!("the type {}", ty.sty)
                            );
                    }
                    ty::RawPtr(_) |
                    ty::Ref(_, _, _) |
                    ty::FnPtr(_) => {}
                    _ => { unreachable!(); }
                }

                let ptr_size = self.pointer_size();
                let ptr_max = u128::max_value() >> (128 - ptr_size.bits());
                return if lo > hi {
                    if lo - hi == 1 {
                        // no gap, all values are ok
                        Ok(())
                    } else if hi < ptr_max || lo > 1 {
                        let max = u128::max_value() >> (128 - size.bits());
                        validation_failure!(
                            "pointer",
                            path,
                            format!("something in the range {:?} or {:?}", 0..=lo, hi..=max)
                        )
                    } else {
                        Ok(())
                    }
                } else if hi < ptr_max || lo > 1 {
                    validation_failure!(
                        "pointer",
                        path,
                        format!("something in the range {:?}", scalar.valid_range)
                    )
                } else {
                    Ok(())
                };
            },
        };

        // char gets a special treatment, because its number space is not contiguous so `TyLayout`
        // has no special checks for chars
        match ty.sty {
            ty::Char => {
                debug_assert_eq!(size.bytes(), 4);
                if ::std::char::from_u32(bits as u32).is_none() {
                    return validation_failure!(
                        "character",
                        path,
                        "a valid unicode codepoint"
                    );
                }
            }
            _ => {},
        }

        use std::ops::RangeInclusive;
        let in_range = |bound: RangeInclusive<u128>| bound.contains(&bits);
        if lo > hi {
            if in_range(0..=hi) || in_range(lo..=u128::max_value()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?} or {:?}", ..=hi, lo..)
                )
            }
        } else {
            if in_range(scalar.valid_range.clone()) {
                Ok(())
            } else {
                validation_failure!(
                    bits,
                    path,
                    format!("something in the range {:?}", scalar.valid_range)
                )
            }
        }
    }

    /// Validate a reference, potentially recursively. `place` is assumed to already be
    /// dereferenced, i.e. it describes the target.
    fn validate_ref(
        &self,
        place: MPlaceTy<'tcx>,
        path: &mut Vec<PathElem>,
        ref_tracking: Option<&mut RefTracking<'tcx>>,
    ) -> EvalResult<'tcx> {
        // Before we do anything else, make sure this is entirely in-bounds.
        if !place.layout.is_zst() {
            let ptr = try_validation!(place.ptr.to_ptr(),
                "integer pointer in non-ZST reference", path);
            let size = self.size_and_align_of(place.extra, place.layout)?.0;
            try_validation!(self.memory.check_bounds(ptr, size, false),
                "dangling reference (not entirely in bounds)", path);
            // Skip recursion for some external statics
            let alloc_kind = self.tcx.alloc_map.lock().get(ptr.alloc_id);
            if let Some(AllocType::Static(did)) = alloc_kind {
                // statics from other crates are already checked.
                // they might be checked at a different type, but for now we want
                // to avoid recursing too deeply.
                // extern statics cannot be validated as they have no body.
                if !did.is_local() || self.tcx.is_foreign_item(did) {
                    return Ok(());
                }
            }
        }
        // Check if we have encountered this pointer+layout combination
        // before.  Proceed recursively even for integer pointers, no
        // reason to skip them! They are valid for some ZST, but not for others
        // (e.g. `!` is a ZST).
        let op = place.into();
        if let Some(ref_tracking) = ref_tracking {
            if ref_tracking.seen.insert(op) {
                trace!("Recursing below ptr {:#?}", *op);
                ref_tracking.todo.push((op, path_clone_and_deref(path)));
            }
        }
        Ok(())
    }

    /// This function checks the data at `op`.  `op` is assumed to cover valid memory if it
    /// is an indirect operand.
    /// It will error if the bits at the destination do not match the ones described by the layout.
    /// The `path` may be pushed to, but the part that is present when the function
    /// starts must not be changed!
    pub fn validate_operand(
        &self,
        dest: OpTy<'tcx>,
        path: &mut Vec<PathElem>,
        mut ref_tracking: Option<&mut RefTracking<'tcx>>,
    ) -> EvalResult<'tcx> {
        trace!("validate_operand: {:?}, {:#?}", *dest, dest.layout);

        // Find the right variant.  We have to handle this as a prelude, not via
        // proper recursion with the new inner layout, to be able to later nicely
        // print the field names of the enum field that is being accessed.
        let (variant, dest) = match dest.layout.variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                let variant = match self.read_discriminant(dest) {
                    Ok(res) => res.1,
                    Err(err) => match err.kind {
                        EvalErrorKind::InvalidDiscriminant(val) =>
                            return validation_failure!(
                                format!("invalid enum discriminant {}", val), path
                            ),
                        _ =>
                            return validation_failure!(
                                format!("non-integer enum discriminant"), path
                            ),
                    }
                };
                let inner_dest = self.operand_downcast(dest, variant)?;
                // Put the variant projection onto the path, as a field
                path.push(PathElem::Field(dest.layout.ty
                                          .ty_adt_def()
                                          .unwrap()
                                          .variants[variant].name));
                trace!("variant layout: {:#?}", dest.layout);
                (variant, inner_dest)
            },
            layout::Variants::Single { index } => {
                // Pre-processing for trait objects: Treat them at their real type.
                // (We do not do this for slices and strings: For slices it is not needed,
                // `mplace_array_fields` does the right thing, and for strings there is no
                // real type that would show the actual length.)
                let dest = match dest.layout.ty.sty {
                    ty::Dynamic(..) => {
                        let dest = dest.to_mem_place(); // immediate trait objects are not a thing
                        try_validation!(self.unpack_dyn_trait(dest),
                            "invalid vtable in fat pointer", path).1.into()
                    }
                    _ => dest
                };
                (index, dest)
            }
        };

        // Remember the length, in case we need to truncate
        let path_len = path.len();

        // Validate all fields
        match dest.layout.fields {
            // Primitives appear as Union with 0 fields -- except for fat pointers.
            // We still check `layout.fields`, not `layout.abi`, because `layout.abi`
            // is `Scalar` for newtypes around scalars, but we want to descend through the
            // fields to get a proper `path`.
            layout::FieldPlacement::Union(0) => {
                match dest.layout.abi {
                    // nothing to do, whatever the pointer points to, it is never going to be read
                    layout::Abi::Uninhabited =>
                        return validation_failure!("a value of an uninhabited type", path),
                    // check that the scalar is a valid pointer or that its bit range matches the
                    // expectation.
                    layout::Abi::Scalar(ref scalar_layout) => {
                        let size = scalar_layout.value.size(self);
                        let value = try_validation!(self.read_value(dest),
                            "uninitialized or unrepresentable data", path);
                        let scalar = value.to_scalar_or_undef();
                        self.validate_scalar(scalar, size, scalar_layout, &path, dest.layout.ty)?;
                        // Recursively check *safe* references
                        if dest.layout.ty.builtin_deref(true).is_some() &&
                            !dest.layout.ty.is_unsafe_ptr()
                        {
                            self.validate_ref(self.ref_to_mplace(value)?, path, ref_tracking)?;
                        }
                    },
                    _ => bug!("bad abi for FieldPlacement::Union(0): {:#?}", dest.layout.abi),
                }
            }
            layout::FieldPlacement::Arbitrary { .. }
                if dest.layout.ty.builtin_deref(true).is_some() =>
            {
                // This is a fat pointer.
                let ptr = try_validation!(self.read_value(dest.into()),
                    "undefined location in fat pointer", path);
                let ptr = try_validation!(self.ref_to_mplace(ptr),
                    "undefined metadata in fat pointer", path);
                // check metadata early, for better diagnostics
                match self.tcx.struct_tail(ptr.layout.ty).sty {
                    ty::Dynamic(..) => {
                        let vtable = try_validation!(ptr.extra.unwrap().to_ptr(),
                            "non-pointer vtable in fat pointer", path);
                        try_validation!(self.read_drop_type_from_vtable(vtable),
                            "invalid drop fn in vtable", path);
                        try_validation!(self.read_size_and_align_from_vtable(vtable),
                            "invalid size or align in vtable", path);
                        // FIXME: More checks for the vtable.
                    }
                    ty::Slice(..) | ty::Str => {
                        try_validation!(ptr.extra.unwrap().to_usize(self),
                            "non-integer slice length in fat pointer", path);
                    }
                    _ =>
                        bug!("Unexpected unsized type tail: {:?}",
                            self.tcx.struct_tail(ptr.layout.ty)
                        ),
                }
                // for safe ptrs, recursively check it
                if !dest.layout.ty.is_unsafe_ptr() {
                    self.validate_ref(ptr, path, ref_tracking)?;
                }
            }
            // Compound data structures
            layout::FieldPlacement::Union(_) => {
                // We can't check unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
            },
            layout::FieldPlacement::Array { stride, .. } if !dest.layout.is_zst() => {
                let dest = dest.to_mem_place(); // non-ZST array/slice/str cannot be immediate
                match dest.layout.ty.sty {
                    // Special handling for strings to verify UTF-8
                    ty::Str => {
                        try_validation!(self.read_str(dest),
                            "uninitialized or non-UTF-8 data in str", path);
                    }
                    // Special handling for arrays/slices of builtin integer types
                    ty::Array(tys, ..) | ty::Slice(tys) if {
                        // This optimization applies only for integer types
                        match tys.sty {
                            ty::Int(..) | ty::Uint(..) => true,
                            _ => false,
                        }
                    } => {
                        // This is the length of the array/slice.
                        let len = dest.len(self)?;
                        // Since primitive types are naturally aligned and tightly packed in arrays,
                        // we can use the stride to get the size of the integral type.
                        let ty_size = stride.bytes();
                        // This is the size in bytes of the whole array.
                        let size = Size::from_bytes(ty_size * len);

                        match self.memory.read_bytes(dest.ptr, size) {
                            // In the happy case, we needn't check anything else.
                            Ok(_) => {},
                            // Some error happened, try to provide a more detailed description.
                            Err(err) => {
                                // For some errors we might be able to provide extra information
                                match err.kind {
                                    EvalErrorKind::ReadUndefBytes(offset) => {
                                        // Some byte was undefined, determine which
                                        // element that byte belongs to so we can
                                        // provide an index.
                                        let i = (offset.bytes() / ty_size) as usize;
                                        path.push(PathElem::ArrayElem(i));

                                        return validation_failure!(
                                            "undefined bytes", path
                                        )
                                    },
                                    // Other errors shouldn't be possible
                                    _ => return Err(err),
                                }
                            }
                        }
                    },
                    _ => {
                        // This handles the unsized case correctly as well, as well as
                        // SIMD an all sorts of other array-like types.
                        for (i, field) in self.mplace_array_fields(dest)?.enumerate() {
                            let field = field?;
                            path.push(PathElem::ArrayElem(i));
                            self.validate_operand(
                                field.into(),
                                path,
                                ref_tracking.as_mut().map(|r| &mut **r)
                            )?;
                            path.truncate(path_len);
                        }
                    }
                }
            },
            layout::FieldPlacement::Array { .. } => {
                // An empty array.  Nothing to do.
            }
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                for i in 0..offsets.len() {
                    let field = self.operand_field(dest, i as u64)?;
                    path.push(self.aggregate_field_path_elem(dest.layout.ty, variant, i));
                    self.validate_operand(field, path, ref_tracking.as_mut().map(|r| &mut **r))?;
                    path.truncate(path_len);
                }
            }
        }
        Ok(())
    }

    fn aggregate_field_path_elem(&self, ty: Ty<'tcx>, variant: usize, field: usize) -> PathElem {
        match ty.sty {
            // generators and closures.
            ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                let node_id = self.tcx.hir.as_local_node_id(def_id).unwrap();
                let freevar = self.tcx.with_freevars(node_id, |fv| fv[field]);
                PathElem::ClosureVar(self.tcx.hir.name(freevar.var_id()))
            }

            // tuples
            ty::Tuple(_) => PathElem::TupleElem(field),

            // enums
            ty::Adt(def, ..) if def.is_enum() => {
                let variant = &def.variants[variant];
                PathElem::Field(variant.fields[field].ident.name)
            }

            // other ADTs
            ty::Adt(def, _) => PathElem::Field(def.non_enum_variant().fields[field].ident.name),

            // nothing else has an aggregate layout
            _ => bug!("aggregate_field_path_elem: got non-aggregate type {:?}", ty),
        }
    }
}
