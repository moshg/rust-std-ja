// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{ArgAttribute, ArgType, CastTarget, FnType, LayoutExt, PassMode, Reg, RegKind, Uniform};
use context::CodegenCx;
use rustc::ty::layout::{self, Size};

fn extend_integer_width_mips(arg: &mut ArgType, bits: u64) {
    // Always sign extend u32 values on 64-bit mips
    if let layout::Abi::Scalar(ref scalar) = arg.layout.abi {
        if let layout::Int(i, signed) = scalar.value {
            if !signed && i.size().bits() == 32 {
                if let PassMode::Direct(ref mut attrs) = arg.mode {
                    attrs.set(ArgAttribute::SExt);
                    return;
                }
            }
        }
    }

    arg.extend_integer_width_to(bits);
}

fn bits_to_int_reg(bits: u64) -> Reg {
    if bits <= 8 {
        Reg::i8()
    } else if bits <= 16 {
        Reg::i16()
    } else if bits <= 32 {
        Reg::i32()
    } else {
        Reg::i64()
    }
}

fn float_reg<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, ret: &ArgType<'tcx>, i: usize) -> Option<Reg> {
    match ret.layout.field(cx, i).abi {
        layout::Abi::Scalar(ref scalar) => match scalar.value {
            layout::F32 => Some(Reg::f32()),
            layout::F64 => Some(Reg::f64()),
            _ => None
        },
        _ => None
    }
}

fn classify_ret_ty<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, ret: &mut ArgType<'tcx>) {
    if !ret.layout.is_aggregate() {
        extend_integer_width_mips(ret, 64);
        return;
    }

    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 128 {
        // Unlike other architectures which return aggregates in registers, MIPS n64 limits the
        // use of float registers to structures (not unions) containing exactly one or two
        // float fields.

        if let layout::FieldPlacement::Arbitrary { .. } = ret.layout.fields {
            if ret.layout.fields.count() == 1 {
                if let Some(reg) = float_reg(cx, ret, 0) {
                    ret.cast_to(reg);
                    return;
                }
            } else if ret.layout.fields.count() == 2 {
                if let Some(reg0) = float_reg(cx, ret, 0) {
                    if let Some(reg1) = float_reg(cx, ret, 1) {
                        ret.cast_to(CastTarget::Pair(reg0, reg1));
                        return;
                    }
                }
            }
        }

        // Cast to a uniform int structure
        ret.cast_to(Uniform {
            unit: bits_to_int_reg(bits),
            total: size
        });
    } else {
        ret.make_indirect();
    }
}

fn classify_arg_ty<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, arg: &mut ArgType<'tcx>) {
    if !arg.layout.is_aggregate() {
        extend_integer_width_mips(arg, 64);
        return;
    }

    let dl = &cx.tcx.data_layout;
    let size = arg.layout.size;
    let mut prefix = [RegKind::Integer; 8];
    let mut prefix_index = 0;

    match arg.layout.fields {
        layout::FieldPlacement::Array { .. } => {
            // Arrays are passed indirectly
            arg.make_indirect();
            return;
        }
        layout::FieldPlacement::Union(_) => {
            // Unions and are always treated as a series of 64-bit integer chunks
        },
        layout::FieldPlacement::Arbitrary { .. } => {
            // Structures are split up into a series of 64-bit integer chunks, but any aligned
            // doubles not part of another aggregate are passed as floats.
            let mut last_offset = Size::from_bytes(0);

            for i in 0..arg.layout.fields.count() {
                let field = arg.layout.field(cx, i);
                let offset = arg.layout.fields.offset(i);

                // We only care about aligned doubles
                if let layout::Abi::Scalar(ref scalar) = field.abi {
                    if let layout::F64 = scalar.value {
                        if offset.is_abi_aligned(dl.f64_align) {
                            // Skip over enough integers to cover [last_offset, offset)
                            assert!(last_offset.is_abi_aligned(dl.f64_align));
                            prefix_index += ((offset - last_offset).bits() / 64) as usize;

                            if prefix_index >= prefix.len() {
                                break;
                            }

                            prefix[prefix_index] = RegKind::Float;
                            prefix_index += 1;
                            last_offset = offset + Reg::f64().size;
                        }
                    }
                }
            }
        }
    };

    // Extract first 8 chunks as the prefix
    arg.cast_to(CastTarget::ChunkedPrefix {
        prefix: prefix,
        chunk: Size::from_bytes(8),
        total: size
    });
}

pub fn compute_abi_info<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg);
    }
}
