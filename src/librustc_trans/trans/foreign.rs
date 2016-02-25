// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::link;
use llvm::{ValueRef, get_param};
use llvm;
use middle::weak_lang_items;
use trans::abi::{Abi, FnType};
use trans::attributes;
use trans::base::{llvm_linkage_by_name, push_ctxt};
use trans::base;
use trans::build::*;
use trans::common::*;
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use trans::value::Value;
use middle::infer;
use middle::ty::{self, Ty, TyCtxt};
use middle::subst::Substs;

use std::cmp;
use std::iter::once;
use libc::c_uint;
use syntax::attr;
use syntax::parse::token::{InternedString, special_idents};
use syntax::ast;
use syntax::attr::AttrMetaMethods;

use rustc_front::hir;

///////////////////////////////////////////////////////////////////////////
// Calls to external functions

pub fn register_static(ccx: &CrateContext,
                       foreign_item: &hir::ForeignItem) -> ValueRef {
    let ty = ccx.tcx().node_id_to_type(foreign_item.id);
    let llty = type_of::type_of(ccx, ty);

    let ident = link_name(foreign_item.name, &foreign_item.attrs);
    let c = match attr::first_attr_value_str_by_name(&foreign_item.attrs,
                                                     "linkage") {
        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        Some(name) => {
            let linkage = match llvm_linkage_by_name(&name) {
                Some(linkage) => linkage,
                None => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "invalid linkage specified");
                }
            };
            let llty2 = match ty.sty {
                ty::TyRawPtr(ref mt) => type_of::type_of(ccx, mt.ty),
                _ => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "must have type `*T` or `*mut T`");
                }
            };
            unsafe {
                // Declare a symbol `foo` with the desired linkage.
                let g1 = declare::declare_global(ccx, &ident[..], llty2);
                llvm::SetLinkage(g1, linkage);

                // Declare an internal global `extern_with_linkage_foo` which
                // is initialized with the address of `foo`.  If `foo` is
                // discarded during linking (for example, if `foo` has weak
                // linkage and there are no definitions), then
                // `extern_with_linkage_foo` will instead be initialized to
                // zero.
                let mut real_name = "_rust_extern_with_linkage_".to_string();
                real_name.push_str(&ident);
                let g2 = declare::define_global(ccx, &real_name[..], llty).unwrap_or_else(||{
                    ccx.sess().span_fatal(foreign_item.span,
                                          &format!("symbol `{}` is already defined", ident))
                });
                llvm::SetLinkage(g2, llvm::InternalLinkage);
                llvm::LLVMSetInitializer(g2, g1);
                g2
            }
        }
        None => // Generate an external declaration.
            declare::declare_global(ccx, &ident[..], llty),
    };

    // Handle thread-local external statics.
    for attr in foreign_item.attrs.iter() {
        if attr.check_name("thread_local") {
            llvm::set_thread_local(c, true);
        }
    }

    return c;
}

/// Prepares a call to a native function. This requires adapting
/// from the Rust argument passing rules to the native rules.
///
/// # Parameters
///
/// - `callee_ty`: Rust type for the function we are calling
/// - `llfn`: the function pointer we are calling
/// - `llretptr`: where to store the return value of the function
/// - `llargs_rust`: a list of the argument values, prepared
///   as they would be if calling a Rust function
/// - `passed_arg_tys`: Rust type for the arguments. Normally we
///   can derive these from callee_ty but in the case of variadic
///   functions passed_arg_tys will include the Rust type of all
///   the arguments including the ones not specified in the fn's signature.
pub fn trans_native_call<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     callee_ty: Ty<'tcx>,
                                     llfn: ValueRef,
                                     llretptr: ValueRef,
                                     llargs_rust: &[ValueRef],
                                     passed_arg_tys: Vec<Ty<'tcx>>,
                                     call_debug_loc: DebugLoc)
                                     -> Block<'blk, 'tcx>
{
    let ccx = bcx.ccx();

    debug!("trans_native_call(callee_ty={:?}, llfn={:?}, llretptr={:?})",
           callee_ty, Value(llfn), Value(llretptr));

    let (fn_abi, fn_sig) = match callee_ty.sty {
        ty::TyFnDef(_, _, ref fn_ty) |
        ty::TyFnPtr(ref fn_ty) => (fn_ty.abi, &fn_ty.sig),
        _ => ccx.sess().bug("trans_native_call called on non-function type")
    };
    let fn_sig = ccx.tcx().erase_late_bound_regions(fn_sig);
    let fn_sig = infer::normalize_associated_type(ccx.tcx(), &fn_sig);

    let extra_args = &passed_arg_tys[fn_sig.inputs.len()..];
    let fn_type = FnType::new(ccx, fn_abi, &fn_sig, extra_args);

    let mut llargs_foreign = Vec::new();

    // If the foreign ABI expects return value by pointer, supply the
    // pointer that Rust gave us. Sometimes we have to bitcast
    // because foreign fns return slightly different (but equivalent)
    // views on the same type (e.g., i64 in place of {i32,i32}).
    if fn_type.ret.is_indirect() {
        match fn_type.ret.cast {
            Some(ty) => {
                let llcastedretptr =
                    BitCast(bcx, llretptr, ty.ptr_to());
                llargs_foreign.push(llcastedretptr);
            }
            None => {
                llargs_foreign.push(llretptr);
            }
        }
    }

    let mut i = 0;
    for &passed_arg_ty in &passed_arg_tys {
        let arg_ty = fn_type.args[i];

        if arg_ty.is_ignore() {
            i += 1;
            continue;
        }

        if type_is_fat_ptr(ccx.tcx(), passed_arg_ty) {
            // Fat pointers are one pointer and one integer or pointer.
            let (a, b) = (fn_type.args[i], fn_type.args[i + 1]);
            assert_eq!((a.cast, b.cast), (None, None));
            assert!(!a.is_indirect() && !b.is_indirect());

            if let Some(ty) = a.pad {
                llargs_foreign.push(C_undef(ty));
            }
            llargs_foreign.push(llargs_rust[i]);
            i += 1;

            if let Some(ty) = b.pad {
                llargs_foreign.push(C_undef(ty));
            }
            llargs_foreign.push(llargs_rust[i]);
            i += 1;
            continue;
        }

        // Does Rust pass this argument by pointer?
        let rust_indirect = type_of::arg_is_indirect(ccx, passed_arg_ty);

        let mut llarg_rust = llargs_rust[i];
        i += 1;
        debug!("argument {}, llarg_rust={:?}, rust_indirect={}, arg_ty={:?}",
               i,
               Value(llarg_rust),
               rust_indirect,
               arg_ty);

        // Ensure that we always have the Rust value indirectly,
        // because it makes bitcasting easier.
        if !rust_indirect {
            let scratch = base::alloc_ty(bcx, passed_arg_ty, "__arg");
            base::store_ty(bcx, llarg_rust, scratch, passed_arg_ty);
            llarg_rust = scratch;
        }

        debug!("llarg_rust={:?} (after indirection)",
               Value(llarg_rust));

        // Check whether we need to do any casting
        if let Some(ty) = arg_ty.cast {
            llarg_rust = BitCast(bcx, llarg_rust, ty.ptr_to());
        }

        debug!("llarg_rust={:?} (after casting)",
               Value(llarg_rust));

        // Finally, load the value if needed for the foreign ABI
        let foreign_indirect = arg_ty.is_indirect();
        let llarg_foreign = if foreign_indirect {
            llarg_rust
        } else if passed_arg_ty.is_bool() {
            let val = LoadRangeAssert(bcx, llarg_rust, 0, 2, llvm::False);
            Trunc(bcx, val, Type::i1(bcx.ccx()))
        } else {
            Load(bcx, llarg_rust)
        };

        debug!("argument {}, llarg_foreign={:?}",
               i, Value(llarg_foreign));

        // fill padding with undef value
        if let Some(ty) = arg_ty.pad {
            llargs_foreign.push(C_undef(ty));
        }
        llargs_foreign.push(llarg_foreign);
    }

    // A function pointer is called without the declaration available, so we have to apply
    // any attributes with ABI implications directly to the call instruction.


    let llforeign_retval = CallWithConv(bcx,
                                        llfn,
                                        &llargs_foreign[..],
                                        fn_type.cconv,
                                        Some(fn_type.llvm_attrs()),
                                        call_debug_loc);

    // If the function we just called does not use an outpointer,
    // store the result into the rust outpointer. Cast the outpointer
    // type to match because some ABIs will use a different type than
    // the Rust type. e.g., a {u32,u32} struct could be returned as
    // u64.
    let llrust_ret_ty = fn_type.ret.original_ty;
    if llrust_ret_ty != Type::void(ccx) && !fn_type.ret.is_indirect() {
        let llforeign_ret_ty = fn_type.ret.cast.unwrap_or(llrust_ret_ty);

        debug!("llretptr={:?}", Value(llretptr));
        debug!("llforeign_retval={:?}", Value(llforeign_retval));
        debug!("llrust_ret_ty={:?}", llrust_ret_ty);
        debug!("llforeign_ret_ty={:?}", llforeign_ret_ty);

        if llrust_ret_ty == llforeign_ret_ty {
            match fn_sig.output {
                ty::FnConverging(result_ty) => {
                    base::store_ty(bcx, llforeign_retval, llretptr, result_ty)
                }
                ty::FnDiverging => {}
            }
        } else {
            // The actual return type is a struct, but the ABI
            // adaptation code has cast it into some scalar type.  The
            // code that follows is the only reliable way I have
            // found to do a transform like i64 -> {i32,i32}.
            // Basically we dump the data onto the stack then memcpy it.
            //
            // Other approaches I tried:
            // - Casting rust ret pointer to the foreign type and using Store
            //   is (a) unsafe if size of foreign type > size of rust type and
            //   (b) runs afoul of strict aliasing rules, yielding invalid
            //   assembly under -O (specifically, the store gets removed).
            // - Truncating foreign type to correct integral type and then
            //   bitcasting to the struct type yields invalid cast errors.
            let llscratch = base::alloca(bcx, llforeign_ret_ty, "__cast");
            base::call_lifetime_start(bcx, llscratch);
            Store(bcx, llforeign_retval, llscratch);
            let llscratch_i8 = BitCast(bcx, llscratch, Type::i8(ccx).ptr_to());
            let llretptr_i8 = BitCast(bcx, llretptr, Type::i8(ccx).ptr_to());
            let llrust_size = machine::llsize_of_store(ccx, llrust_ret_ty);
            let llforeign_align = machine::llalign_of_min(ccx, llforeign_ret_ty);
            let llrust_align = machine::llalign_of_min(ccx, llrust_ret_ty);
            let llalign = cmp::min(llforeign_align, llrust_align);
            debug!("llrust_size={}", llrust_size);
            base::call_memcpy(bcx, llretptr_i8, llscratch_i8,
                              C_uint(ccx, llrust_size), llalign as u32);
            base::call_lifetime_end(bcx, llscratch);
        }
    }

    return bcx;
}

///////////////////////////////////////////////////////////////////////////
// Rust functions with foreign ABIs
//
// These are normal Rust functions defined with foreign ABIs.  For
// now, and perhaps forever, we translate these using a "layer of
// indirection". That is, given a Rust declaration like:
//
//     extern "C" fn foo(i: u32) -> u32 { ... }
//
// we will generate a function like:
//
//     S foo(T i) {
//         S r;
//         foo0(&r, NULL, i);
//         return r;
//     }
//
//     #[inline_always]
//     void foo0(uint32_t *r, void *env, uint32_t i) { ... }
//
// Here the (internal) `foo0` function follows the Rust ABI as normal,
// where the `foo` function follows the C ABI. We rely on LLVM to
// inline the one into the other. Of course we could just generate the
// correct code in the first place, but this is much simpler.

pub fn trans_rust_fn_with_foreign_abi<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                decl: &hir::FnDecl,
                                                body: &hir::Block,
                                                attrs: &[ast::Attribute],
                                                llwrapfn: ValueRef,
                                                param_substs: &'tcx Substs<'tcx>,
                                                id: ast::NodeId,
                                                hash: Option<&str>) {
    let _icx = push_ctxt("foreign::build_foreign_fn");

    let fnty = ccx.tcx().node_id_to_type(id);
    let mty = monomorphize::apply_param_substs(ccx.tcx(), param_substs, &fnty);
    let (fn_abi, fn_sig) = match mty.sty {
        ty::TyFnDef(_, _, ref fn_ty) => (fn_ty.abi, &fn_ty.sig),
        _ => ccx.sess().bug("trans_rust_fn_with_foreign_abi called on non-function type")
    };
    let fn_sig = ccx.tcx().erase_late_bound_regions(fn_sig);
    let fn_sig = infer::normalize_associated_type(ccx.tcx(), &fn_sig);
    let fn_ty = FnType::new(ccx, fn_abi, &fn_sig, &[]);

    unsafe { // unsafe because we call LLVM operations
        // Build up the Rust function (`foo0` above).
        let llrustfn = build_rust_fn(ccx, decl, body, param_substs, attrs, id, hash);

        // Build up the foreign wrapper (`foo` above).
        return build_wrap_fn(ccx, llrustfn, llwrapfn, &fn_sig, &fn_ty, mty);
    }

    fn build_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               decl: &hir::FnDecl,
                               body: &hir::Block,
                               param_substs: &'tcx Substs<'tcx>,
                               attrs: &[ast::Attribute],
                               id: ast::NodeId,
                               hash: Option<&str>)
                               -> ValueRef
    {
        let _icx = push_ctxt("foreign::foreign::build_rust_fn");
        let tcx = ccx.tcx();
        let t = tcx.node_id_to_type(id);
        let t = monomorphize::apply_param_substs(tcx, param_substs, &t);

        let path =
            tcx.map.def_path(tcx.map.local_def_id(id))
                   .into_iter()
                   .map(|e| e.data.as_interned_str())
                   .chain(once(special_idents::clownshoe_abi.name.as_str()));
        let ps = link::mangle(path, hash);

        // Compute the type that the function would have if it were just a
        // normal Rust function. This will be the type of the wrappee fn.
        let rust_fn_ty = match t.sty {
            ty::TyFnDef(_, _, ref f) => {
                assert!(f.abi != Abi::Rust);
                assert!(f.abi != Abi::RustIntrinsic);
                assert!(f.abi != Abi::PlatformIntrinsic);
                tcx.mk_fn_ptr(ty::BareFnTy {
                    unsafety: f.unsafety,
                    abi: Abi::Rust,
                    sig: f.sig.clone()
                })
            }
            _ => {
                unreachable!("build_rust_fn: extern fn {} has ty {:?}, \
                              expected a fn item type",
                              tcx.map.path_to_string(id), t);
            }
        };

        debug!("build_rust_fn: path={} id={} ty={:?}",
               ccx.tcx().map.path_to_string(id),
               id, rust_fn_ty);

        let llfn = declare::define_internal_fn(ccx, &ps, rust_fn_ty);
        attributes::from_fn_attrs(ccx, attrs, llfn);
        base::trans_fn(ccx, decl, body, llfn, param_substs, id, attrs);
        llfn
    }

    unsafe fn build_wrap_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                      llrustfn: ValueRef,
                                      llwrapfn: ValueRef,
                                      fn_sig: &ty::FnSig<'tcx>,
                                      fn_ty: &FnType,
                                      t: Ty<'tcx>) {
        let _icx = push_ctxt(
            "foreign::trans_rust_fn_with_foreign_abi::build_wrap_fn");

        debug!("build_wrap_fn(llrustfn={:?}, llwrapfn={:?}, t={:?})",
               Value(llrustfn),
               Value(llwrapfn),
               t);

        // Avoid all the Rust generation stuff and just generate raw
        // LLVM here.
        //
        // We want to generate code like this:
        //
        //     S foo(T i) {
        //         S r;
        //         foo0(&r, NULL, i);
        //         return r;
        //     }

        if llvm::LLVMCountBasicBlocks(llwrapfn) != 0 {
            ccx.sess().bug("wrapping a function inside non-empty wrapper, most likely cause is \
                           multiple functions being wrapped");
        }

        let ptr = "the block\0".as_ptr();
        let the_block = llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), llwrapfn,
                                                            ptr as *const _);

        let builder = ccx.builder();
        builder.position_at_end(the_block);

        // Array for the arguments we will pass to the rust function.
        let mut llrust_args = Vec::new();
        let mut next_foreign_arg_counter: c_uint = 0;
        let mut next_foreign_arg = |pad: bool| -> c_uint {
            next_foreign_arg_counter += if pad {
                2
            } else {
                1
            };
            next_foreign_arg_counter - 1
        };

        // If there is an out pointer on the foreign function
        let foreign_outptr = {
            if fn_ty.ret.is_indirect() {
                Some(get_param(llwrapfn, next_foreign_arg(false)))
            } else {
                None
            }
        };

        let rustfn_ty = Type::from_ref(llvm::LLVMTypeOf(llrustfn)).element_type();
        let mut rust_param_tys = rustfn_ty.func_params().into_iter();
        // Push Rust return pointer, using null if it will be unused.
        let rust_uses_outptr = match fn_sig.output {
            ty::FnConverging(ret_ty) => type_of::return_uses_outptr(ccx, ret_ty),
            ty::FnDiverging => false
        };
        let return_alloca: Option<ValueRef>;
        let llrust_ret_ty = if rust_uses_outptr {
            rust_param_tys.next().expect("Missing return type!").element_type()
        } else {
            rustfn_ty.return_type()
        };
        if rust_uses_outptr {
            // Rust expects to use an outpointer. If the foreign fn
            // also uses an outpointer, we can reuse it, but the types
            // may vary, so cast first to the Rust type. If the
            // foreign fn does NOT use an outpointer, we will have to
            // alloca some scratch space on the stack.
            match foreign_outptr {
                Some(llforeign_outptr) => {
                    debug!("out pointer, foreign={:?}",
                           Value(llforeign_outptr));
                    let llrust_retptr =
                        builder.bitcast(llforeign_outptr, llrust_ret_ty.ptr_to());
                    debug!("out pointer, foreign={:?} (casted)",
                           Value(llrust_retptr));
                    llrust_args.push(llrust_retptr);
                    return_alloca = None;
                }

                None => {
                    let slot = builder.alloca(llrust_ret_ty, "return_alloca");
                    debug!("out pointer, \
                            allocad={:?}, \
                            llrust_ret_ty={:?}, \
                            return_ty={:?}",
                           Value(slot),
                           llrust_ret_ty,
                           fn_sig.output);
                    llrust_args.push(slot);
                    return_alloca = Some(slot);
                }
            }
        } else {
            // Rust does not expect an outpointer. If the foreign fn
            // does use an outpointer, then we will do a store of the
            // value that the Rust fn returns.
            return_alloca = None;
        };

        // Build up the arguments to the call to the rust function.
        // Careful to adapt for cases where the native convention uses
        // a pointer and Rust does not or vice versa.
        let mut tys = fn_ty.args.iter().zip(rust_param_tys);
        for i in 0..fn_sig.inputs.len() {
            let rust_ty = fn_sig.inputs[i];
            let rust_indirect = type_of::arg_is_indirect(ccx, rust_ty);
            let (llforeign_arg_ty, llty) = tys.next().expect("Not enough parameter types!");
            let llrust_ty = if rust_indirect {
                llty.element_type()
            } else {
                llty
            };
            let foreign_indirect = llforeign_arg_ty.is_indirect();

            if llforeign_arg_ty.is_ignore() {
                debug!("skipping ignored arg #{}", i);
                llrust_args.push(C_undef(llrust_ty));
                continue;
            }

            // skip padding
            let foreign_index = next_foreign_arg(llforeign_arg_ty.pad.is_some());
            let mut llforeign_arg = get_param(llwrapfn, foreign_index);

            if type_is_fat_ptr(ccx.tcx(), rust_ty) {
                // Fat pointers are one pointer and one integer or pointer.
                let a = llforeign_arg_ty;
                let (b, _) = tys.next().expect("Not enough parameter types!");
                assert_eq!((a.cast, b.cast), (None, None));
                assert!(!a.is_indirect() && !b.is_indirect());

                llrust_args.push(llforeign_arg);
                let foreign_index = next_foreign_arg(llforeign_arg_ty.pad.is_some());
                llrust_args.push(get_param(llwrapfn, foreign_index));
                continue;
            }

            debug!("llforeign_arg {}{}: {:?}", "#",
                   i, Value(llforeign_arg));
            debug!("rust_indirect = {}, foreign_indirect = {}",
                   rust_indirect, foreign_indirect);

            // Ensure that the foreign argument is indirect (by
            // pointer).  It makes adapting types easier, since we can
            // always just bitcast pointers.
            if !foreign_indirect {
                llforeign_arg = if rust_ty.is_bool() {
                    let lltemp = builder.alloca(Type::bool(ccx), "");
                    builder.store(builder.zext(llforeign_arg, Type::bool(ccx)), lltemp);
                    lltemp
                } else {
                    let lltemp = builder.alloca(val_ty(llforeign_arg), "");
                    builder.store(llforeign_arg, lltemp);
                    lltemp
                }
            }

            // If the types in the ABI and the Rust types don't match,
            // bitcast the llforeign_arg pointer so it matches the types
            // Rust expects.
            if llforeign_arg_ty.cast.is_some() && !type_is_fat_ptr(ccx.tcx(), rust_ty){
                assert!(!foreign_indirect);
                llforeign_arg = builder.bitcast(llforeign_arg, llrust_ty.ptr_to());
            }

            let llrust_arg = if rust_indirect || type_is_fat_ptr(ccx.tcx(), rust_ty) {
                llforeign_arg
            } else {
                if rust_ty.is_bool() {
                    let tmp = builder.load_range_assert(llforeign_arg, 0, 2, llvm::False);
                    builder.trunc(tmp, Type::i1(ccx))
                } else if type_of::type_of(ccx, rust_ty).is_aggregate() {
                    // We want to pass small aggregates as immediate values, but using an aggregate
                    // LLVM type for this leads to bad optimizations, so its arg type is an
                    // appropriately sized integer and we have to convert it
                    let tmp = builder.bitcast(llforeign_arg,
                                              type_of::arg_type_of(ccx, rust_ty).ptr_to());
                    let load = builder.load(tmp);
                    llvm::LLVMSetAlignment(load, type_of::align_of(ccx, rust_ty));
                    load
                } else {
                    builder.load(llforeign_arg)
                }
            };

            debug!("llrust_arg {}{}: {:?}", "#",
                   i, Value(llrust_arg));
            llrust_args.push(llrust_arg);
        }

        // Perform the call itself
        debug!("calling llrustfn = {:?}, t = {:?}",
               Value(llrustfn), t);
        let attributes = attributes::from_fn_type(ccx, t);
        let llrust_ret_val = builder.call(llrustfn, &llrust_args,
                                          None, Some(attributes));

        // Get the return value where the foreign fn expects it.
        let llforeign_ret_ty = fn_ty.ret.cast.unwrap_or(fn_ty.ret.original_ty);
        match foreign_outptr {
            None if llforeign_ret_ty == Type::void(ccx) => {
                // Function returns `()` or `bot`, which in Rust is the LLVM
                // type "{}" but in foreign ABIs is "Void".
                builder.ret_void();
            }

            None if rust_uses_outptr => {
                // Rust uses an outpointer, but the foreign ABI does not. Load.
                let llrust_outptr = return_alloca.unwrap();
                let llforeign_outptr_casted =
                    builder.bitcast(llrust_outptr, llforeign_ret_ty.ptr_to());
                let llforeign_retval = builder.load(llforeign_outptr_casted);
                builder.ret(llforeign_retval);
            }

            None if llforeign_ret_ty != llrust_ret_ty => {
                // Neither ABI uses an outpointer, but the types don't
                // quite match. Must cast. Probably we should try and
                // examine the types and use a concrete llvm cast, but
                // right now we just use a temp memory location and
                // bitcast the pointer, which is the same thing the
                // old wrappers used to do.
                let lltemp = builder.alloca(llforeign_ret_ty, "");
                let lltemp_casted = builder.bitcast(lltemp, llrust_ret_ty.ptr_to());
                builder.store(llrust_ret_val, lltemp_casted);
                let llforeign_retval = builder.load(lltemp);
                builder.ret(llforeign_retval);
            }

            None => {
                // Neither ABI uses an outpointer, and the types
                // match. Easy peasy.
                builder.ret(llrust_ret_val);
            }

            Some(llforeign_outptr) if !rust_uses_outptr => {
                // Foreign ABI requires an out pointer, but Rust doesn't.
                // Store Rust return value.
                let llforeign_outptr_casted =
                    builder.bitcast(llforeign_outptr, llrust_ret_ty.ptr_to());
                builder.store(llrust_ret_val, llforeign_outptr_casted);
                builder.ret_void();
            }

            Some(_) => {
                // Both ABIs use outpointers. Easy peasy.
                builder.ret_void();
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// General ABI Support
//
// This code is kind of a confused mess and needs to be reworked given
// the massive simplifications that have occurred.

pub fn link_name(name: ast::Name, attrs: &[ast::Attribute]) -> InternedString {
    match attr::first_attr_value_str_by_name(attrs, "link_name") {
        Some(ln) => ln.clone(),
        None => match weak_lang_items::link_name(attrs) {
            Some(name) => name,
            None => name.as_str(),
        }
    }
}
