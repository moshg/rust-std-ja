// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// DO NOT EDIT: autogenerated by etc/platform-intrinsics/generator.py
// ignore-tidy-linelength

#![allow(unused_imports)]

use {Intrinsic, Type};
use IntrinsicDef::Named;

// The default inlining settings trigger a pathological behaviour in
// LLVM, which causes makes compilation very slow. See #28273.
#[inline(never)]
pub fn find(name: &str) -> Option<Intrinsic> {
    if !name.starts_with("powerpc") { return None }
    Some(match &name["powerpc".len()..] {
        "_vec_perm" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I32x4, &::I32x4, &::I8x16]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vperm")
        },
        "_vec_mradds" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I16x8, &::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vmhraddshs")
        },
        "_vec_cmpb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::F32x4, &::F32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vcmpbfp")
        },
        "_vec_cmpeqb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vcmpequb")
        },
        "_vec_cmpeqh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vcmpequh")
        },
        "_vec_cmpeqw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vcmpequw")
        },
        "_vec_cmpgtub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vcmpgtub")
        },
        "_vec_cmpgtuh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vcmpgtuh")
        },
        "_vec_cmpgtuw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vcmpgtuw")
        },
        "_vec_cmpgtsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vcmpgtsb")
        },
        "_vec_cmpgtsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vcmpgtsh")
        },
        "_vec_cmpgtsw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vcmpgtsw")
        },
        "_vec_maxsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vmaxsb")
        },
        "_vec_maxub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vmaxub")
        },
        "_vec_maxsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vmaxsh")
        },
        "_vec_maxuh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vmaxuh")
        },
        "_vec_maxsw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmaxsw")
        },
        "_vec_maxuw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmaxuw")
        },
        "_vec_minsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vminsb")
        },
        "_vec_minub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vminub")
        },
        "_vec_minsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vminsh")
        },
        "_vec_minuh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vminuh")
        },
        "_vec_minsw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vminsw")
        },
        "_vec_minuw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vminuw")
        },
        "_vec_subsbs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vsubsbs")
        },
        "_vec_sububs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vsububs")
        },
        "_vec_subshs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vsubshs")
        },
        "_vec_subuhs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vsubuhs")
        },
        "_vec_subsws" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vsubsws")
        },
        "_vec_subuws" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vsubuws")
        },
        "_vec_subc" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vsubcuw")
        },
        "_vec_addsbs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vaddsbs")
        },
        "_vec_addubs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vaddubs")
        },
        "_vec_addshs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vaddshs")
        },
        "_vec_adduhs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vadduhs")
        },
        "_vec_addsws" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vaddsws")
        },
        "_vec_adduws" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vadduws")
        },
        "_vec_addc" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vaddcuw")
        },
        "_vec_mulesb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vmulesb")
        },
        "_vec_muleub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vmuleub")
        },
        "_vec_mulesh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmulesh")
        },
        "_vec_muleuh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmuleuh")
        },
        "_vec_mulosb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vmulosb")
        },
        "_vec_muloub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vmuloub")
        },
        "_vec_mulosh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmulosh")
        },
        "_vec_mulouh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmulouh")
        },
        "_vec_avgsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I8x16]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vavgsb")
        },
        "_vec_avgub" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U8x16]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vavgub")
        },
        "_vec_avgsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vavgsh")
        },
        "_vec_avguh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vavguh")
        },
        "_vec_avgsw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vavgsw")
        },
        "_vec_avguw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vavguw")
        },
        "_vec_packssh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::I8x16,
            definition: Named("llvm.ppc.altivec.vpkshss")
        },
        "_vec_packsuh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U16x8, &::U16x8]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vpkuhus")
        },
        "_vec_packssw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vpkswss")
        },
        "_vec_packsuw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U32x4, &::U32x4]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vpkuwus")
        },
        "_vec_packsush" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I16x8]; &INPUTS },
            output: &::U8x16,
            definition: Named("llvm.ppc.altivec.vpkshus")
        },
        "_vec_packsusw" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::U16x8,
            definition: Named("llvm.ppc.altivec.vpkswus")
        },
        "_vec_packpx" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vpkpx")
        },
        "_vec_unpacklsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 1] = [&::I8x16]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vupklsb")
        },
        "_vec_unpacklsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 1] = [&::I16x8]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vupklsh")
        },
        "_vec_unpackhsb" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 1] = [&::I8x16]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vupkhsb")
        },
        "_vec_unpackhsh" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 1] = [&::I16x8]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vupkhsh")
        },
        "_vec_madds" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I16x8, &::I16x8, &::I16x8]; &INPUTS },
            output: &::I16x8,
            definition: Named("llvm.ppc.altivec.vmhaddshs")
        },
        "_vec_msumubm" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::U8x16, &::U8x16, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmsumubm")
        },
        "_vec_msumuhm" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::U16x8, &::U16x8, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmsumuhm")
        },
        "_vec_msummbm" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I8x16, &::U8x16, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmsummbm")
        },
        "_vec_msumshm" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I16x8, &::I16x8, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmsumshm")
        },
        "_vec_msumshs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::I16x8, &::I16x8, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vmsumshs")
        },
        "_vec_msumuhs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::U16x8, &::U16x8, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vmsumuhs")
        },
        "_vec_sum2s" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vsum2sws")
        },
        "_vec_sum4sbs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I8x16, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vsum4sbs")
        },
        "_vec_sum4ubs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::U8x16, &::U32x4]; &INPUTS },
            output: &::U32x4,
            definition: Named("llvm.ppc.altivec.vsum4ubs")
        },
        "_vec_sum4shs" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I16x8, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vsum4shs")
        },
        "_vec_sums" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 2] = [&::I32x4, &::I32x4]; &INPUTS },
            output: &::I32x4,
            definition: Named("llvm.ppc.altivec.vsumsws")
        },
        "_vec_madd" => Intrinsic {
            inputs: { static INPUTS: [&'static Type; 3] = [&::F32x4, &::F32x4, &::F32x4]; &INPUTS },
            output: &::F32x4,
            definition: Named("llvm.ppc.altivec.vmaddfp")
        },
        _ => return None,
    })
}
