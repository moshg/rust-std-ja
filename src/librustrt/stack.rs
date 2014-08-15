// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rust stack-limit management
//!
//! Currently Rust uses a segmented-stack-like scheme in order to detect stack
//! overflow for rust tasks. In this scheme, the prologue of all functions are
//! preceded with a check to see whether the current stack limits are being
//! exceeded.
//!
//! This module provides the functionality necessary in order to manage these
//! stack limits (which are stored in platform-specific locations). The
//! functions here are used at the borders of the task lifetime in order to
//! manage these limits.
//!
//! This function is an unstable module because this scheme for stack overflow
//! detection is not guaranteed to continue in the future. Usage of this module
//! is discouraged unless absolutely necessary.

// iOS related notes
//
// It is possible to implement it using idea from
// http://www.opensource.apple.com/source/Libc/Libc-825.40.1/pthreads/pthread_machdep.h
//
// In short: _pthread_{get,set}_specific_direct allows extremely fast
// access, exactly what is required for segmented stack
// There is a pool of reserved slots for Apple internal use (0..119)
// First dynamic allocated pthread key starts with 257 (on iOS7)
// So using slot 149 should be pretty safe ASSUMING space is reserved
// for every key < first dynamic key
//
// There is also an opportunity to steal keys reserved for Garbage Collection
// ranges 80..89 and 110..119, especially considering the fact Garbage Collection
// never supposed to work on iOS. But as everybody knows it - there is a chance
// that those slots will be re-used, like it happened with key 95 (moved from
// JavaScriptCore to CoreText)
//
// Unfortunately Apple rejected patch to LLVM which generated
// corresponding prolog, decision was taken to disable segmented
// stack support on iOS.

pub static RED_ZONE: uint = 20 * 1024;

/// This function is invoked from rust's current __morestack function. Segmented
/// stacks are currently not enabled as segmented stacks, but rather one giant
/// stack segment. This means that whenever we run out of stack, we want to
/// truly consider it to be stack overflow rather than allocating a new stack.
#[cfg(not(test))] // in testing, use the original libstd's version
#[lang = "stack_exhausted"]
extern fn stack_exhausted() {
    use core::prelude::*;
    use alloc::boxed::Box;
    use local::Local;
    use task::Task;
    use core::intrinsics;

    unsafe {
        // We're calling this function because the stack just ran out. We need
        // to call some other rust functions, but if we invoke the functions
        // right now it'll just trigger this handler being called again. In
        // order to alleviate this, we move the stack limit to be inside of the
        // red zone that was allocated for exactly this reason.
        let limit = get_sp_limit();
        record_sp_limit(limit - RED_ZONE / 2);

        // This probably isn't the best course of action. Ideally one would want
        // to unwind the stack here instead of just aborting the entire process.
        // This is a tricky problem, however. There's a few things which need to
        // be considered:
        //
        //  1. We're here because of a stack overflow, yet unwinding will run
        //     destructors and hence arbitrary code. What if that code overflows
        //     the stack? One possibility is to use the above allocation of an
        //     extra 10k to hope that we don't hit the limit, and if we do then
        //     abort the whole program. Not the best, but kind of hard to deal
        //     with unless we want to switch stacks.
        //
        //  2. LLVM will optimize functions based on whether they can unwind or
        //     not. It will flag functions with 'nounwind' if it believes that
        //     the function cannot trigger unwinding, but if we do unwind on
        //     stack overflow then it means that we could unwind in any function
        //     anywhere. We would have to make sure that LLVM only places the
        //     nounwind flag on functions which don't call any other functions.
        //
        //  3. The function that overflowed may have owned arguments. These
        //     arguments need to have their destructors run, but we haven't even
        //     begun executing the function yet, so unwinding will not run the
        //     any landing pads for these functions. If this is ignored, then
        //     the arguments will just be leaked.
        //
        // Exactly what to do here is a very delicate topic, and is possibly
        // still up in the air for what exactly to do. Some relevant issues:
        //
        //  #3555 - out-of-stack failure leaks arguments
        //  #3695 - should there be a stack limit?
        //  #9855 - possible strategies which could be taken
        //  #9854 - unwinding on windows through __morestack has never worked
        //  #2361 - possible implementation of not using landing pads

        let task: Option<Box<Task>> = Local::try_take();
        let name = match task {
            Some(ref task) => {
                task.name.as_ref().map(|n| n.as_slice())
            }
            None => None
        };
        let name = name.unwrap_or("<unknown>");

        // See the message below for why this is not emitted to the
        // task's logger. This has the additional conundrum of the
        // logger may not be initialized just yet, meaning that an FFI
        // call would happen to initialized it (calling out to libuv),
        // and the FFI call needs 2MB of stack when we just ran out.
        rterrln!("task '{}' has overflowed its stack", name);

        intrinsics::abort();
    }
}

// Windows maintains a record of upper and lower stack bounds in the Thread Information
// Block (TIB), and some syscalls do check that addresses which are supposed to be in
// the stack, indeed lie between these two values.
// (See https://github.com/rust-lang/rust/issues/3445#issuecomment-26114839)
//
// When using Rust-managed stacks (libgreen), we must maintain these values accordingly.
// For OS-managed stacks (libnative), we let the OS manage them for us.
//
// On all other platforms both variants behave identically.

#[inline(always)]
pub unsafe fn record_os_managed_stack_bounds(stack_lo: uint, _stack_hi: uint) {
    record_sp_limit(stack_lo + RED_ZONE);
}

#[inline(always)]
pub unsafe fn record_rust_managed_stack_bounds(stack_lo: uint, stack_hi: uint) {
    // When the old runtime had segmented stacks, it used a calculation that was
    // "limit + RED_ZONE + FUDGE". The red zone was for things like dynamic
    // symbol resolution, llvm function calls, etc. In theory this red zone
    // value is 0, but it matters far less when we have gigantic stacks because
    // we don't need to be so exact about our stack budget. The "fudge factor"
    // was because LLVM doesn't emit a stack check for functions < 256 bytes in
    // size. Again though, we have giant stacks, so we round all these
    // calculations up to the nice round number of 20k.
    record_sp_limit(stack_lo + RED_ZONE);

    return target_record_stack_bounds(stack_lo, stack_hi);

    #[cfg(not(windows))] #[inline(always)]
    unsafe fn target_record_stack_bounds(_stack_lo: uint, _stack_hi: uint) {}

    #[cfg(windows, target_arch = "x86")] #[inline(always)]
    unsafe fn target_record_stack_bounds(stack_lo: uint, stack_hi: uint) {
        // stack range is at TIB: %fs:0x04 (top) and %fs:0x08 (bottom)
        asm!("mov $0, %fs:0x04" :: "r"(stack_hi) :: "volatile");
        asm!("mov $0, %fs:0x08" :: "r"(stack_lo) :: "volatile");
    }
    #[cfg(windows, target_arch = "x86_64")] #[inline(always)]
    unsafe fn target_record_stack_bounds(stack_lo: uint, stack_hi: uint) {
        // stack range is at TIB: %gs:0x08 (top) and %gs:0x10 (bottom)
        asm!("mov $0, %gs:0x08" :: "r"(stack_hi) :: "volatile");
        asm!("mov $0, %gs:0x10" :: "r"(stack_lo) :: "volatile");
    }
}

/// Records the current limit of the stack as specified by `end`.
///
/// This is stored in an OS-dependent location, likely inside of the thread
/// local storage. The location that the limit is stored is a pre-ordained
/// location because it's where LLVM has emitted code to check.
///
/// Note that this cannot be called under normal circumstances. This function is
/// changing the stack limit, so upon returning any further function calls will
/// possibly be triggering the morestack logic if you're not careful.
///
/// Also note that this and all of the inside functions are all flagged as
/// "inline(always)" because they're messing around with the stack limits.  This
/// would be unfortunate for the functions themselves to trigger a morestack
/// invocation (if they were an actual function call).
#[inline(always)]
pub unsafe fn record_sp_limit(limit: uint) {
    return target_record_sp_limit(limit);

    // x86-64
    #[cfg(target_arch = "x86_64", target_os = "macos")]
    #[cfg(target_arch = "x86_64", target_os = "ios")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $$0x60+90*8, %rsi
              movq $0, %gs:(%rsi)" :: "r"(limit) : "rsi" : "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "linux")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $0, %fs:112" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "windows")] #[inline(always)]
    #[cfg(stage0, target_arch = "x86_64", target_os = "win32")] // NOTE: Remove after snapshot
    unsafe fn target_record_sp_limit(limit: uint) {
        // see: http://en.wikipedia.org/wiki/Win32_Thread_Information_Block
        // store this inside of the "arbitrary data slot", but double the size
        // because this is 64 bit instead of 32 bit
        asm!("movq $0, %gs:0x28" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $0, %fs:24" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86_64", target_os = "dragonfly")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movq $0, %fs:32" :: "r"(limit) :: "volatile")
    }

    // x86
    #[cfg(target_arch = "x86", target_os = "macos")]
    #[cfg(target_arch = "x86", target_os = "ios")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movl $$0x48+90*4, %eax
              movl $0, %gs:(%eax)" :: "r"(limit) : "eax" : "volatile")
    }
    #[cfg(target_arch = "x86", target_os = "linux")]
    #[cfg(target_arch = "x86", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        asm!("movl $0, %gs:48" :: "r"(limit) :: "volatile")
    }
    #[cfg(target_arch = "x86", target_os = "windows")] #[inline(always)]
    #[cfg(stage0, target_arch = "x86", target_os = "win32")] // NOTE: Remove after snapshot
    unsafe fn target_record_sp_limit(limit: uint) {
        // see: http://en.wikipedia.org/wiki/Win32_Thread_Information_Block
        // store this inside of the "arbitrary data slot"
        asm!("movl $0, %fs:0x14" :: "r"(limit) :: "volatile")
    }

    // mips, arm - Some brave soul can port these to inline asm, but it's over
    //             my head personally
    #[cfg(target_arch = "mips")]
    #[cfg(target_arch = "mipsel")]
    #[cfg(target_arch = "arm", not(target_os = "ios"))] #[inline(always)]
    unsafe fn target_record_sp_limit(limit: uint) {
        use libc::c_void;
        return record_sp_limit(limit as *const c_void);
        extern {
            fn record_sp_limit(limit: *const c_void);
        }
    }

    // iOS segmented stack is disabled for now, see related notes
    #[cfg(target_arch = "arm", target_os = "ios")] #[inline(always)]
    unsafe fn target_record_sp_limit(_: uint) {
    }
}

/// The counterpart of the function above, this function will fetch the current
/// stack limit stored in TLS.
///
/// Note that all of these functions are meant to be exact counterparts of their
/// brethren above, except that the operands are reversed.
///
/// As with the setter, this function does not have a __morestack header and can
/// therefore be called in a "we're out of stack" situation.
#[inline(always)]
pub unsafe fn get_sp_limit() -> uint {
    return target_get_sp_limit();

    // x86-64
    #[cfg(target_arch = "x86_64", target_os = "macos")]
    #[cfg(target_arch = "x86_64", target_os = "ios")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movq $$0x60+90*8, %rsi
              movq %gs:(%rsi), $0" : "=r"(limit) :: "rsi" : "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "linux")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movq %fs:112, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "windows")] #[inline(always)]
    #[cfg(stage0, target_arch = "x86_64", target_os = "win32")] // NOTE: Remove after snapshot
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movq %gs:0x28, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movq %fs:24, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86_64", target_os = "dragonfly")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movq %fs:32, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }


    // x86
    #[cfg(target_arch = "x86", target_os = "macos")]
    #[cfg(target_arch = "x86", target_os = "ios")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movl $$0x48+90*4, %eax
              movl %gs:(%eax), $0" : "=r"(limit) :: "eax" : "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86", target_os = "linux")]
    #[cfg(target_arch = "x86", target_os = "freebsd")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movl %gs:48, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }
    #[cfg(target_arch = "x86", target_os = "windows")] #[inline(always)]
    #[cfg(stage0, target_arch = "x86", target_os = "win32")] // NOTE: Remove after snapshot
    unsafe fn target_get_sp_limit() -> uint {
        let limit;
        asm!("movl %fs:0x14, $0" : "=r"(limit) ::: "volatile");
        return limit;
    }

    // mips, arm - Some brave soul can port these to inline asm, but it's over
    //             my head personally
    #[cfg(target_arch = "mips")]
    #[cfg(target_arch = "mipsel")]
    #[cfg(target_arch = "arm", not(target_os = "ios"))] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        use libc::c_void;
        return get_sp_limit() as uint;
        extern {
            fn get_sp_limit() -> *const c_void;
        }
    }

    // iOS doesn't support segmented stacks yet. This function might
    // be called by runtime though so it is unsafe to mark it as
    // unreachable, let's return a fixed constant.
    #[cfg(target_arch = "arm", target_os = "ios")] #[inline(always)]
    unsafe fn target_get_sp_limit() -> uint {
        1024
    }
}
