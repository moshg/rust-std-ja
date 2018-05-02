// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use boxed::FnBox;
use cmp;
use ffi::CStr;
use io;
use libc;
use mem;
use ptr;
use sys::os;
use time::Duration;

use sys_common::thread::*;

#[cfg(not(target_os = "l4re"))]
pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;
#[cfg(target_os = "l4re")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 1024 * 1024;

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

// The pthread_attr_setstacksize symbol doesn't exist in the emscripten libc,
// so we have to not link to it to satisfy emcc's ERROR_ON_UNDEFINED_SYMBOLS.
#[cfg(not(target_os = "emscripten"))]
unsafe fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                    stack_size: libc::size_t) -> libc::c_int {
    libc::pthread_attr_setstacksize(attr, stack_size)
}

#[cfg(target_os = "emscripten")]
unsafe fn pthread_attr_setstacksize(_attr: *mut libc::pthread_attr_t,
                                    _stack_size: libc::size_t) -> libc::c_int {
    panic!()
}

impl Thread {
    pub unsafe fn new<'a>(stack: usize, p: Box<FnBox() + 'a>)
                          -> io::Result<Thread> {
        let p = box p;
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);

        let stack_size = cmp::max(stack, min_stack_size(&attr));

        match pthread_attr_setstacksize(&mut attr,
                                        stack_size) {
            0 => {}
            n => {
                assert_eq!(n, libc::EINVAL);
                // EINVAL means |stack_size| is either too small or not a
                // multiple of the system page size.  Because it's definitely
                // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                // Round up to the nearest page and try again.
                let page_size = os::page_size();
                let stack_size = (stack_size + page_size - 1) &
                                 (-(page_size as isize - 1) as usize - 1);
                assert_eq!(libc::pthread_attr_setstacksize(&mut attr,
                                                           stack_size), 0);
            }
        };

        let ret = libc::pthread_create(&mut native, &attr, thread_start,
                                       &*p as *const _ as *mut _);
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);

        return if ret != 0 {
            Err(io::Error::from_raw_os_error(ret))
        } else {
            mem::forget(p); // ownership passed to pthread_create
            Ok(Thread { id: native })
        };

        extern fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe { start_thread(main as *mut u8); }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { libc::sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    #[cfg(any(target_os = "linux",
              target_os = "android"))]
    pub fn set_name(name: &CStr) {
        const PR_SET_NAME: libc::c_int = 15;
        // pthread wrapper only appeared in glibc 2.12, so we use syscall
        // directly.
        unsafe {
            libc::prctl(PR_SET_NAME, name.as_ptr() as libc::c_ulong, 0, 0, 0);
        }
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "openbsd"))]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_setname_np(name.as_ptr());
        }
    }

    #[cfg(target_os = "netbsd")]
    pub fn set_name(name: &CStr) {
        use ffi::CString;
        let cname = CString::new(&b"%s"[..]).unwrap();
        unsafe {
            libc::pthread_setname_np(libc::pthread_self(), cname.as_ptr(),
                                     name.as_ptr() as *mut libc::c_void);
        }
    }
    #[cfg(any(target_env = "newlib",
              target_os = "solaris",
              target_os = "haiku",
              target_os = "l4re",
              target_os = "emscripten"))]
    pub fn set_name(_name: &CStr) {
        // Newlib, Illumos, Haiku, and Emscripten have no way to set a thread name.
    }
    #[cfg(target_os = "fuchsia")]
    pub fn set_name(_name: &CStr) {
        // FIXME: determine whether Fuchsia has a way to set a thread name.
    }

    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as _;

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        unsafe {
            while secs > 0 || nsecs > 0 {
                let mut ts = libc::timespec {
                    tv_sec: cmp::min(libc::time_t::max_value() as u64, secs) as libc::time_t,
                    tv_nsec: nsecs,
                };
                secs -= ts.tv_sec as u64;
                if libc::nanosleep(&ts, &mut ts) == -1 {
                    assert_eq!(os::errno(), libc::EINTR);
                    secs += ts.tv_sec as u64;
                    nsecs = ts.tv_nsec;
                } else {
                    nsecs = 0;
                }
            }
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = libc::pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            assert!(ret == 0,
                    "failed to join thread: {}", io::Error::from_raw_os_error(ret));
        }
    }

    pub fn id(&self) -> libc::pthread_t { self.id }

    pub fn into_id(self) -> libc::pthread_t {
        let id = self.id;
        mem::forget(self);
        id
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        let ret = unsafe { libc::pthread_detach(self.id) };
        debug_assert_eq!(ret, 0);
    }
}

#[cfg(all(not(all(target_os = "linux", not(target_env = "musl"))),
          not(target_os = "freebsd"),
          not(target_os = "macos"),
          not(target_os = "bitrig"),
          not(all(target_os = "netbsd", not(target_vendor = "rumprun"))),
          not(target_os = "openbsd"),
          not(target_os = "solaris")))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use ops::Range;
    pub type Guard = Range<usize>;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
    pub unsafe fn deinit() {}
}


#[cfg(any(all(target_os = "linux", not(target_env = "musl")),
          target_os = "freebsd",
          target_os = "macos",
          target_os = "bitrig",
          all(target_os = "netbsd", not(target_vendor = "rumprun")),
          target_os = "openbsd",
          target_os = "solaris"))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use libc;
    use libc::{mmap, mprotect};
    use libc::{PROT_NONE, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANON, MAP_FAILED, MAP_FIXED};
    use ops::Range;
    use sys::os;

    // This is initialized in init() and only read from after
    static mut PAGE_SIZE: usize = 0;

    pub type Guard = Range<usize>;

    #[cfg(target_os = "solaris")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = ::mem::zeroed();
        assert_eq!(libc::stack_getbounds(&mut current_stack), 0);
        Some(current_stack.ss_sp)
    }

    #[cfg(target_os = "macos")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let stackaddr = libc::pthread_get_stackaddr_np(libc::pthread_self()) as usize -
             libc::pthread_get_stacksize_np(libc::pthread_self());
        Some(stackaddr as *mut libc::c_void)
    }

    #[cfg(any(target_os = "openbsd", target_os = "bitrig"))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = ::mem::zeroed();
        assert_eq!(libc::pthread_stackseg_np(libc::pthread_self(),
                                             &mut current_stack), 0);

        let extra = if cfg!(target_os = "bitrig") {3} else {1} * PAGE_SIZE;
        let stackaddr = if libc::pthread_main_np() == 1 {
            // main thread
            current_stack.ss_sp as usize - current_stack.ss_size + extra
        } else {
            // new thread
            current_stack.ss_sp as usize - current_stack.ss_size
        };
        Some(stackaddr as *mut libc::c_void)
    }

    #[cfg(any(target_os = "android", target_os = "freebsd",
              target_os = "linux", target_os = "netbsd", target_os = "l4re"))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = ::mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
            let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
            let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut stackaddr = ::ptr::null_mut();
            let mut stacksize = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackaddr,
                                                   &mut stacksize), 0);
            ret = Some(stackaddr);
        }
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        ret
    }

    // Precondition: PAGE_SIZE is initialized.
    unsafe fn get_stack_start_aligned() -> Option<*mut libc::c_void> {
        assert!(PAGE_SIZE != 0);
        let stackaddr = get_stack_start()?;

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = (stackaddr as usize) % PAGE_SIZE;
        Some(if remainder == 0 {
            stackaddr
        } else {
            ((stackaddr as usize) + PAGE_SIZE - remainder) as *mut libc::c_void
        })
    }

    pub unsafe fn init() -> Option<Guard> {
        PAGE_SIZE = os::page_size();

        let stackaddr = get_stack_start_aligned()?;

        if cfg!(target_os = "linux") {
            // Linux doesn't allocate the whole stack right away, and
            // the kernel has its own stack-guard mechanism to fault
            // when growing too close to an existing mapping.  If we map
            // our own guard, then the kernel starts enforcing a rather
            // large gap above that, rendering much of the possible
            // stack space useless.  See #43052.
            //
            // Instead, we'll just note where we expect rlimit to start
            // faulting, so our handler can report "stack overflow", and
            // trust that the kernel's own stack guard will work.
            let stackaddr = stackaddr as usize;
            Some(stackaddr - PAGE_SIZE..stackaddr)
        } else {
            // Reallocate the last page of the stack.
            // This ensures SIGBUS will be raised on
            // stack overflow.
            // Systems which enforce strict PAX MPROTECT do not allow
            // to mprotect() a mapping with less restrictive permissions
            // than the initial mmap() used, so we mmap() here with
            // read/write permissions and only then mprotect() it to
            // no permissions at all. See issue #50313.
            let result = mmap(stackaddr, PAGE_SIZE, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
            if result != stackaddr || result == MAP_FAILED {
                panic!("failed to allocate a guard page");
            }

            let result = mprotect(stackaddr, PAGE_SIZE, PROT_NONE);
            if result != 0 {
                panic!("failed to protect the guard page");
            }

            let guardaddr = stackaddr as usize;
            let offset = if cfg!(target_os = "freebsd") {
                2
            } else {
                1
            };

            Some(guardaddr..guardaddr + offset * PAGE_SIZE)
        }
    }

    pub unsafe fn deinit() {
        if !cfg!(target_os = "linux") {
            if let Some(stackaddr) = get_stack_start_aligned() {
                // Remove the protection on the guard page.
                // FIXME: we cannot unmap the page, because when we mmap()
                // above it may be already mapped by the OS, which we can't
                // detect from mmap()'s return value. If we unmap this page,
                // it will lead to failure growing stack size on platforms like
                // macOS. Instead, just restore the page to a writable state.
                // This ain't Linux, so we probably don't need to care about
                // execstack.
                let result = mprotect(stackaddr, PAGE_SIZE, PROT_READ | PROT_WRITE);

                if result != 0 {
                    panic!("unable to reset the guard page");
                }
            }
        }
    }

    #[cfg(any(target_os = "macos",
              target_os = "bitrig",
              target_os = "openbsd",
              target_os = "solaris"))]
    pub unsafe fn current() -> Option<Guard> {
        let stackaddr = get_stack_start()? as usize;
        Some(stackaddr - PAGE_SIZE..stackaddr)
    }

    #[cfg(any(target_os = "android", target_os = "freebsd",
              target_os = "linux", target_os = "netbsd", target_os = "l4re"))]
    pub unsafe fn current() -> Option<Guard> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = ::mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
            let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
            let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut guardsize = 0;
            assert_eq!(libc::pthread_attr_getguardsize(&attr, &mut guardsize), 0);
            if guardsize == 0 {
                panic!("there is no guard page");
            }
            let mut stackaddr = ::ptr::null_mut();
            let mut size = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackaddr,
                                                   &mut size), 0);

            let stackaddr = stackaddr as usize;
            ret = if cfg!(target_os = "freebsd") {
                // FIXME does freebsd really fault *below* the guard addr?
                let guardaddr = stackaddr - guardsize;
                Some(guardaddr - PAGE_SIZE..guardaddr)
            } else if cfg!(target_os = "netbsd") {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", target_env = "gnu")) {
                // glibc used to include the guard area within the stack, as noted in the BUGS
                // section of `man pthread_attr_getguardsize`.  This has been corrected starting
                // with glibc 2.27, and in some distro backports, so the guard is now placed at the
                // end (below) the stack.  There's no easy way for us to know which we have at
                // runtime, so we'll just match any fault in the range right above or below the
                // stack base to call that fault a stack overflow.
                Some(stackaddr - guardsize..stackaddr + guardsize)
            } else {
                Some(stackaddr..stackaddr + guardsize)
            };
        }
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        ret
    }
}

// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus however many bytes are needed for thread-local
// storage.  We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
#[cfg(target_os = "linux")]
#[allow(deprecated)]
fn min_stack_size(attr: *const libc::pthread_attr_t) -> usize {
    weak!(fn __pthread_get_minstack(*const libc::pthread_attr_t) -> libc::size_t);

    match __pthread_get_minstack.get() {
        None => libc::PTHREAD_STACK_MIN,
        Some(f) => unsafe { f(attr) },
    }
}

// No point in looking up __pthread_get_minstack() on non-glibc
// platforms.
#[cfg(all(not(target_os = "linux"),
          not(target_os = "netbsd")))]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}

#[cfg(target_os = "netbsd")]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    2048 // just a guess
}
