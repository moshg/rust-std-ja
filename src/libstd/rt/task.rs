// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Language-level runtime services that should reasonably expected
//! to be available 'everywhere'. Local heaps, GC, unwinding,
//! local storage, and logging. Even a 'freestanding' Rust would likely want
//! to implement this.

use any::AnyOwnExt;
use borrow;
use cleanup;
use io::Writer;
use libc::{c_char, size_t};
use local_data;
use ops::Drop;
use option::{Option, Some, None};
use prelude::drop;
use result::{Result, Ok, Err};
use rt::Runtime;
use rt::borrowck::BorrowRecord;
use rt::borrowck;
use rt::local::Local;
use rt::local_heap::LocalHeap;
use rt::logging::StdErrLogger;
use rt::rtio::LocalIo;
use rt::unwind::Unwinder;
use send_str::SendStr;
use sync::arc::UnsafeArc;
use sync::atomics::{AtomicUint, SeqCst, INIT_ATOMIC_UINT};
use task::{TaskResult, TaskOpts};
use unstable::finally::Finally;
use unstable::mutex::{Mutex, MUTEX_INIT};

#[cfg(stage0)]
pub use rt::unwind::begin_unwind;

// These two statics are used as bookeeping to keep track of the rust runtime's
// count of threads. In 1:1 contexts, this is used to know when to return from
// the main function, and in M:N contexts this is used to know when to shut down
// the pool of schedulers.
static mut TASK_COUNT: AtomicUint = INIT_ATOMIC_UINT;
static mut TASK_LOCK: Mutex = MUTEX_INIT;

// The Task struct represents all state associated with a rust
// task. There are at this point two primary "subtypes" of task,
// however instead of using a subtype we just have a "task_type" field
// in the struct. This contains a pointer to another struct that holds
// the type-specific state.

pub struct Task {
    heap: LocalHeap,
    gc: GarbageCollector,
    storage: LocalStorage,
    unwinder: Unwinder,
    death: Death,
    destroyed: bool,
    name: Option<SendStr>,
    // Dynamic borrowck debugging info
    borrow_list: Option<~[BorrowRecord]>,

    logger: Option<StdErrLogger>,
    stdout_handle: Option<~Writer>,

    priv imp: Option<~Runtime>,
}

pub struct GarbageCollector;
pub struct LocalStorage(Option<local_data::Map>);

/// A handle to a blocked task. Usually this means having the ~Task pointer by
/// ownership, but if the task is killable, a killer can steal it at any time.
pub enum BlockedTask {
    Owned(~Task),
    Shared(UnsafeArc<AtomicUint>),
}

/// Per-task state related to task death, killing, failure, etc.
pub struct Death {
    // Action to be done with the exit code. If set, also makes the task wait
    // until all its watched children exit before collecting the status.
    on_exit: Option<proc(TaskResult)>,
}

pub struct BlockedTaskIterator {
    priv inner: UnsafeArc<AtomicUint>,
}

impl Task {
    pub fn new() -> Task {
        Task {
            heap: LocalHeap::new(),
            gc: GarbageCollector,
            storage: LocalStorage(None),
            unwinder: Unwinder::new(),
            death: Death::new(),
            destroyed: false,
            name: None,
            borrow_list: None,
            logger: None,
            stdout_handle: None,
            imp: None,
        }
    }

    /// Executes the given closure as if it's running inside this task. The task
    /// is consumed upon entry, and the destroyed task is returned from this
    /// function in order for the caller to free. This function is guaranteed to
    /// not unwind because the closure specified is run inside of a `rust_try`
    /// block. (this is the only try/catch block in the world).
    ///
    /// This function is *not* meant to be abused as a "try/catch" block. This
    /// is meant to be used at the absolute boundaries of a task's lifetime, and
    /// only for that purpose.
    pub fn run(~self, f: ||) -> ~Task {
        // Need to put ourselves into TLS, but also need access to the unwinder.
        // Unsafely get a handle to the task so we can continue to use it after
        // putting it in tls (so we can invoke the unwinder).
        let handle: *mut Task = unsafe {
            *cast::transmute::<&~Task, &*mut Task>(&self)
        };
        Local::put(self);
        unsafe { TASK_COUNT.fetch_add(1, SeqCst); }

        // The only try/catch block in the world. Attempt to run the task's
        // client-specified code and catch any failures.
        let try_block = || {

            // Run the task main function, then do some cleanup.
            f.finally(|| {
                fn flush(w: Option<~Writer>) {
                    match w {
                        Some(mut w) => { w.flush(); }
                        None => {}
                    }
                }

                // First, destroy task-local storage. This may run user dtors.
                //
                // FIXME #8302: Dear diary. I'm so tired and confused.
                // There's some interaction in rustc between the box
                // annihilator and the TLS dtor by which TLS is
                // accessed from annihilated box dtors *after* TLS is
                // destroyed. Somehow setting TLS back to null, as the
                // old runtime did, makes this work, but I don't currently
                // understand how. I would expect that, if the annihilator
                // reinvokes TLS while TLS is uninitialized, that
                // TLS would be reinitialized but never destroyed,
                // but somehow this works. I have no idea what's going
                // on but this seems to make things magically work. FML.
                //
                // (added after initial comment) A possible interaction here is
                // that the destructors for the objects in TLS themselves invoke
                // TLS, or possibly some destructors for those objects being
                // annihilated invoke TLS. Sadly these two operations seemed to
                // be intertwined, and miraculously work for now...
                let mut task = Local::borrow(None::<Task>);
                let storage = task.get().storage.take();
                drop(task);
                drop(storage);

                // Destroy remaining boxes. Also may run user dtors.
                unsafe { cleanup::annihilate(); }

                // Finally flush and destroy any output handles which the task
                // owns. There are no boxes here, and no user destructors should
                // run after this any more.
                let mut task = Local::borrow(None::<Task>);
                let stdout = task.get().stdout_handle.take();
                let logger = task.get().logger.take();
                drop(task);

                flush(stdout);
                drop(logger);
            })
        };

        unsafe { (*handle).unwinder.try(try_block); }

        // Cleanup the dynamic borrowck debugging info
        borrowck::clear_task_borrow_list();

        // Here we must unsafely borrow the task in order to not remove it from
        // TLS. When collecting failure, we may attempt to send on a channel (or
        // just run aribitrary code), so we must be sure to still have a local
        // task in TLS.
        unsafe {
            let me: *mut Task = Local::unsafe_borrow();
            (*me).death.collect_failure((*me).unwinder.result());

            // see comments on these statics for why they're used
            if TASK_COUNT.fetch_sub(1, SeqCst) == 1 {
                TASK_LOCK.lock();
                TASK_LOCK.signal();
                TASK_LOCK.unlock();
            }
        }
        let mut me: ~Task = Local::take();
        me.destroyed = true;
        return me;
    }

    /// Inserts a runtime object into this task, transferring ownership to the
    /// task. It is illegal to replace a previous runtime object in this task
    /// with this argument.
    pub fn put_runtime(&mut self, ops: ~Runtime) {
        assert!(self.imp.is_none());
        self.imp = Some(ops);
    }

    /// Attempts to extract the runtime as a specific type. If the runtime does
    /// not have the provided type, then the runtime is not removed. If the
    /// runtime does have the specified type, then it is removed and returned
    /// (transfer of ownership).
    ///
    /// It is recommended to only use this method when *absolutely necessary*.
    /// This function may not be available in the future.
    pub fn maybe_take_runtime<T: 'static>(&mut self) -> Option<~T> {
        // This is a terrible, terrible function. The general idea here is to
        // take the runtime, cast it to ~Any, check if it has the right type,
        // and then re-cast it back if necessary. The method of doing this is
        // pretty sketchy and involves shuffling vtables of trait objects
        // around, but it gets the job done.
        //
        // XXX: This function is a serious code smell and should be avoided at
        //      all costs. I have yet to think of a method to avoid this
        //      function, and I would be saddened if more usage of the function
        //      crops up.
        unsafe {
            let imp = self.imp.take_unwrap();
            let &(vtable, _): &(uint, uint) = cast::transmute(&imp);
            match imp.wrap().move::<T>() {
                Ok(t) => Some(t),
                Err(t) => {
                    let (_, obj): (uint, uint) = cast::transmute(t);
                    let obj: ~Runtime = cast::transmute((vtable, obj));
                    self.put_runtime(obj);
                    None
                }
            }
        }
    }

    /// Spawns a sibling to this task. The newly spawned task is configured with
    /// the `opts` structure and will run `f` as the body of its code.
    pub fn spawn_sibling(mut ~self, opts: TaskOpts, f: proc()) {
        let ops = self.imp.take_unwrap();
        ops.spawn_sibling(self, opts, f)
    }

    /// Deschedules the current task, invoking `f` `amt` times. It is not
    /// recommended to use this function directly, but rather communication
    /// primitives in `std::comm` should be used.
    pub fn deschedule(mut ~self, amt: uint,
                      f: |BlockedTask| -> Result<(), BlockedTask>) {
        let ops = self.imp.take_unwrap();
        ops.deschedule(amt, self, f)
    }

    /// Wakes up a previously blocked task, optionally specifiying whether the
    /// current task can accept a change in scheduling. This function can only
    /// be called on tasks that were previously blocked in `deschedule`.
    pub fn reawaken(mut ~self, can_resched: bool) {
        let ops = self.imp.take_unwrap();
        ops.reawaken(self, can_resched);
    }

    /// Yields control of this task to another task. This function will
    /// eventually return, but possibly not immediately. This is used as an
    /// opportunity to allow other tasks a chance to run.
    pub fn yield_now(mut ~self) {
        let ops = self.imp.take_unwrap();
        ops.yield_now(self);
    }

    /// Similar to `yield_now`, except that this function may immediately return
    /// without yielding (depending on what the runtime decides to do).
    pub fn maybe_yield(mut ~self) {
        let ops = self.imp.take_unwrap();
        ops.maybe_yield(self);
    }

    /// Acquires a handle to the I/O factory that this task contains, normally
    /// stored in the task's runtime. This factory may not always be available,
    /// which is why the return type is `Option`
    pub fn local_io<'a>(&'a mut self) -> Option<LocalIo<'a>> {
        self.imp.get_mut_ref().local_io()
    }

    /// The main function of all rust executables will by default use this
    /// function. This function will *block* the OS thread (hence the `unsafe`)
    /// waiting for all known tasks to complete. Once this function has
    /// returned, it is guaranteed that no more user-defined code is still
    /// running.
    pub unsafe fn wait_for_other_tasks(&mut self) {
        TASK_COUNT.fetch_sub(1, SeqCst); // don't count ourselves
        TASK_LOCK.lock();
        while TASK_COUNT.load(SeqCst) > 0 {
            TASK_LOCK.wait();
        }
        TASK_LOCK.unlock();
        TASK_COUNT.fetch_add(1, SeqCst); // add ourselves back in
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        rtdebug!("called drop for a task: {}", borrow::to_uint(self));
        rtassert!(self.destroyed);
    }
}

impl Iterator<BlockedTask> for BlockedTaskIterator {
    fn next(&mut self) -> Option<BlockedTask> {
        Some(Shared(self.inner.clone()))
    }
}

impl BlockedTask {
    /// Returns Some if the task was successfully woken; None if already killed.
    pub fn wake(self) -> Option<~Task> {
        match self {
            Owned(task) => Some(task),
            Shared(arc) => unsafe {
                match (*arc.get()).swap(0, SeqCst) {
                    0 => None,
                    n => Some(cast::transmute(n)),
                }
            }
        }
    }

    // This assertion has two flavours because the wake involves an atomic op.
    // In the faster version, destructors will fail dramatically instead.
    #[cfg(not(test))] pub fn trash(self) { }
    #[cfg(test)]      pub fn trash(self) { assert!(self.wake().is_none()); }

    /// Create a blocked task, unless the task was already killed.
    pub fn block(task: ~Task) -> BlockedTask {
        Owned(task)
    }

    /// Converts one blocked task handle to a list of many handles to the same.
    pub fn make_selectable(self, num_handles: uint) -> Take<BlockedTaskIterator>
    {
        let arc = match self {
            Owned(task) => {
                let flag = unsafe { AtomicUint::new(cast::transmute(task)) };
                UnsafeArc::new(flag)
            }
            Shared(arc) => arc.clone(),
        };
        BlockedTaskIterator{ inner: arc }.take(num_handles)
    }

    /// Convert to an unsafe uint value. Useful for storing in a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_to_uint(self) -> uint {
        match self {
            Owned(task) => {
                let blocked_task_ptr: uint = cast::transmute(task);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr
            }
            Shared(arc) => {
                let blocked_task_ptr: uint = cast::transmute(~arc);
                rtassert!(blocked_task_ptr & 0x1 == 0);
                blocked_task_ptr | 0x1
            }
        }
    }

    /// Convert from an unsafe uint value. Useful for retrieving a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_from_uint(blocked_task_ptr: uint) -> BlockedTask {
        if blocked_task_ptr & 0x1 == 0 {
            Owned(cast::transmute(blocked_task_ptr))
        } else {
            let ptr: ~UnsafeArc<AtomicUint> =
                cast::transmute(blocked_task_ptr & !1);
            Shared(*ptr)
        }
    }
}

impl Death {
    pub fn new() -> Death {
        Death { on_exit: None, }
    }

    /// Collect failure exit codes from children and propagate them to a parent.
    pub fn collect_failure(&mut self, result: TaskResult) {
        match self.on_exit.take() {
            Some(f) => f(result),
            None => {}
        }
    }
}

impl Drop for Death {
    fn drop(&mut self) {
        // make this type noncopyable
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use prelude::*;
    use task;

    #[test]
    fn local_heap() {
        let a = @5;
        let b = a;
        assert!(*a == 5);
        assert!(*b == 5);
    }

    #[test]
    fn tls() {
        use local_data;
        local_data_key!(key: @~str)
        local_data::set(key, @~"data");
        assert!(*local_data::get(key, |k| k.map(|k| *k)).unwrap() == ~"data");
        local_data_key!(key2: @~str)
        local_data::set(key2, @~"data");
        assert!(*local_data::get(key2, |k| k.map(|k| *k)).unwrap() == ~"data");
    }

    #[test]
    fn unwind() {
        let result = task::try(proc()());
        rtdebug!("trying first assert");
        assert!(result.is_ok());
        let result = task::try::<()>(proc() fail!());
        rtdebug!("trying second assert");
        assert!(result.is_err());
    }

    #[test]
    fn rng() {
        use rand::{rng, Rng};
        let mut r = rng();
        let _ = r.next_u32();
    }

    #[test]
    fn logging() {
        info!("here i am. logging in a newsched task");
    }

    #[test]
    fn comm_stream() {
        let (port, chan) = Chan::new();
        chan.send(10);
        assert!(port.recv() == 10);
    }

    #[test]
    fn comm_shared_chan() {
        let (port, chan) = SharedChan::new();
        chan.send(10);
        assert!(port.recv() == 10);
    }

    #[test]
    fn heap_cycles() {
        use option::{Option, Some, None};

        struct List {
            next: Option<@mut List>,
        }

        let a = @mut List { next: None };
        let b = @mut List { next: Some(a) };

        a.next = Some(b);
    }

    #[test]
    #[should_fail]
    fn test_begin_unwind() { begin_unwind("cause", file!(), line!()) }

    // Task blocking tests

    #[test]
    fn block_and_wake() {
        let task = ~Task::new();
        let mut task = BlockedTask::block(task).wake().unwrap();
        task.destroyed = true;
    }
}
