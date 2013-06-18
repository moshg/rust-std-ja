// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use sys;
use cast::transmute;
use cell::Cell;
use clone::Clone;

use super::sleeper_list::SleeperList;
use super::work_queue::WorkQueue;
use super::stack::{StackPool, StackSegment};
use super::rtio::{EventLoop, EventLoopObject, RemoteCallbackObject};
use super::context::Context;
use super::task::Task;
use super::message_queue::MessageQueue;
use rt::local_ptr;
use rt::local::Local;
use rt::rtio::RemoteCallback;
use rt::metrics::SchedMetrics;

//use to_str::ToStr;

/// To allow for using pointers as scheduler ids
use borrow::{to_uint};

/// The Scheduler is responsible for coordinating execution of Coroutines
/// on a single thread. When the scheduler is running it is owned by
/// thread local storage and the running task is owned by the
/// scheduler.
///
/// XXX: This creates too many callbacks to run_sched_once, resulting
/// in too much allocation and too many events.
pub struct Scheduler {
    /// A queue of available work. Under a work-stealing policy there
    /// is one per Scheduler.
    priv work_queue: WorkQueue<~Coroutine>,
    /// The queue of incoming messages from other schedulers.
    /// These are enqueued by SchedHandles after which a remote callback
    /// is triggered to handle the message.
    priv message_queue: MessageQueue<SchedMessage>,
    /// A shared list of sleeping schedulers. We'll use this to wake
    /// up schedulers when pushing work onto the work queue.
    priv sleeper_list: SleeperList,
    /// Indicates that we have previously pushed a handle onto the
    /// SleeperList but have not yet received the Wake message.
    /// Being `true` does not necessarily mean that the scheduler is
    /// not active since there are multiple event sources that may
    /// wake the scheduler. It just prevents the scheduler from pushing
    /// multiple handles onto the sleeper list.
    priv sleepy: bool,
    /// A flag to indicate we've received the shutdown message and should
    /// no longer try to go to sleep, but exit instead.
    no_sleep: bool,
    stack_pool: StackPool,
    /// The event loop used to drive the scheduler and perform I/O
    event_loop: ~EventLoopObject,
    /// The scheduler's saved context.
    /// Always valid when a task is executing, otherwise not
    priv saved_context: Context,
    /// The currently executing task
    current_task: Option<~Coroutine>,
    /// An action performed after a context switch on behalf of the
    /// code running before the context switch
    priv cleanup_job: Option<CleanupJob>,
    metrics: SchedMetrics,
    /// Should this scheduler run any task, or only pinned tasks?
    run_anything: bool
}

pub struct SchedHandle {
    priv remote: ~RemoteCallbackObject,
    priv queue: MessageQueue<SchedMessage>,
    sched_id: uint
}

pub struct Coroutine {
    /// The segment of stack on which the task is currently running or,
    /// if the task is blocked, on which the task will resume execution
    priv current_stack_segment: StackSegment,
    /// These are always valid when the task is not running, unless
    /// the task is dead
    priv saved_context: Context,
    /// The heap, GC, unwinding, local storage, logging
    task: ~Task,
}

// A scheduler home is either a handle to the home scheduler, or an
// explicit "AnySched".
pub enum SchedHome {
    AnySched,
    Sched(SchedHandle)
}

pub enum SchedMessage {
    Wake,
    Shutdown,
    PinnedTask(~Coroutine)
}

enum CleanupJob {
    DoNothing,
    GiveTask(~Coroutine, UnsafeTaskReceiver)
}

impl Scheduler {
    pub fn in_task_context(&self) -> bool { self.current_task.is_some() }

    pub fn sched_id(&self) -> uint { to_uint(self) }

    pub fn new(event_loop: ~EventLoopObject,
               work_queue: WorkQueue<~Coroutine>,
               sleeper_list: SleeperList)
        -> Scheduler {

        Scheduler::new_special(event_loop, work_queue, sleeper_list, true)

    }

    pub fn new_special(event_loop: ~EventLoopObject,
                       work_queue: WorkQueue<~Coroutine>,
                       sleeper_list: SleeperList,
                       run_anything: bool)
        -> Scheduler {

        // Lazily initialize the runtime TLS key
        local_ptr::init_tls_key();

        Scheduler {
            sleeper_list: sleeper_list,
            message_queue: MessageQueue::new(),
            sleepy: false,
            no_sleep: false,
            event_loop: event_loop,
            work_queue: work_queue,
            stack_pool: StackPool::new(),
            saved_context: Context::empty(),
            current_task: None,
            cleanup_job: None,
            metrics: SchedMetrics::new(),
            run_anything: run_anything
        }
    }

    // XXX: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.
    pub fn run(~self) -> ~Scheduler {
        assert!(!self.in_task_context());

        let mut self_sched = self;

        // Always run through the scheduler loop at least once so that
        // we enter the sleep state and can then be woken up by other
        // schedulers.
        self_sched.event_loop.callback(Scheduler::run_sched_once);

        unsafe {
            let event_loop: *mut ~EventLoopObject = {
                let event_loop: *mut ~EventLoopObject = &mut self_sched.event_loop;
                event_loop
            };

            // Give ownership of the scheduler (self) to the thread
            Local::put(self_sched);

            (*event_loop).run();
        }

        rtdebug!("run taking sched");
        let sched = Local::take::<Scheduler>();
        // XXX: Reenable this once we're using a per-task queue. With a shared
        // queue this is not true
        //assert!(sched.work_queue.is_empty());
        rtdebug!("scheduler metrics: %s\n", {
            use to_str::ToStr;
            sched.metrics.to_str()
        });
        return sched;
    }

    fn run_sched_once() {

        let mut sched = Local::take::<Scheduler>();
        sched.metrics.turns += 1;

        // First, check the message queue for instructions.
        // XXX: perf. Check for messages without atomics.
        // It's ok if we miss messages occasionally, as long as
        // we sync and check again before sleeping.
        if sched.interpret_message_queue() {
            // We performed a scheduling action. There may be other work
            // to do yet, so let's try again later.
            rtdebug!("run_sched_once, interpret_message_queue taking sched");
            let mut sched = Local::take::<Scheduler>();
            sched.metrics.messages_received += 1;
            sched.event_loop.callback(Scheduler::run_sched_once);
            Local::put(sched);
            return;
        }

        // Now, look in the work queue for tasks to run
        rtdebug!("run_sched_once taking");
        let sched = Local::take::<Scheduler>();
        if sched.resume_task_from_queue() {
            // We performed a scheduling action. There may be other work
            // to do yet, so let's try again later.
            let mut sched = Local::take::<Scheduler>();
            sched.metrics.tasks_resumed_from_queue += 1;
            sched.event_loop.callback(Scheduler::run_sched_once);
            Local::put(sched);
            return;
        }

        // If we got here then there was no work to do.
        // Generate a SchedHandle and push it to the sleeper list so
        // somebody can wake us up later.
        rtdebug!("no work to do");
        let mut sched = Local::take::<Scheduler>();
        sched.metrics.wasted_turns += 1;
        if !sched.sleepy && !sched.no_sleep {
            rtdebug!("sleeping");
            sched.metrics.sleepy_times += 1;
            sched.sleepy = true;
            let handle = sched.make_handle();
            sched.sleeper_list.push(handle);
        } else {
            rtdebug!("not sleeping");
        }
        Local::put(sched);
    }

    pub fn make_handle(&mut self) -> SchedHandle {
        let remote = self.event_loop.remote_callback(Scheduler::run_sched_once);

        return SchedHandle {
            remote: remote,
            queue: self.message_queue.clone(),
            sched_id: self.sched_id()
        };
    }

    /// Schedule a task to be executed later.
    ///
    /// Pushes the task onto the work stealing queue and tells the
    /// event loop to run it later. Always use this instead of pushing
    /// to the work queue directly.
    pub fn enqueue_task(&mut self, task: ~Coroutine) {

        // We don't want to queue tasks that belong on other threads,
        // so we send them home at enqueue time.

        // The borrow checker doesn't like our disassembly of the
        // Coroutine struct and partial use and mutation of the
        // fields. So completely disassemble here and stop using?

        // XXX perf: I think we might be able to shuffle this code to
        // only destruct when we need to.

        rtdebug!("a task was queued on: %u", self.sched_id());

        let this = self;

        // We push the task onto our local queue clone.
        this.work_queue.push(task);
        this.event_loop.callback(Scheduler::run_sched_once);

        // We've made work available. Notify a
        // sleeping scheduler.

        // XXX: perf. Check for a sleeper without
        // synchronizing memory.  It's not critical
        // that we always find it.

        // XXX: perf. If there's a sleeper then we
        // might as well just send it the task
        // directly instead of pushing it to the
        // queue. That is essentially the intent here
        // and it is less work.
        match this.sleeper_list.pop() {
            Some(handle) => {
                let mut handle = handle;
                handle.send(Wake)
            }
            None => { (/* pass */) }
        };
    }

    // * Scheduler-context operations

    fn interpret_message_queue(~self) -> bool {
        assert!(!self.in_task_context());

        rtdebug!("looking for scheduler messages");

        let mut this = self;
        match this.message_queue.pop() {
            Some(PinnedTask(task)) => {
                rtdebug!("recv BiasedTask message in sched: %u",
                         this.sched_id());
                let mut task = task;
                task.task.home = Some(Sched(this.make_handle()));
                this.resume_task_immediately(task);
                return true;
            }

            Some(Wake) => {
                rtdebug!("recv Wake message");
                this.sleepy = false;
                Local::put(this);
                return true;
            }
            Some(Shutdown) => {
                rtdebug!("recv Shutdown message");
                if this.sleepy {
                    // There may be an outstanding handle on the
                    // sleeper list.  Pop them all to make sure that's
                    // not the case.
                    loop {
                        match this.sleeper_list.pop() {
                            Some(handle) => {
                                let mut handle = handle;
                                handle.send(Wake);
                            }
                            None => break
                        }
                    }
                }
                // No more sleeping. After there are no outstanding
                // event loop references we will shut down.
                this.no_sleep = true;
                this.sleepy = false;
                Local::put(this);
                return true;
            }
            None => {
                Local::put(this);
                return false;
            }
        }
    }

    /// Given an input Coroutine sends it back to its home scheduler.
    fn send_task_home(task: ~Coroutine) {
        let mut task = task;
        let mut home = task.task.home.swap_unwrap();
        match home {
            Sched(ref mut home_handle) => {
                home_handle.send(PinnedTask(task));
            }
            AnySched => {
                rtabort!("error: cannot send anysched task home");
            }
        }
    }

    // Resume a task from the queue - but also take into account that
    // it might not belong here.
    fn resume_task_from_queue(~self) -> bool {
        assert!(!self.in_task_context());

        rtdebug!("looking in work queue for task to schedule");
        let mut this = self;

        // The borrow checker imposes the possibly absurd requirement
        // that we split this into two match expressions. This is due
        // to the inspection of the internal bits of task, as that
        // can't be in scope when we act on task.
        match this.work_queue.pop() {
            Some(task) => {
                let action_id = {
                    let home = &task.task.home;
                    match home {
                        &Some(Sched(ref home_handle))
                        if home_handle.sched_id != this.sched_id() => {
                            0
                        }
                        &Some(AnySched) if this.run_anything => {
                            1
                        }
                        &Some(AnySched) => {
                            2
                        }
                        &Some(Sched(_)) => {
                            3
                        }
                        &None => {
                            4
                        }
                    }
                };

                match action_id {
                    0 => {
                        rtdebug!("sending task home");
                        Scheduler::send_task_home(task);
                        Local::put(this);
                        return false;
                    }
                    1 => {
                        rtdebug!("resuming now");
                        this.resume_task_immediately(task);
                        return true;
                    }
                    2 => {
                        rtdebug!("re-queueing")
                        this.enqueue_task(task);
                        Local::put(this);
                        return false;
                    }
                    3 => {
                        rtdebug!("resuming now");
                        this.resume_task_immediately(task);
                        return true;
                    }
                    4 => {
                        rtabort!("task home was None!");
                    }
                    _ => {
                        rtabort!("literally, you should not be here");
                    }
                }
            }

            None => {
               rtdebug!("no tasks in queue");
               Local::put(this);
               return false;
           }
        }
    }

    // * Task-context operations

    /// Called by a running task to end execution, after which it will
    /// be recycled by the scheduler for reuse in a new task.
    pub fn terminate_current_task(~self) {
        assert!(self.in_task_context());

        rtdebug!("ending running task");

        do self.deschedule_running_task_and_then |sched, dead_task| {
            let dead_task = Cell::new(dead_task);
            dead_task.take().recycle(&mut sched.stack_pool);
        }

        rtabort!("control reached end of task");
    }

    pub fn schedule_task(~self, task: ~Coroutine) {
        assert!(self.in_task_context());

        // is the task home?
        let is_home = task.is_home_no_tls(&self);

        // does the task have a home?
        let homed = task.homed();

        let mut this = self;

        if is_home || (!homed && this.run_anything) {
            // here we know we are home, execute now OR we know we
            // aren't homed, and that this sched doesn't care
            do this.switch_running_tasks_and_then(task) |sched, last_task| {
                let last_task = Cell::new(last_task);
                sched.enqueue_task(last_task.take());
            }
        } else if !homed && !this.run_anything {
            // the task isn't homed, but it can't be run here
            this.enqueue_task(task);
            Local::put(this);
        } else {
            // task isn't home, so don't run it here, send it home
            Scheduler::send_task_home(task);
            Local::put(this);
        }
    }

    // Core scheduling ops

    pub fn resume_task_immediately(~self, task: ~Coroutine) {
        let mut this = self;
        assert!(!this.in_task_context());

        rtdebug!("scheduling a task");
        this.metrics.context_switches_sched_to_task += 1;

        // Store the task in the scheduler so it can be grabbed later
        this.current_task = Some(task);
        this.enqueue_cleanup_job(DoNothing);

        Local::put(this);

        // Take pointers to both the task and scheduler's saved registers.
        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (sched_context, _, next_task_context) = (*sched).get_contexts();
            let next_task_context = next_task_context.unwrap();
            // Context switch to the task, restoring it's registers
            // and saving the scheduler's
            Context::swap(sched_context, next_task_context);

            let sched = Local::unsafe_borrow::<Scheduler>();
            // The running task should have passed ownership elsewhere
            assert!((*sched).current_task.is_none());

            // Running tasks may have asked us to do some cleanup
            (*sched).run_cleanup_job();
        }
    }

    /// Block a running task, context switch to the scheduler, then pass the
    /// blocked task to a closure.
    ///
    /// # Safety note
    ///
    /// The closure here is a *stack* closure that lives in the
    /// running task.  It gets transmuted to the scheduler's lifetime
    /// and called while the task is blocked.
    ///
    /// This passes a Scheduler pointer to the fn after the context switch
    /// in order to prevent that fn from performing further scheduling operations.
    /// Doing further scheduling could easily result in infinite recursion.
    pub fn deschedule_running_task_and_then(~self, f: &fn(&mut Scheduler, ~Coroutine)) {
        let mut this = self;
        assert!(this.in_task_context());

        rtdebug!("blocking task");
        this.metrics.context_switches_task_to_sched += 1;

        unsafe {
            let blocked_task = this.current_task.swap_unwrap();
            let f_fake_region = transmute::<&fn(&mut Scheduler, ~Coroutine),
                                            &fn(&mut Scheduler, ~Coroutine)>(f);
            let f_opaque = ClosureConverter::from_fn(f_fake_region);
            this.enqueue_cleanup_job(GiveTask(blocked_task, f_opaque));
        }

        Local::put(this);

        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (sched_context, last_task_context, _) = (*sched).get_contexts();
            let last_task_context = last_task_context.unwrap();
            Context::swap(last_task_context, sched_context);

            // We could be executing in a different thread now
            let sched = Local::unsafe_borrow::<Scheduler>();
            (*sched).run_cleanup_job();
        }
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
    pub fn switch_running_tasks_and_then(~self, next_task: ~Coroutine,
                                         f: &fn(&mut Scheduler, ~Coroutine)) {
        let mut this = self;
        assert!(this.in_task_context());

        rtdebug!("switching tasks");
        this.metrics.context_switches_task_to_task += 1;

        let old_running_task = this.current_task.swap_unwrap();
        let f_fake_region = unsafe {
            transmute::<&fn(&mut Scheduler, ~Coroutine),
                        &fn(&mut Scheduler, ~Coroutine)>(f)
        };
        let f_opaque = ClosureConverter::from_fn(f_fake_region);
        this.enqueue_cleanup_job(GiveTask(old_running_task, f_opaque));
        this.current_task = Some(next_task);

        Local::put(this);

        unsafe {
            let sched = Local::unsafe_borrow::<Scheduler>();
            let (_, last_task_context, next_task_context) = (*sched).get_contexts();
            let last_task_context = last_task_context.unwrap();
            let next_task_context = next_task_context.unwrap();
            Context::swap(last_task_context, next_task_context);

            // We could be executing in a different thread now
            let sched = Local::unsafe_borrow::<Scheduler>();
            (*sched).run_cleanup_job();
        }
    }



    // * Other stuff

    pub fn enqueue_cleanup_job(&mut self, job: CleanupJob) {
        assert!(self.cleanup_job.is_none());
        self.cleanup_job = Some(job);
    }

    pub fn run_cleanup_job(&mut self) {
        rtdebug!("running cleanup job");

        assert!(self.cleanup_job.is_some());

        let cleanup_job = self.cleanup_job.swap_unwrap();
        match cleanup_job {
            DoNothing => { }
            GiveTask(task, f) => (f.to_fn())(self, task)
        }
    }

    /// Get mutable references to all the contexts that may be involved in a
    /// context switch.
    ///
    /// Returns (the scheduler context, the optional context of the
    /// task in the cleanup list, the optional context of the task in
    /// the current task slot).  When context switching to a task,
    /// callers should first arrange for that task to be located in the
    /// Scheduler's current_task slot and set up the
    /// post-context-switch cleanup job.
    pub fn get_contexts<'a>(&'a mut self) -> (&'a mut Context,
                                              Option<&'a mut Context>,
                                              Option<&'a mut Context>) {
        let last_task = match self.cleanup_job {
            Some(GiveTask(~ref task, _)) => {
                Some(task)
            }
            Some(DoNothing) => {
                None
            }
            None => fail!("all context switches should have a cleanup job")
        };
        // XXX: Pattern matching mutable pointers above doesn't work
        // because borrowck thinks the three patterns are conflicting
        // borrows
        unsafe {
            let last_task = transmute::<Option<&Coroutine>, Option<&mut Coroutine>>(last_task);
            let last_task_context = match last_task {
                Some(t) => Some(&mut t.saved_context), None => None
            };
            let next_task_context = match self.current_task {
                Some(ref mut t) => Some(&mut t.saved_context), None => None
            };
            // XXX: These transmutes can be removed after snapshot
            return (transmute(&mut self.saved_context),
                    last_task_context,
                    transmute(next_task_context));
        }
    }
}

impl SchedHandle {
    pub fn send(&mut self, msg: SchedMessage) {
        self.queue.push(msg);
        self.remote.fire();
    }
}

impl Coroutine {

    /// This function checks that a coroutine is running "home".
    pub fn is_home(&self) -> bool {
        rtdebug!("checking if coroutine is home");
        do Local::borrow::<Scheduler,bool> |sched| {
            match self.task.home {
                Some(AnySched) => { false }
                Some(Sched(SchedHandle { sched_id: ref id, _ })) => {
                    *id == sched.sched_id()
                }
                None => { rtabort!("error: homeless task!"); }
            }
        }
    }

    /// Without access to self, but with access to the "expected home
    /// id", see if we are home.
    fn is_home_using_id(id: uint) -> bool {
        rtdebug!("checking if coroutine is home using id");
        do Local::borrow::<Scheduler,bool> |sched| {
            if sched.sched_id() == id {
                true
            } else {
                false
            }
        }
    }

    /// Check if this coroutine has a home
    fn homed(&self) -> bool {
        rtdebug!("checking if this coroutine has a home");
        match self.task.home {
            Some(AnySched) => { false }
            Some(Sched(_)) => { true }
            None => { rtabort!("error: homeless task!");
                    }
        }
    }

    /// A version of is_home that does not need to use TLS, it instead
    /// takes local scheduler as a parameter.
    fn is_home_no_tls(&self, sched: &~Scheduler) -> bool {
        rtdebug!("checking if coroutine is home without tls");
        match self.task.home {
            Some(AnySched) => { true }
            Some(Sched(SchedHandle { sched_id: ref id, _})) => {
                *id == sched.sched_id()
            }
            None => { rtabort!("error: homeless task!"); }
        }
    }

    /// Check TLS for the scheduler to see if we are on a special
    /// scheduler.
    pub fn on_special() -> bool {
        rtdebug!("checking if coroutine is executing on special sched");
        do Local::borrow::<Scheduler,bool>() |sched| {
            !sched.run_anything
        }
    }

    // Created new variants of "new" that takes a home scheduler
    // parameter. The original with_task now calls with_task_homed
    // using the AnySched paramter.

    pub fn new_homed(stack_pool: &mut StackPool, home: SchedHome, start: ~fn()) -> Coroutine {
        Coroutine::with_task_homed(stack_pool, ~Task::new_root(), start, home)
    }

    pub fn new_root(stack_pool: &mut StackPool, start: ~fn()) -> Coroutine {
        Coroutine::with_task(stack_pool, ~Task::new_root(), start)
    }

    pub fn with_task_homed(stack_pool: &mut StackPool,
                           task: ~Task,
                           start: ~fn(),
                           home: SchedHome) -> Coroutine {

        static MIN_STACK_SIZE: uint = 1000000; // XXX: Too much stack

        let start = Coroutine::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(MIN_STACK_SIZE);
        // NB: Context holds a pointer to that ~fn
        let initial_context = Context::new(start, &mut stack);
        let mut crt = Coroutine {
            current_stack_segment: stack,
            saved_context: initial_context,
            task: task,
        };
        crt.task.home = Some(home);
        return crt;
    }

    pub fn with_task(stack_pool: &mut StackPool,
                 task: ~Task,
                 start: ~fn()) -> Coroutine {
        Coroutine::with_task_homed(stack_pool,
                                   task,
                                   start,
                                   AnySched)
    }

    fn build_start_wrapper(start: ~fn()) -> ~fn() {
        // XXX: The old code didn't have this extra allocation
        let start_cell = Cell::new(start);
        let wrapper: ~fn() = || {
            // This is the first code to execute after the initial
            // context switch to the task. The previous context may
            // have asked us to do some cleanup.
            unsafe {
                let sched = Local::unsafe_borrow::<Scheduler>();
                (*sched).run_cleanup_job();

                let sched = Local::unsafe_borrow::<Scheduler>();
                let task = (*sched).current_task.get_mut_ref();
                // FIXME #6141: shouldn't neet to put `start()` in
                // another closure
                let start_cell = Cell::new(start_cell.take());
                do task.task.run {
                    // N.B. Removing `start` from the start wrapper
                    // closure by emptying a cell is critical for
                    // correctness. The ~Task pointer, and in turn the
                    // closure used to initialize the first call
                    // frame, is destroyed in scheduler context, not
                    // task context.  So any captured closures must
                    // not contain user-definable dtors that expect to
                    // be in task context. By moving `start` out of
                    // the closure, all the user code goes out of
                    // scope while the task is still running.
                    let start = start_cell.take();
                    start();
                };
            }

            let sched = Local::take::<Scheduler>();
            sched.terminate_current_task();
        };
        return wrapper;
    }

    /// Destroy the task and try to reuse its components
    pub fn recycle(~self, stack_pool: &mut StackPool) {
        match self {
            ~Coroutine {current_stack_segment, _} => {
                stack_pool.give_segment(current_stack_segment);
            }
        }
    }
}

// XXX: Some hacks to put a &fn in Scheduler without borrowck
// complaining
type UnsafeTaskReceiver = sys::Closure;
trait ClosureConverter {
    fn from_fn(&fn(&mut Scheduler, ~Coroutine)) -> Self;
    fn to_fn(self) -> &fn(&mut Scheduler, ~Coroutine);
}
impl ClosureConverter for UnsafeTaskReceiver {
    fn from_fn(f: &fn(&mut Scheduler, ~Coroutine)) -> UnsafeTaskReceiver { unsafe { transmute(f) } }
    fn to_fn(self) -> &fn(&mut Scheduler, ~Coroutine) { unsafe { transmute(self) } }
}

#[cfg(test)]
mod test {
    use int;
    use cell::Cell;
    use iterator::IteratorUtil;
    use unstable::run_in_bare_thread;
    use task::spawn;
    use rt::local::Local;
    use rt::test::*;
    use super::*;
    use rt::thread::Thread;
    use ptr::to_uint;
    use vec::MutableVector;

    // Confirm that a sched_id actually is the uint form of the
    // pointer to the scheduler struct.

    #[test]
    fn simple_sched_id_test() {
        do run_in_bare_thread {
            let sched = ~new_test_uv_sched();
            assert!(to_uint(sched) == sched.sched_id());
        }
    }

    // Compare two scheduler ids that are different, this should never
    // fail but may catch a mistake someday.

    #[test]
    fn compare_sched_id_test() {
        do run_in_bare_thread {
            let sched_one = ~new_test_uv_sched();
            let sched_two = ~new_test_uv_sched();
            assert!(sched_one.sched_id() != sched_two.sched_id());
        }
    }

    // A simple test to check if a homed task run on a single
    // scheduler ends up executing while home.

    #[test]
    fn test_home_sched() {
        do run_in_bare_thread {
            let mut task_ran = false;
            let task_ran_ptr: *mut bool = &mut task_ran;
            let mut sched = ~new_test_uv_sched();

            let sched_handle = sched.make_handle();
            let sched_id = sched.sched_id();

            let task = ~do Coroutine::new_homed(&mut sched.stack_pool,
                                                Sched(sched_handle)) {
                unsafe { *task_ran_ptr = true };
                let sched = Local::take::<Scheduler>();
                assert!(sched.sched_id() == sched_id);
                Local::put::<Scheduler>(sched);
            };
            sched.enqueue_task(task);
            sched.run();
            assert!(task_ran);
        }
    }

    // A test for each state of schedule_task

    #[test]
    fn test_schedule_home_states() {

        use rt::uv::uvio::UvEventLoop;
        use rt::sched::Shutdown;
        use rt::sleeper_list::SleeperList;
        use rt::work_queue::WorkQueue;

        do run_in_bare_thread {
//            let nthreads = 2;

            let sleepers = SleeperList::new();
            let work_queue = WorkQueue::new();

            // our normal scheduler
            let mut normal_sched = ~Scheduler::new(
                ~UvEventLoop::new(),
                work_queue.clone(),
                sleepers.clone());

            let normal_handle = Cell::new(normal_sched.make_handle());

            // our special scheduler
            let mut special_sched = ~Scheduler::new_special(
                ~UvEventLoop::new(),
                work_queue.clone(),
                sleepers.clone(),
                true);

            let special_handle = Cell::new(special_sched.make_handle());
            let special_handle2 = Cell::new(special_sched.make_handle());
            let special_id = special_sched.sched_id();
            let t1_handle = special_sched.make_handle();
            let t4_handle = special_sched.make_handle();

            let t1f = ~do Coroutine::new_homed(&mut special_sched.stack_pool,
                                            Sched(t1_handle)) {
                let is_home = Coroutine::is_home_using_id(special_id);
                rtdebug!("t1 should be home: %b", is_home);
                assert!(is_home);
            };
            let t1f = Cell::new(t1f);

            let t2f = ~do Coroutine::new_root(&mut normal_sched.stack_pool) {
                let on_special = Coroutine::on_special();
                rtdebug!("t2 should not be on special: %b", on_special);
                assert!(!on_special);
            };
            let t2f = Cell::new(t2f);

            let t3f = ~do Coroutine::new_root(&mut normal_sched.stack_pool) {
                // not on special
                let on_special = Coroutine::on_special();
                rtdebug!("t3 should not be on special: %b", on_special);
                assert!(!on_special);
            };
            let t3f = Cell::new(t3f);

            let t4f = ~do Coroutine::new_homed(&mut special_sched.stack_pool,
                                            Sched(t4_handle)) {
                // is home
                let home = Coroutine::is_home_using_id(special_id);
                rtdebug!("t4 should be home: %b", home);
                assert!(home);
            };
            let t4f = Cell::new(t4f);

            // we have four tests, make them as closures
            let t1: ~fn() = || {
                // task is home on special
                let task = t1f.take();
                let sched = Local::take::<Scheduler>();
                sched.schedule_task(task);
            };
            let t2: ~fn() = || {
                // not homed, task doesn't care
                let task = t2f.take();
                let sched = Local::take::<Scheduler>();
                sched.schedule_task(task);
            };
            let t3: ~fn() = || {
                // task not homed, must leave
                let task = t3f.take();
                let sched = Local::take::<Scheduler>();
                sched.schedule_task(task);
            };
            let t4: ~fn() = || {
                // task not home, send home
                let task = t4f.take();
                let sched = Local::take::<Scheduler>();
                sched.schedule_task(task);
            };

            let t1 = Cell::new(t1);
            let t2 = Cell::new(t2);
            let t3 = Cell::new(t3);
            let t4 = Cell::new(t4);

            // build a main task that runs our four tests
            let main_task = ~do Coroutine::new_root(&mut normal_sched.stack_pool) {
                // the two tasks that require a normal start location
                t2.take()();
                t4.take()();
                normal_handle.take().send(Shutdown);
                special_handle.take().send(Shutdown);
            };

            // task to run the two "special start" tests
            let special_task = ~do Coroutine::new_homed(
                &mut special_sched.stack_pool,
                Sched(special_handle2.take())) {
                t1.take()();
                t3.take()();
            };

            // enqueue the main tasks
            normal_sched.enqueue_task(special_task);
            normal_sched.enqueue_task(main_task);

            let nsched_cell = Cell::new(normal_sched);
            let normal_thread = do Thread::start {
                let sched = nsched_cell.take();
                sched.run();
            };

            let ssched_cell = Cell::new(special_sched);
            let special_thread = do Thread::start {
                let sched = ssched_cell.take();
                sched.run();
            };

            // wait for the end
            let _thread1 = normal_thread;
            let _thread2 = special_thread;

        }
    }

    // The following test is a bit of a mess, but it trys to do
    // something tricky so I'm not sure how to get around this in the
    // short term.

    // A number of schedulers are created, and then a task is created
    // and assigned a home scheduler. It is then "started" on a
    // different scheduler. The scheduler it is started on should
    // observe that the task is not home, and send it home.

    // This test is light in that it does very little.

    #[test]
    fn test_transfer_task_home() {

        use rt::uv::uvio::UvEventLoop;
        use rt::sched::Shutdown;
        use rt::sleeper_list::SleeperList;
        use rt::work_queue::WorkQueue;
        use uint;
        use container::Container;
        use vec::OwnedVector;

        do run_in_bare_thread {

            static N: uint = 8;

            let sleepers = SleeperList::new();
            let work_queue = WorkQueue::new();

            let mut handles = ~[];
            let mut scheds = ~[];

            for uint::range(0, N) |_| {
                let loop_ = ~UvEventLoop::new();
                let mut sched = ~Scheduler::new(loop_,
                                                work_queue.clone(),
                                                sleepers.clone());
                let handle = sched.make_handle();
                rtdebug!("sched id: %u", handle.sched_id);
                handles.push(handle);
                scheds.push(sched);
            };

            let handles = Cell::new(handles);

            let home_handle = scheds[6].make_handle();
            let home_id = home_handle.sched_id;
            let home = Sched(home_handle);

            let main_task = ~do Coroutine::new_homed(&mut scheds[1].stack_pool, home) {

                // Here we check if the task is running on its home.
                let sched = Local::take::<Scheduler>();
                rtdebug!("run location scheduler id: %u, home: %u",
                         sched.sched_id(),
                         home_id);
                assert!(sched.sched_id() == home_id);
                Local::put::<Scheduler>(sched);

                let mut handles = handles.take();
                for handles.mut_iter().advance |handle| {
                    handle.send(Shutdown);
                }
            };

            scheds[0].enqueue_task(main_task);

            let mut threads = ~[];

            while !scheds.is_empty() {
                let sched = scheds.pop();
                let sched_cell = Cell::new(sched);
                let thread = do Thread::start {
                    let sched = sched_cell.take();
                    sched.run();
                };
                threads.push(thread);
            }

            let _threads = threads;
        }
    }

    // Do it a lot

    #[test]
    fn test_stress_schedule_task_states() {
        let n = stress_factor() * 120;
        for int::range(0,n as int) |_| {
            test_schedule_home_states();
        }
    }

    // The goal is that this is the high-stress test for making sure
    // homing is working. It allocates RUST_RT_STRESS tasks that
    // do nothing but assert that they are home at execution
    // time. These tasks are queued to random schedulers, so sometimes
    // they are home and sometimes not. It also runs RUST_RT_STRESS
    // times.

    #[test]
    fn test_stress_homed_tasks() {
        let n = stress_factor();
        for int::range(0,n as int) |_| {
            run_in_mt_newsched_task_random_homed();
        }
    }

    #[test]
    fn test_simple_scheduling() {
        do run_in_bare_thread {
            let mut task_ran = false;
            let task_ran_ptr: *mut bool = &mut task_ran;

            let mut sched = ~new_test_uv_sched();
            let task = ~do Coroutine::new_root(&mut sched.stack_pool) {
                unsafe { *task_ran_ptr = true; }
            };
            sched.enqueue_task(task);
            sched.run();
            assert!(task_ran);
        }
    }

    #[test]
    fn test_several_tasks() {
        do run_in_bare_thread {
            let total = 10;
            let mut task_count = 0;
            let task_count_ptr: *mut int = &mut task_count;

            let mut sched = ~new_test_uv_sched();
            for int::range(0, total) |_| {
                let task = ~do Coroutine::new_root(&mut sched.stack_pool) {
                    unsafe { *task_count_ptr = *task_count_ptr + 1; }
                };
                sched.enqueue_task(task);
            }
            sched.run();
            assert_eq!(task_count, total);
        }
    }

    #[test]
    fn test_swap_tasks_then() {
        do run_in_bare_thread {
            let mut count = 0;
            let count_ptr: *mut int = &mut count;

            let mut sched = ~new_test_uv_sched();
            let task1 = ~do Coroutine::new_root(&mut sched.stack_pool) {
                unsafe { *count_ptr = *count_ptr + 1; }
                let mut sched = Local::take::<Scheduler>();
                let task2 = ~do Coroutine::new_root(&mut sched.stack_pool) {
                    unsafe { *count_ptr = *count_ptr + 1; }
                };
                // Context switch directly to the new task
                do sched.switch_running_tasks_and_then(task2) |sched, task1| {
                    let task1 = Cell::new(task1);
                    sched.enqueue_task(task1.take());
                }
                unsafe { *count_ptr = *count_ptr + 1; }
            };
            sched.enqueue_task(task1);
            sched.run();
            assert_eq!(count, 3);
        }
    }

    #[bench] #[test] #[ignore(reason = "long test")]
    fn test_run_a_lot_of_tasks_queued() {
        do run_in_bare_thread {
            static MAX: int = 1000000;
            let mut count = 0;
            let count_ptr: *mut int = &mut count;

            let mut sched = ~new_test_uv_sched();

            let start_task = ~do Coroutine::new_root(&mut sched.stack_pool) {
                run_task(count_ptr);
            };
            sched.enqueue_task(start_task);
            sched.run();

            assert_eq!(count, MAX);

            fn run_task(count_ptr: *mut int) {
                do Local::borrow::<Scheduler, ()> |sched| {
                    let task = ~do Coroutine::new_root(&mut sched.stack_pool) {
                        unsafe {
                            *count_ptr = *count_ptr + 1;
                            if *count_ptr != MAX {
                                run_task(count_ptr);
                            }
                        }
                    };
                    sched.enqueue_task(task);
                }
            };
        }
    }

    #[test]
    fn test_block_task() {
        do run_in_bare_thread {
            let mut sched = ~new_test_uv_sched();
            let task = ~do Coroutine::new_root(&mut sched.stack_pool) {
                let sched = Local::take::<Scheduler>();
                assert!(sched.in_task_context());
                do sched.deschedule_running_task_and_then() |sched, task| {
                    let task = Cell::new(task);
                    assert!(!sched.in_task_context());
                    sched.enqueue_task(task.take());
                }
            };
            sched.enqueue_task(task);
            sched.run();
        }
    }

    #[test]
    fn test_io_callback() {
        // This is a regression test that when there are no schedulable tasks
        // in the work queue, but we are performing I/O, that once we do put
        // something in the work queue again the scheduler picks it up and doesn't
        // exit before emptying the work queue
        do run_in_newsched_task {
            do spawn {
                let sched = Local::take::<Scheduler>();
                do sched.deschedule_running_task_and_then |sched, task| {
                    let task = Cell::new(task);
                    do sched.event_loop.callback_ms(10) {
                        rtdebug!("in callback");
                        let mut sched = Local::take::<Scheduler>();
                        sched.enqueue_task(task.take());
                        Local::put(sched);
                    }
                }
            }
        }
    }

    #[test]
    fn handle() {
        use rt::comm::*;

        do run_in_bare_thread {
            let (port, chan) = oneshot::<()>();
            let port_cell = Cell::new(port);
            let chan_cell = Cell::new(chan);
            let mut sched1 = ~new_test_uv_sched();
            let handle1 = sched1.make_handle();
            let handle1_cell = Cell::new(handle1);
            let task1 = ~do Coroutine::new_root(&mut sched1.stack_pool) {
                chan_cell.take().send(());
            };
            sched1.enqueue_task(task1);

            let mut sched2 = ~new_test_uv_sched();
            let task2 = ~do Coroutine::new_root(&mut sched2.stack_pool) {
                port_cell.take().recv();
                // Release the other scheduler's handle so it can exit
                handle1_cell.take();
            };
            sched2.enqueue_task(task2);

            let sched1_cell = Cell::new(sched1);
            let _thread1 = do Thread::start {
                let sched1 = sched1_cell.take();
                sched1.run();
            };

            let sched2_cell = Cell::new(sched2);
            let _thread2 = do Thread::start {
                let sched2 = sched2_cell.take();
                sched2.run();
            };
        }
    }

    #[test]
    fn multithreading() {
        use rt::comm::*;
        use iter::Times;
        use vec::OwnedVector;
        use container::Container;

        do run_in_mt_newsched_task {
            let mut ports = ~[];
            for 10.times {
                let (port, chan) = oneshot();
                let chan_cell = Cell::new(chan);
                do spawntask_later {
                    chan_cell.take().send(());
                }
                ports.push(port);
            }

            while !ports.is_empty() {
                ports.pop().recv();
            }
        }
    }

    #[test]
    fn thread_ring() {
        use rt::comm::*;
        use comm::{GenericPort, GenericChan};

        do run_in_mt_newsched_task {
            let (end_port, end_chan) = oneshot();

            let n_tasks = 10;
            let token = 2000;

            let mut (p, ch1) = stream();
            ch1.send((token, end_chan));
            let mut i = 2;
            while i <= n_tasks {
                let (next_p, ch) = stream();
                let imm_i = i;
                let imm_p = p;
                do spawntask_random {
                    roundtrip(imm_i, n_tasks, &imm_p, &ch);
                };
                p = next_p;
                i += 1;
            }
            let imm_p = p;
            let imm_ch = ch1;
            do spawntask_random {
                roundtrip(1, n_tasks, &imm_p, &imm_ch);
            }

            end_port.recv();
        }

        fn roundtrip(id: int, n_tasks: int,
                     p: &Port<(int, ChanOne<()>)>, ch: &Chan<(int, ChanOne<()>)>) {
            while (true) {
                match p.recv() {
                    (1, end_chan) => {
                        debug!("%d\n", id);
                        end_chan.send(());
                        return;
                    }
                    (token, end_chan) => {
                        debug!("thread: %d   got token: %d", id, token);
                        ch.send((token - 1, end_chan));
                        if token <= n_tasks {
                            return;
                        }
                    }
                }
            }
        }

    }

    #[test]
    fn start_closure_dtor() {
        use ops::Drop;

        // Regression test that the `start` task entrypoint can
        // contain dtors that use task resources
        do run_in_newsched_task {
            struct S { field: () }

            impl Drop for S {
                fn finalize(&self) {
                    let _foo = @0;
                }
            }

            let s = S { field: () };

            do spawntask {
                let _ss = &s;
            }
        }
    }

}
