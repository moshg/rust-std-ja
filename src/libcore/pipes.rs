/*! Runtime support for message passing with protocol enforcement.


Pipes consist of two endpoints. One endpoint can send messages and
the other can receive messages. The set of legal messages and which
directions they can flow at any given point are determined by a
protocol. Below is an example protocol.

~~~
proto! pingpong (
    ping: send {
        ping -> pong
    }
    pong: recv {
        pong -> ping
    }
)
~~~

The `proto!` syntax extension will convert this into a module called
`pingpong`, which includes a set of types and functions that can be
used to write programs that follow the pingpong protocol.

*/

/* IMPLEMENTATION NOTES

The initial design for this feature is available at:

https://github.com/eholk/rust/wiki/Proposal-for-channel-contracts

Much of the design in that document is still accurate. There are
several components for the pipe implementation. First of all is the
syntax extension. To see how that works, it is best see comments in
libsyntax/ext/pipes.rs.

This module includes two related pieces of the runtime
implementation: support for unbounded and bounded
protocols. The main difference between the two is the type of the
buffer that is carried along in the endpoint data structures.


The heart of the implementation is the packet type. It contains a
header and a payload field. Much of the code in this module deals with
the header field. This is where the synchronization information is
stored. In the case of a bounded protocol, the header also includes a
pointer to the buffer the packet is contained in.

Packets represent a single message in a protocol. The payload field
gets instatiated at the type of the message, which is usually an enum
generated by the pipe compiler. Packets are conceptually single use,
although in bounded protocols they are reused each time around the
loop.


Packets are usually handled through a send_packet_buffered or
recv_packet_buffered object. Each packet is referenced by one
send_packet and one recv_packet, and these wrappers enforce that only
one end can send and only one end can receive. The structs also
include a destructor that marks packets are terminated if the sender
or receiver destroys the object before sending or receiving a value.

The *_packet_buffered structs take two type parameters. The first is
the message type for the current packet (or state). The second
represents the type of the whole buffer. For bounded protocols, the
protocol compiler generates a struct with a field for each protocol
state. This generated struct is used as the buffer type parameter. For
unbounded protocols, the buffer is simply one packet, so there is a
shorthand struct called send_packet and recv_packet, where the buffer
type is just `packet<T>`. Using the same underlying structure for both
bounded and unbounded protocols allows for less code duplication.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

import unsafe::{forget, reinterpret_cast, transmute};
import either::{Either, Left, Right};
import option::unwrap;

// Things used by code generated by the pipe compiler.
export entangle, get_buffer, drop_buffer;
export send_packet_buffered, recv_packet_buffered;
export packet, mk_packet, entangle_buffer, has_buffer, buffer_header;

// export these so we can find them in the buffer_resource
// destructor. This is probably a symptom of #3005.
export atomic_add_acq, atomic_sub_rel;

// User-level things
export send_packet, recv_packet, send, recv, try_recv, peek;
export select, select2, selecti, select2i, selectable;
export spawn_service, spawn_service_recv;
export stream, port, chan, SharedChan, PortSet, channel;
export oneshot, chan_one, port_one;
export recv_one, try_recv_one, send_one, try_send_one;

// Functions used by the protocol compiler
export rt;

#[doc(hidden)]
const SPIN_COUNT: uint = 0;

macro_rules! move_it (
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
)

#[doc(hidden)]
enum state {
    empty,
    full,
    blocked,
    terminated
}

struct buffer_header {
    // Tracks whether this buffer needs to be freed. We can probably
    // get away with restricting it to 0 or 1, if we're careful.
    let mut ref_count: int;

    new() { self.ref_count = 0; }

    // We may want a drop, and to be careful about stringing this
    // thing along.
}

// This is for protocols to associate extra data to thread around.
#[doc(hidden)]
type buffer<T: send> = {
    header: buffer_header,
    data: T,
};

struct packet_header {
    let mut state: state;
    let mut blocked_task: *rust_task;

    // This is a reinterpret_cast of a ~buffer, that can also be cast
    // to a buffer_header if need be.
    let mut buffer: *libc::c_void;

    new() {
        self.state = empty;
        self.blocked_task = ptr::null();
        self.buffer = ptr::null();
    }

    // Returns the old state.
    unsafe fn mark_blocked(this: *rust_task) -> state {
        rustrt::rust_task_ref(this);
        let old_task = swap_task(&mut self.blocked_task, this);
        assert old_task.is_null();
        swap_state_acq(&mut self.state, blocked)
    }

    unsafe fn unblock() {
        let old_task = swap_task(&mut self.blocked_task, ptr::null());
        if !old_task.is_null() { rustrt::rust_task_deref(old_task) }
        match swap_state_acq(&mut self.state, empty) {
          empty | blocked => (),
          terminated => self.state = terminated,
          full => self.state = full
        }
    }

    // unsafe because this can do weird things to the space/time
    // continuum. It ends making multiple unique pointers to the same
    // thing. You'll proobably want to forget them when you're done.
    unsafe fn buf_header() -> ~buffer_header {
        assert self.buffer.is_not_null();
        reinterpret_cast(self.buffer)
    }

    fn set_buffer<T: send>(b: ~buffer<T>) unsafe {
        self.buffer = reinterpret_cast(b);
    }
}

#[doc(hidden)]
type packet<T: send> = {
    header: packet_header,
    mut payload: Option<T>,
};

#[doc(hidden)]
trait has_buffer {
    fn set_buffer(b: *libc::c_void);
}

impl<T: send> packet<T>: has_buffer {
    fn set_buffer(b: *libc::c_void) {
        self.header.buffer = b;
    }
}

#[doc(hidden)]
fn mk_packet<T: send>() -> packet<T> {
    {
        header: packet_header(),
        mut payload: None
    }
}

#[doc(hidden)]
fn unibuffer<T: send>() -> ~buffer<packet<T>> {
    let b = ~{
        header: buffer_header(),
        data: {
            header: packet_header(),
            mut payload: None,
        }
    };

    unsafe {
        b.data.header.buffer = reinterpret_cast(b);
    }

    b
}

#[doc(hidden)]
fn packet<T: send>() -> *packet<T> {
    let b = unibuffer();
    let p = ptr::addr_of(b.data);
    // We'll take over memory management from here.
    unsafe { forget(b) }
    p
}

#[doc(hidden)]
fn entangle_buffer<T: send, Tstart: send>(
    +buffer: ~buffer<T>,
    init: fn(*libc::c_void, x: &T) -> *packet<Tstart>)
    -> (send_packet_buffered<Tstart, T>, recv_packet_buffered<Tstart, T>)
{
    let p = init(unsafe { reinterpret_cast(buffer) }, &buffer.data);
    unsafe { forget(buffer) }
    (send_packet_buffered(p), recv_packet_buffered(p))
}

#[abi = "rust-intrinsic"]
#[doc(hidden)]
extern mod rusti {
    fn atomic_xchg(dst: &mut int, src: int) -> int;
    fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
    fn atomic_xchg_rel(dst: &mut int, src: int) -> int;

    fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
    fn atomic_xsub_rel(dst: &mut int, src: int) -> int;
}

// If I call the rusti versions directly from a polymorphic function,
// I get link errors. This is a bug that needs investigated more.
#[doc(hidden)]
fn atomic_xchng_rel(dst: &mut int, src: int) -> int {
    rusti::atomic_xchg_rel(dst, src)
}

#[doc(hidden)]
fn atomic_add_acq(dst: &mut int, src: int) -> int {
    rusti::atomic_xadd_acq(dst, src)
}

#[doc(hidden)]
fn atomic_sub_rel(dst: &mut int, src: int) -> int {
    rusti::atomic_xsub_rel(dst, src)
}

#[doc(hidden)]
fn swap_task(+dst: &mut *rust_task, src: *rust_task) -> *rust_task {
    // It might be worth making both acquire and release versions of
    // this.
    unsafe {
        transmute(rusti::atomic_xchg(transmute(dst), src as int))
    }
}

#[doc(hidden)]
type rust_task = libc::c_void;

#[doc(hidden)]
extern mod rustrt {
    #[rust_stack]
    fn rust_get_task() -> *rust_task;
    #[rust_stack]
    fn rust_task_ref(task: *rust_task);
    fn rust_task_deref(task: *rust_task);

    #[rust_stack]
    fn task_clear_event_reject(task: *rust_task);

    fn task_wait_event(this: *rust_task, killed: &mut *libc::c_void) -> bool;
    pure fn task_signal_event(target: *rust_task, event: *libc::c_void);
}

#[doc(hidden)]
fn wait_event(this: *rust_task) -> *libc::c_void {
    let mut event = ptr::null();

    let killed = rustrt::task_wait_event(this, &mut event);
    if killed && !task::failing() {
        fail ~"killed"
    }
    event
}

#[doc(hidden)]
fn swap_state_acq(+dst: &mut state, src: state) -> state {
    unsafe {
        transmute(rusti::atomic_xchg_acq(transmute(dst), src as int))
    }
}

#[doc(hidden)]
fn swap_state_rel(+dst: &mut state, src: state) -> state {
    unsafe {
        transmute(rusti::atomic_xchg_rel(transmute(dst), src as int))
    }
}

#[doc(hidden)]
unsafe fn get_buffer<T: send>(p: *packet_header) -> ~buffer<T> {
    transmute((*p).buf_header())
}

// This could probably be done with SharedMutableState to avoid move_it!().
struct buffer_resource<T: send> {
    let buffer: ~buffer<T>;
    new(+b: ~buffer<T>) {
        //let p = ptr::addr_of(*b);
        //error!("take %?", p);
        atomic_add_acq(&mut b.header.ref_count, 1);
        self.buffer = b;
    }

    drop unsafe {
        let b = move_it!(self.buffer);
        //let p = ptr::addr_of(*b);
        //error!("drop %?", p);
        let old_count = atomic_sub_rel(&mut b.header.ref_count, 1);
        //let old_count = atomic_xchng_rel(b.header.ref_count, 0);
        if old_count == 1 {
            // The new count is 0.

            // go go gadget drop glue
        }
        else {
            forget(b)
        }
    }
}

#[doc(hidden)]
fn send<T: send, Tbuffer: send>(+p: send_packet_buffered<T, Tbuffer>,
                                +payload: T) -> bool {
    let header = p.header();
    let p_ = p.unwrap();
    let p = unsafe { &*p_ };
    assert ptr::addr_of(p.header) == header;
    assert p.payload.is_none();
    p.payload <- Some(payload);
    let old_state = swap_state_rel(&mut p.header.state, full);
    match old_state {
        empty => {
            // Yay, fastpath.

            // The receiver will eventually clean this up.
            //unsafe { forget(p); }
            return true;
        }
        full => fail ~"duplicate send",
        blocked => {
            debug!("waking up task for %?", p_);
            let old_task = swap_task(&mut p.header.blocked_task, ptr::null());
            if !old_task.is_null() {
                rustrt::task_signal_event(
                    old_task, ptr::addr_of(p.header) as *libc::c_void);
                rustrt::rust_task_deref(old_task);
            }

            // The receiver will eventually clean this up.
            //unsafe { forget(p); }
            return true;
        }
        terminated => {
            // The receiver will never receive this. Rely on drop_glue
            // to clean everything up.
            return false;
        }
    }
}

/** Receives a message from a pipe.

Fails if the sender closes the connection.

*/
fn recv<T: send, Tbuffer: send>(+p: recv_packet_buffered<T, Tbuffer>) -> T {
    option::unwrap_expect(try_recv(p), "connection closed")
}

/** Attempts to receive a message from a pipe.

Returns `none` if the sender has closed the connection without sending
a message, or `Some(T)` if a message was received.

*/
fn try_recv<T: send, Tbuffer: send>(+p: recv_packet_buffered<T, Tbuffer>)
    -> Option<T>
{
    let p_ = p.unwrap();
    let p = unsafe { &*p_ };

    struct drop_state {
        p: &packet_header;

        drop {
            if task::failing() {
                self.p.state = terminated;
                let old_task = swap_task(&mut self.p.blocked_task,
                                         ptr::null());
                if !old_task.is_null() {
                    rustrt::rust_task_deref(old_task);
                }
            }
        }
    };

    let _drop_state = drop_state { p: &p.header };

    // optimistic path
    match p.header.state {
      full => {
        let mut payload = None;
        payload <-> p.payload;
        p.header.state = empty;
        return Some(option::unwrap(payload))
      },
      terminated => return None,
      _ => {}
    }

    // regular path
    let this = rustrt::rust_get_task();
    rustrt::task_clear_event_reject(this);
    rustrt::rust_task_ref(this);
    let old_task = swap_task(&mut p.header.blocked_task, this);
    assert old_task.is_null();
    let mut first = true;
    let mut count = SPIN_COUNT;
    loop {
        rustrt::task_clear_event_reject(this);
        let old_state = swap_state_acq(&mut p.header.state,
                                       blocked);
        match old_state {
          empty => {
            debug!("no data available on %?, going to sleep.", p_);
            if count == 0 {
                wait_event(this);
            }
            else {
                count -= 1;
                // FIXME (#524): Putting the yield here destroys a lot
                // of the benefit of spinning, since we still go into
                // the scheduler at every iteration. However, without
                // this everything spins too much because we end up
                // sometimes blocking the thing we are waiting on.
                task::yield();
            }
            debug!("woke up, p.state = %?", copy p.header.state);
          }
          blocked => if first {
            fail ~"blocking on already blocked packet"
          },
          full => {
            let mut payload = None;
            payload <-> p.payload;
            let old_task = swap_task(&mut p.header.blocked_task, ptr::null());
            if !old_task.is_null() {
                rustrt::rust_task_deref(old_task);
            }
            p.header.state = empty;
            return Some(option::unwrap(payload))
          }
          terminated => {
            // This assert detects when we've accidentally unsafely
            // casted too big of a number to a state.
            assert old_state == terminated;

            let old_task = swap_task(&mut p.header.blocked_task, ptr::null());
            if !old_task.is_null() {
                rustrt::rust_task_deref(old_task);
            }
            return None;
          }
        }
        first = false;
    }
}

/// Returns true if messages are available.
pure fn peek<T: send, Tb: send>(p: &recv_packet_buffered<T, Tb>) -> bool {
    match unsafe {(*p.header()).state} {
      empty => false,
      blocked => fail ~"peeking on blocked packet",
      full | terminated => true
    }
}

impl<T: send, Tb: send> recv_packet_buffered<T, Tb> {
    pure fn peek() -> bool {
        peek(&self)
    }
}

#[doc(hidden)]
fn sender_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { &*p };
    match swap_state_rel(&mut p.header.state, terminated) {
      empty => {
        // The receiver will eventually clean up.
      }
      blocked => {
        // wake up the target
        let old_task = swap_task(&mut p.header.blocked_task, ptr::null());
        if !old_task.is_null() {
            rustrt::task_signal_event(
                old_task,
                ptr::addr_of(p.header) as *libc::c_void);
            rustrt::rust_task_deref(old_task);
        }
        // The receiver will eventually clean up.
      }
      full => {
        // This is impossible
        fail ~"you dun goofed"
      }
      terminated => {
        assert p.header.blocked_task.is_null();
        // I have to clean up, use drop_glue
      }
    }
}

#[doc(hidden)]
fn receiver_terminate<T: send>(p: *packet<T>) {
    let p = unsafe { &*p };
    match swap_state_rel(&mut p.header.state, terminated) {
      empty => {
        assert p.header.blocked_task.is_null();
        // the sender will clean up
      }
      blocked => {
        let old_task = swap_task(&mut p.header.blocked_task, ptr::null());
        if !old_task.is_null() {
            rustrt::rust_task_deref(old_task);
            assert old_task == rustrt::rust_get_task();
        }
      }
      terminated | full => {
        assert p.header.blocked_task.is_null();
        // I have to clean up, use drop_glue
      }
    }
}

/** Returns when one of the packet headers reports data is available.

This function is primarily intended for building higher level waiting
functions, such as `select`, `select2`, etc.

It takes a vector slice of packet_headers and returns an index into
that vector. The index points to an endpoint that has either been
closed by the sender or has a message waiting to be received.

*/
fn wait_many<T: selectable>(pkts: &[T]) -> uint {
    let this = rustrt::rust_get_task();

    rustrt::task_clear_event_reject(this);
    let mut data_avail = false;
    let mut ready_packet = pkts.len();
    for pkts.eachi |i, p| unsafe {
        let p = unsafe { &*p.header() };
        let old = p.mark_blocked(this);
        match old {
          full | terminated => {
            data_avail = true;
            ready_packet = i;
            (*p).state = old;
            break;
          }
          blocked => fail ~"blocking on blocked packet",
          empty => ()
        }
    }

    while !data_avail {
        debug!("sleeping on %? packets", pkts.len());
        let event = wait_event(this) as *packet_header;
        let pos = vec::position(pkts, |p| p.header() == event);

        match pos {
          Some(i) => {
            ready_packet = i;
            data_avail = true;
          }
          None => debug!("ignoring spurious event, %?", event)
        }
    }

    debug!("%?", pkts[ready_packet]);

    for pkts.each |p| { unsafe{ (*p.header()).unblock()} }

    debug!("%?, %?", ready_packet, pkts[ready_packet]);

    unsafe {
        assert (*pkts[ready_packet].header()).state == full
            || (*pkts[ready_packet].header()).state == terminated;
    }

    ready_packet
}

/** Receives a message from one of two endpoints.

The return value is `left` if the first endpoint received something,
or `right` if the second endpoint receives something. In each case,
the result includes the other endpoint as well so it can be used
again. Below is an example of using `select2`.

~~~
match select2(a, b) {
  left((none, b)) {
    // endpoint a was closed.
  }
  right((a, none)) {
    // endpoint b was closed.
  }
  left((Some(_), b)) {
    // endpoint a received a message
  }
  right(a, Some(_)) {
    // endpoint b received a message.
  }
}
~~~

Sometimes messages will be available on both endpoints at once. In
this case, `select2` may return either `left` or `right`.

*/
fn select2<A: send, Ab: send, B: send, Bb: send>(
    +a: recv_packet_buffered<A, Ab>,
    +b: recv_packet_buffered<B, Bb>)
    -> Either<(Option<A>, recv_packet_buffered<B, Bb>),
              (recv_packet_buffered<A, Ab>, Option<B>)>
{
    let i = wait_many([a.header(), b.header()]/_);

    match i {
      0 => Left((try_recv(a), b)),
      1 => Right((a, try_recv(b))),
      _ => fail ~"select2 return an invalid packet"
    }
}

#[doc(hidden)]
trait selectable {
    pure fn header() -> *packet_header;
}

impl *packet_header: selectable {
    pure fn header() -> *packet_header { self }
}

/// Returns the index of an endpoint that is ready to receive.
fn selecti<T: selectable>(endpoints: &[T]) -> uint {
    wait_many(endpoints)
}

/// Returns 0 or 1 depending on which endpoint is ready to receive
fn select2i<A: selectable, B: selectable>(a: &A, b: &B) -> Either<(), ()> {
    match wait_many([a.header(), b.header()]/_) {
      0 => Left(()),
      1 => Right(()),
      _ => fail ~"wait returned unexpected index"
    }
}

/** Waits on a set of endpoints. Returns a message, its index, and a
 list of the remaining endpoints.

*/
fn select<T: send, Tb: send>(+endpoints: ~[recv_packet_buffered<T, Tb>])
    -> (uint, Option<T>, ~[recv_packet_buffered<T, Tb>])
{
    let ready = wait_many(endpoints.map(|p| p.header()));
    let mut remaining = endpoints;
    let port = vec::swap_remove(remaining, ready);
    let result = try_recv(port);
    (ready, result, remaining)
}

/** The sending end of a pipe. It can be used to send exactly one
message.

*/
type send_packet<T: send> = send_packet_buffered<T, packet<T>>;

#[doc(hidden)]
fn send_packet<T: send>(p: *packet<T>) -> send_packet<T> {
    send_packet_buffered(p)
}

struct send_packet_buffered<T: send, Tbuffer: send> {
    let mut p: Option<*packet<T>>;
    let mut buffer: Option<buffer_resource<Tbuffer>>;
    new(p: *packet<T>) {
        //debug!("take send %?", p);
        self.p = Some(p);
        unsafe {
            self.buffer = Some(
                buffer_resource(
                    get_buffer(ptr::addr_of((*p).header))));
        };
    }
    drop {
        //if self.p != none {
        //    debug!("drop send %?", option::get(self.p));
        //}
        if self.p != None {
            let mut p = None;
            p <-> self.p;
            sender_terminate(option::unwrap(p))
        }
        //unsafe { error!("send_drop: %?",
        //                if self.buffer == none {
        //                    "none"
        //                } else { "some" }); }
    }
    fn unwrap() -> *packet<T> {
        let mut p = None;
        p <-> self.p;
        option::unwrap(p)
    }

    pure fn header() -> *packet_header {
        match self.p {
          Some(packet) => unsafe {
            let packet = &*packet;
            let header = ptr::addr_of(packet.header);
            //forget(packet);
            header
          },
          None => fail ~"packet already consumed"
        }
    }

    fn reuse_buffer() -> buffer_resource<Tbuffer> {
        //error!("send reuse_buffer");
        let mut tmp = None;
        tmp <-> self.buffer;
        option::unwrap(tmp)
    }
}

/// Represents the receive end of a pipe. It can receive exactly one
/// message.
type recv_packet<T: send> = recv_packet_buffered<T, packet<T>>;

#[doc(hidden)]
fn recv_packet<T: send>(p: *packet<T>) -> recv_packet<T> {
    recv_packet_buffered(p)
}

struct recv_packet_buffered<T: send, Tbuffer: send> : selectable {
    let mut p: Option<*packet<T>>;
    let mut buffer: Option<buffer_resource<Tbuffer>>;
    new(p: *packet<T>) {
        //debug!("take recv %?", p);
        self.p = Some(p);
        unsafe {
            self.buffer = Some(
                buffer_resource(
                    get_buffer(ptr::addr_of((*p).header))));
        };
    }
    drop {
        //if self.p != none {
        //    debug!("drop recv %?", option::get(self.p));
        //}
        if self.p != None {
            let mut p = None;
            p <-> self.p;
            receiver_terminate(option::unwrap(p))
        }
        //unsafe { error!("recv_drop: %?",
        //                if self.buffer == none {
        //                    "none"
        //                } else { "some" }); }
    }
    fn unwrap() -> *packet<T> {
        let mut p = None;
        p <-> self.p;
        option::unwrap(p)
    }

    pure fn header() -> *packet_header {
        match self.p {
          Some(packet) => unsafe {
            let packet = &*packet;
            let header = ptr::addr_of(packet.header);
            //forget(packet);
            header
          },
          None => fail ~"packet already consumed"
        }
    }

    fn reuse_buffer() -> buffer_resource<Tbuffer> {
        //error!("recv reuse_buffer");
        let mut tmp = None;
        tmp <-> self.buffer;
        option::unwrap(tmp)
    }
}

#[doc(hidden)]
fn entangle<T: send>() -> (send_packet<T>, recv_packet<T>) {
    let p = packet();
    (send_packet(p), recv_packet(p))
}

/** Spawn a task to provide a service.

It takes an initialization function that produces a send and receive
endpoint. The send endpoint is returned to the caller and the receive
endpoint is passed to the new task.

*/
fn spawn_service<T: send, Tb: send>(
    init: extern fn() -> (send_packet_buffered<T, Tb>,
                          recv_packet_buffered<T, Tb>),
    +service: fn~(+recv_packet_buffered<T, Tb>))
    -> send_packet_buffered<T, Tb>
{
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = ~mut Some(server);
    do task::spawn |move service| {
        let mut server_ = None;
        server_ <-> *server;
        service(option::unwrap(server_))
    }

    client
}

/** Like `spawn_service_recv`, but for protocols that start in the
receive state.

*/
fn spawn_service_recv<T: send, Tb: send>(
    init: extern fn() -> (recv_packet_buffered<T, Tb>,
                          send_packet_buffered<T, Tb>),
    +service: fn~(+send_packet_buffered<T, Tb>))
    -> recv_packet_buffered<T, Tb>
{
    let (client, server) = init();

    // This is some nasty gymnastics required to safely move the pipe
    // into a new task.
    let server = ~mut Some(server);
    do task::spawn |move service| {
        let mut server_ = None;
        server_ <-> *server;
        service(option::unwrap(server_))
    }

    client
}

// Streams - Make pipes a little easier in general.

proto! streamp (
    open:send<T: send> {
        data(T) -> open<T>
    }
)

/// A trait for things that can send multiple messages.
trait channel<T: send> {
    // It'd be nice to call this send, but it'd conflict with the
    // built in send kind.

    /// Sends a message.
    fn send(+x: T);

    /// Sends a message, or report if the receiver has closed the connection.
    fn try_send(+x: T) -> bool;
}

/// A trait for things that can receive multiple messages.
trait recv<T: send> {
    /// Receives a message, or fails if the connection closes.
    fn recv() -> T;

    /** Receives a message if one is available, or returns `none` if
    the connection is closed.

    */
    fn try_recv() -> Option<T>;

    /** Returns true if a message is available or the connection is
    closed.

    */
    pure fn peek() -> bool;
}

#[doc(hidden)]
type chan_<T:send> = { mut endp: Option<streamp::client::open<T>> };

/// An endpoint that can send many messages.
enum chan<T:send> {
    chan_(chan_<T>)
}

#[doc(hidden)]
type port_<T:send> = { mut endp: Option<streamp::server::open<T>> };

/// An endpoint that can receive many messages.
enum port<T:send> {
    port_(port_<T>)
}

/** Creates a `(chan, port)` pair.

These allow sending or receiving an unlimited number of messages.

*/
fn stream<T:send>() -> (chan<T>, port<T>) {
    let (c, s) = streamp::init();

    (chan_({ mut endp: Some(c) }), port_({ mut endp: Some(s) }))
}

impl<T: send> chan<T>: channel<T> {
    fn send(+x: T) {
        let mut endp = None;
        endp <-> self.endp;
        self.endp = Some(
            streamp::client::data(unwrap(endp), x))
    }

    fn try_send(+x: T) -> bool {
        let mut endp = None;
        endp <-> self.endp;
        match move streamp::client::try_data(unwrap(endp), x) {
            Some(move next) => {
                self.endp = Some(next);
                true
            }
            None => false
        }
    }
}

impl<T: send> port<T>: recv<T> {
    fn recv() -> T {
        let mut endp = None;
        endp <-> self.endp;
        let streamp::data(x, endp) = pipes::recv(unwrap(endp));
        self.endp = Some(endp);
        x
    }

    fn try_recv() -> Option<T> {
        let mut endp = None;
        endp <-> self.endp;
        match move pipes::try_recv(unwrap(endp)) {
          Some(streamp::data(move x, move endp)) => {
            self.endp = Some(endp);
            Some(x)
          }
          None => None
        }
    }

    pure fn peek() -> bool unchecked {
        let mut endp = None;
        endp <-> self.endp;
        let peek = match endp {
          Some(endp) => pipes::peek(&endp),
          None => fail ~"peeking empty stream"
        };
        self.endp <-> endp;
        peek
    }
}

/// Treat many ports as one.
struct PortSet<T: send> : recv<T> {
    let mut ports: ~[pipes::port<T>];

    new() { self.ports = ~[]; }

    fn add(+port: pipes::port<T>) {
        vec::push(self.ports, port)
    }

    fn chan() -> chan<T> {
        let (ch, po) = stream();
        self.add(po);
        ch
    }

    fn try_recv() -> Option<T> {
        let mut result = None;
        // we have to swap the ports array so we aren't borrowing
        // aliasable mutable memory.
        let mut ports = ~[];
        ports <-> self.ports;
        while result == None && ports.len() > 0 {
            let i = wait_many(ports);
            match move ports[i].try_recv() {
                Some(move m) => {
                  result = Some(m);
                }
                None => {
                    // Remove this port.
                    let _ = vec::swap_remove(ports, i);
                }
            }
        }
        ports <-> self.ports;
        result
    }

    fn recv() -> T {
        option::unwrap_expect(self.try_recv(), "port_set: endpoints closed")
    }

    pure fn peek() -> bool {
        // It'd be nice to use self.port.each, but that version isn't
        // pure.
        for vec::each(self.ports) |p| {
            if p.peek() { return true }
        }
        false
    }
}

impl<T: send> port<T>: selectable {
    pure fn header() -> *packet_header unchecked {
        match self.endp {
          Some(endp) => endp.header(),
          None => fail ~"peeking empty stream"
        }
    }
}

/// A channel that can be shared between many senders.
type SharedChan<T: send> = unsafe::Exclusive<chan<T>>;

impl<T: send> SharedChan<T>: channel<T> {
    fn send(+x: T) {
        let mut xx = Some(x);
        do self.with |chan| {
            let mut x = None;
            x <-> xx;
            chan.send(option::unwrap(x))
        }
    }

    fn try_send(+x: T) -> bool {
        let mut xx = Some(x);
        do self.with |chan| {
            let mut x = None;
            x <-> xx;
            chan.try_send(option::unwrap(x))
        }
    }
}

/// Converts a `chan` into a `shared_chan`.
fn SharedChan<T:send>(+c: chan<T>) -> SharedChan<T> {
    unsafe::exclusive(c)
}

/// Receive a message from one of two endpoints.
trait select2<T: send, U: send> {
    /// Receive a message or return `none` if a connection closes.
    fn try_select() -> Either<Option<T>, Option<U>>;
    /// Receive a message or fail if a connection closes.
    fn select() -> Either<T, U>;
}

impl<T: send, U: send, Left: selectable recv<T>, Right: selectable recv<U>>
    (Left, Right): select2<T, U> {

    fn select() -> Either<T, U> {
        match self {
          (lp, rp) => match select2i(&lp, &rp) {
            Left(()) => Left (lp.recv()),
            Right(()) => Right(rp.recv())
          }
        }
    }

    fn try_select() -> Either<Option<T>, Option<U>> {
        match self {
          (lp, rp) => match select2i(&lp, &rp) {
            Left(()) => Left (lp.try_recv()),
            Right(()) => Right(rp.try_recv())
          }
        }
    }
}

proto! oneshot (
    oneshot:send<T:send> {
        send(T) -> !
    }
)

/// The send end of a oneshot pipe.
type chan_one<T: send> = oneshot::client::oneshot<T>;
/// The receive end of a oneshot pipe.
type port_one<T: send> = oneshot::server::oneshot<T>;

/// Initialiase a (send-endpoint, recv-endpoint) oneshot pipe pair.
fn oneshot<T: send>() -> (chan_one<T>, port_one<T>) {
    oneshot::init()
}

/**
 * Receive a message from a oneshot pipe, failing if the connection was
 * closed.
 */
fn recv_one<T: send>(+port: port_one<T>) -> T {
    let oneshot::send(message) = recv(port);
    message
}

/// Receive a message from a oneshot pipe unless the connection was closed.
fn try_recv_one<T: send> (+port: port_one<T>) -> Option<T> {
    let message = try_recv(port);

    if message == None { None }
    else {
        let oneshot::send(message) = option::unwrap(message);
        Some(message)
    }
}

/// Send a message on a oneshot pipe, failing if the connection was closed.
fn send_one<T: send>(+chan: chan_one<T>, +data: T) {
    oneshot::client::send(chan, data);
}

/**
 * Send a message on a oneshot pipe, or return false if the connection was
 * closed.
 */
fn try_send_one<T: send>(+chan: chan_one<T>, +data: T)
        -> bool {
    oneshot::client::try_send(chan, data).is_some()
}

mod rt {
    // These are used to hide the option constructors from the
    // compiler because their names are changing
    fn make_some<T>(+val: T) -> Option<T> { Some(val) }
    fn make_none<T>() -> Option<T> { None }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_select2() {
        let (c1, p1) = pipes::stream();
        let (c2, p2) = pipes::stream();

        c1.send(~"abc");

        match (p1, p2).select() {
          Right(_) => fail,
          _ => ()
        }

        c2.send(123);
    }

    #[test]
    fn test_oneshot() {
        let (c, p) = oneshot::init();

        oneshot::client::send(c, ());

        recv_one(p)
    }
}
