// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2013-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

// no-pretty-expanded FIXME #15189

use std::iter::repeat;
use std::sync::Arc;
use std::sync::mpsc::channel;
use std::thread::Thread;

//
// Utilities.
//

// returns an infinite iterator of repeated applications of f to x,
// i.e. [x, f(x), f(f(x)), ...], as haskell iterate function.
fn iterate<T, F>(x: T, f: F) -> Iterate<T, F> where F: FnMut(&T) -> T {
    Iterate {f: f, next: x}
}
struct Iterate<T, F> where F: FnMut(&T) -> T {
    f: F,
    next: T
}
impl<T, F> Iterator for Iterate<T, F> where F: FnMut(&T) -> T {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut res = (self.f)(&self.next);
        std::mem::swap(&mut res, &mut self.next);
        Some(res)
    }
}

// a linked list using borrowed next.
enum List<'a, T:'a> {
    Nil,
    Cons(T, &'a List<'a, T>)
}
struct ListIterator<'a, T:'a> {
    cur: &'a List<'a, T>
}
impl<'a, T> List<'a, T> {
    fn iter(&'a self) -> ListIterator<'a, T> {
        ListIterator{cur: self}
    }
}
impl<'a, T> Iterator for ListIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match *self.cur {
            List::Nil => None,
            List::Cons(ref elt, next) => {
                self.cur = next;
                Some(elt)
            }
        }
    }
}

//
// preprocess
//

// Takes a pieces p on the form [(y1, x1), (y2, x2), ...] and returns
// every possible transformations (the 6 rotations with their
// corresponding mirrored piece), with, as minimum coordinates, (0,
// 0).  If all is false, only generate half of the possibilities (used
// to break the symmetry of the board).
fn transform(piece: Vec<(int, int)> , all: bool) -> Vec<Vec<(int, int)>> {
    let mut res: Vec<Vec<(int, int)>> =
        // rotations
        iterate(piece, |rot| rot.iter().map(|&(y, x)| (x + y, -y)).collect())
        .take(if all {6} else {3})
        // mirror
        .flat_map(|cur_piece| {
            iterate(cur_piece, |mir| mir.iter().map(|&(y, x)| (x, y)).collect())
            .take(2)
        }).collect();

    // translating to (0, 0) as minimum coordinates.
    for cur_piece in res.iter_mut() {
        let (dy, dx) = *cur_piece.iter().min_by(|e| *e).unwrap();
        for &(ref mut y, ref mut x) in cur_piece.iter_mut() {
            *y -= dy; *x -= dx;
        }
    }

    res
}

// A mask is a piece somewhere on the board.  It is represented as a
// u64: for i in the first 50 bits, m[i] = 1 if the cell at (i/5, i%5)
// is occupied.  m[50 + id] = 1 if the identifier of the piece is id.

// Takes a piece with minimum coordinate (0, 0) (as generated by
// transform).  Returns the corresponding mask if p translated by (dy,
// dx) is on the board.
fn mask(dy: int, dx: int, id: uint, p: &Vec<(int, int)>) -> Option<u64> {
    let mut m = 1 << (50 + id);
    for &(y, x) in p.iter() {
        let x = x + dx + (y + (dy % 2)) / 2;
        if x < 0 || x > 4 {return None;}
        let y = y + dy;
        if y < 0 || y > 9 {return None;}
        m |= 1 << (y * 5 + x) as uint;
    }
    Some(m)
}

// Makes every possible masks.  masks[i][id] correspond to every
// possible masks for piece with identifier id with minimum coordinate
// (i/5, i%5).
fn make_masks() -> Vec<Vec<Vec<u64> > > {
    let pieces = vec!(
        vec!((0i,0i),(0,1),(0,2),(0,3),(1,3)),
        vec!((0i,0i),(0,2),(0,3),(1,0),(1,1)),
        vec!((0i,0i),(0,1),(0,2),(1,2),(2,1)),
        vec!((0i,0i),(0,1),(0,2),(1,1),(2,1)),
        vec!((0i,0i),(0,2),(1,0),(1,1),(2,1)),
        vec!((0i,0i),(0,1),(0,2),(1,1),(1,2)),
        vec!((0i,0i),(0,1),(1,1),(1,2),(2,1)),
        vec!((0i,0i),(0,1),(0,2),(1,0),(1,2)),
        vec!((0i,0i),(0,1),(0,2),(1,2),(1,3)),
        vec!((0i,0i),(0,1),(0,2),(0,3),(1,2)));

    // To break the central symmetry of the problem, every
    // transformation must be taken except for one piece (piece 3
    // here).
    let transforms: Vec<Vec<Vec<(int, int)>>> =
        pieces.into_iter().enumerate()
        .map(|(id, p)| transform(p, id != 3))
        .collect();

    range(0i, 50).map(|yx| {
        transforms.iter().enumerate().map(|(id, t)| {
            t.iter().filter_map(|p| mask(yx / 5, yx % 5, id, p)).collect()
        }).collect()
    }).collect()
}

// Check if all coordinates can be covered by an unused piece and that
// all unused piece can be placed on the board.
fn is_board_unfeasible(board: u64, masks: &Vec<Vec<Vec<u64>>>) -> bool {
    let mut coverable = board;
    for (i, masks_at) in masks.iter().enumerate() {
        if board & 1 << i != 0 { continue; }
        for (cur_id, pos_masks) in masks_at.iter().enumerate() {
            if board & 1 << (50 + cur_id) != 0 { continue; }
            for &cur_m in pos_masks.iter() {
                if cur_m & board != 0 { continue; }
                coverable |= cur_m;
                // if every coordinates can be covered and every
                // piece can be used.
                if coverable == (1 << 60) - 1 { return false; }
            }
        }
        if coverable & 1 << i == 0 { return true; }
    }
    true
}

// Filter the masks that we can prove to result to unfeasible board.
fn filter_masks(masks: &mut Vec<Vec<Vec<u64>>>) {
    for i in range(0, masks.len()) {
        for j in range(0, (*masks)[i].len()) {
            masks[i][j] =
                (*masks)[i][j].iter().map(|&m| m)
                .filter(|&m| !is_board_unfeasible(m, masks))
                .collect();
        }
    }
}

// Gets the identifier of a mask.
fn get_id(m: u64) -> u8 {
    for id in range(0u8, 10) {
        if m & (1 << (id + 50) as uint) != 0 {return id;}
    }
    panic!("{:016x} does not have a valid identifier", m);
}

// Converts a list of mask to a Vec<u8>.
fn to_vec(raw_sol: &List<u64>) -> Vec<u8> {
    let mut sol = repeat('.' as u8).take(50).collect::<Vec<_>>();
    for &m in raw_sol.iter() {
        let id = '0' as u8 + get_id(m);
        for i in range(0u, 50) {
            if m & 1 << i != 0 {
                sol[i] = id;
            }
        }
    }
    sol
}

// Prints a solution in Vec<u8> form.
fn print_sol(sol: &Vec<u8>) {
    for (i, c) in sol.iter().enumerate() {
        if (i) % 5 == 0 { println!(""); }
        if (i + 5) % 10 == 0 { print!(" "); }
        print!("{} ", *c as char);
    }
    println!("");
}

// The data managed during the search
struct Data {
    // Number of solution found.
    nb: int,
    // Lexicographically minimal solution found.
    min: Vec<u8>,
    // Lexicographically maximal solution found.
    max: Vec<u8>
}
impl Data {
    fn new() -> Data {
        Data {nb: 0, min: vec!(), max: vec!()}
    }
    fn reduce_from(&mut self, other: Data) {
        self.nb += other.nb;
        let Data { min: min, max: max, ..} = other;
        if min < self.min { self.min = min; }
        if max > self.max { self.max = max; }
    }
}

// Records a new found solution.  Returns false if the search must be
// stopped.
fn handle_sol(raw_sol: &List<u64>, data: &mut Data) {
    // because we break the symmetry, 2 solutions correspond to a call
    // to this method: the normal solution, and the same solution in
    // reverse order, i.e. the board rotated by half a turn.
    data.nb += 2;
    let sol1 = to_vec(raw_sol);
    let sol2: Vec<u8> = sol1.iter().rev().map(|x| *x).collect();

    if data.nb == 2 {
        data.min = sol1.clone();
        data.max = sol1.clone();
    }

    if sol1 < data.min {data.min = sol1;}
    else if sol1 > data.max {data.max = sol1;}
    if sol2 < data.min {data.min = sol2;}
    else if sol2 > data.max {data.max = sol2;}
}

fn search(
    masks: &Vec<Vec<Vec<u64>>>,
    board: u64,
    mut i: uint,
    cur: List<u64>,
    data: &mut Data)
{
    // Search for the lesser empty coordinate.
    while board & (1 << i)  != 0 && i < 50 {i += 1;}
    // the board is full: a solution is found.
    if i >= 50 {return handle_sol(&cur, data);}
    let masks_at = &masks[i];

    // for every unused piece
    for id in range(0u, 10).filter(|&id| board & (1 << (id + 50)) == 0) {
        // for each mask that fits on the board
        for m in masks_at[id].iter().filter(|&m| board & *m == 0) {
            // This check is too costly.
            //if is_board_unfeasible(board | m, masks) {continue;}
            search(masks, board | *m, i + 1, List::Cons(*m, &cur), data);
        }
    }
}

fn par_search(masks: Vec<Vec<Vec<u64>>>) -> Data {
    let masks = Arc::new(masks);
    let (tx, rx) = channel();

    // launching the search in parallel on every masks at minimum
    // coordinate (0,0)
    for m in (*masks)[0].iter().flat_map(|masks_pos| masks_pos.iter()) {
        let masks = masks.clone();
        let tx = tx.clone();
        let m = *m;
        Thread::spawn(move|| {
            let mut data = Data::new();
            search(&*masks, m, 1, List::Cons(m, &List::Nil), &mut data);
            tx.send(data).unwrap();
        });
    }

    // collecting the results
    drop(tx);
    let mut data = rx.recv().unwrap();
    for d in rx.iter() { data.reduce_from(d); }
    data
}

fn main () {
    let mut masks = make_masks();
    filter_masks(&mut masks);
    let data = par_search(masks);
    println!("{} solutions found", data.nb);
    print_sol(&data.min);
    print_sol(&data.max);
    println!("");
}
