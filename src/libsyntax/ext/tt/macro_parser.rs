// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is an NFA-based parser, which calls out to the main rust parser for named nonterminals
//! (which it commits to fully when it hits one in a grammar). There's a set of current NFA threads
//! and a set of next ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
//! pathological cases, is worse than traditional use of NFA or Earley parsing, but it's an easier
//! fit for Macro-by-Example-style rules.
//!
//! (In order to prevent the pathological case, we'd need to lazily construct the resulting
//! `NamedMatch`es at the very end. It'd be a pain, and require more memory to keep around old
//! items, but it would also save overhead)
//!
//! We don't say this parser uses the Earley algorithm, because it's unnecessarily innacurate.
//! The macro parser restricts itself to the features of finite state automata. Earley parsers
//! can be described as an extension of NFAs with completion rules, prediction rules, and recursion.
//!
//! Quick intro to how the parser works:
//!
//! A 'position' is a dot in the middle of a matcher, usually represented as a
//! dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a character at a time, maintaining a list
//! of threads consistent with the current position in the input string: `cur_items`.
//!
//! As it processes them, it fills up `eof_items` with threads that would be valid if
//! the macro invocation is now over, `bb_items` with threads that are waiting on
//! a Rust nonterminal like `$e:expr`, and `next_items` with threads that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. The rules for moving the · without
//! consuming any input are called epsilon transitions. It only advances or calls
//! out to the real Rust parser when no `cur_items` threads remain.
//!
//! Example:
//!
//! ```text, ignore
//! Start parsing a a a a b against [· a $( a )* a b].
//!
//! Remaining input: a a a a b
//! next: [· a $( a )* a b]
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a a b
//! cur: [a · $( a )* a b]
//! Descend/Skip (first item).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over a b. - - -
//!
//! Remaining input: ''
//! eof: [a $( a )* a b ·]
//! ```

pub use self::NamedMatch::*;
pub use self::ParseResult::*;
use self::TokenTreeOrTokenTreeVec::*;

use ast::Ident;
use syntax_pos::{self, BytePos, Span};
use codemap::Spanned;
use errors::FatalError;
use ext::tt::quoted::{self, TokenTree};
use parse::{Directory, ParseSess};
use parse::parser::{Parser, PathStyle};
use parse::token::{self, DocComment, Nonterminal, Token};
use print::pprust;
use symbol::keywords;
use tokenstream::TokenStream;
use util::small_vector::SmallVector;

use std::mem;
use std::rc::Rc;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};

// To avoid costly uniqueness checks, we require that `MatchSeq` always has a nonempty body.

/// Either a sequence of token trees or a single one. This is used as the representation of the
/// sequence of tokens that make up a matcher.
#[derive(Clone)]
enum TokenTreeOrTokenTreeVec {
    Tt(TokenTree),
    TtSeq(Vec<TokenTree>),
}

impl TokenTreeOrTokenTreeVec {
    /// Returns the number of constituent token trees of `self`.
    fn len(&self) -> usize {
        match *self {
            TtSeq(ref v) => v.len(),
            Tt(ref tt) => tt.len(),
        }
    }

    /// The the `index`-th token tree of `self`.
    fn get_tt(&self, index: usize) -> TokenTree {
        match *self {
            TtSeq(ref v) => v[index].clone(),
            Tt(ref tt) => tt.get_tt(index),
        }
    }
}

/// An unzipping of `TokenTree`s... see the `stack` field of `MatcherPos`.
///
/// This is used by `inner_parse_loop` to keep track of delimited submatchers that we have
/// descended into.
#[derive(Clone)]
struct MatcherTtFrame {
    /// The "parent" matcher that we are descending into.
    elts: TokenTreeOrTokenTreeVec,
    /// The position of the "dot" in `elts` at the time we descended.
    idx: usize,
}

/// Represents a single "position" (aka "matcher position", aka "item"), as described in the module
/// documentation.
#[derive(Clone)]
struct MatcherPos {
    /// The token or sequence of tokens that make up the matcher
    top_elts: TokenTreeOrTokenTreeVec,
    /// The position of the "dot" in this matcher
    idx: usize,
    /// The beginning position in the source that the beginning of this matcher corresponds to. In
    /// other words, the token in the source at `sp_lo` is matched against the first token of the
    /// matcher.
    sp_lo: BytePos,

    /// For each named metavar in the matcher, we keep track of token trees matched against the
    /// metavar by the black box parser. In particular, there may be more than one match per
    /// metavar if we are in a repetition (each repetition matches each of the variables).
    /// Moreover, matchers and repetitions can be nested; the `matches` field is shared (hence the
    /// `Rc`) among all "nested" matchers. `match_lo`, `match_cur`, and `match_hi` keep track of
    /// the current position of the `self` matcher position in the shared `matches` list.
    matches: Vec<Rc<Vec<NamedMatch>>>,
    /// The position in `matches` corresponding to the first metavar in this matcher's sequence of
    /// token trees. In other words, the first metavar in the first token of `top_elts` corresponds
    /// to `matches[match_lo]`.
    match_lo: usize,
    /// The position in `matches` corresponding to the metavar we are currently trying to match
    /// against the source token stream. `match_lo <= match_cur <= match_hi`.
    match_cur: usize,
    /// Similar to `match_lo` except `match_hi` is the position in `matches` of the _last_ metavar
    /// in this matcher.
    match_hi: usize,

    // Specifically used if we are matching a repetition. If we aren't both should be `None`.
    /// The separator if we are in a repetition
    sep: Option<Token>,
    /// The "parent" matcher position if we are in a repetition. That is, the matcher position just
    /// before we enter the sequence.
    up: Option<Box<MatcherPos>>,

    // Specifically used to "unzip" token trees. By "unzip", we mean to unwrap the delimiters from
    // a delimited token tree (e.g. something wrapped in `(` `)`) or to get the contents of a doc
    // comment...
    /// When matching against matchers with nested delimited submatchers (e.g. `pat ( pat ( .. )
    /// pat ) pat`), we need to keep track of the matchers we are descending into. This stack does
    /// that where the bottom of the stack is the outermost matcher.
    // Also, throughout the comments, this "descent" is often referred to as "unzipping"...
    stack: Vec<MatcherTtFrame>,
}

impl MatcherPos {
    /// Add `m` as a named match for the `idx`-th metavar.
    fn push_match(&mut self, idx: usize, m: NamedMatch) {
        let matches = Rc::make_mut(&mut self.matches[idx]);
        matches.push(m);
    }
}

/// Represents the possible results of an attempted parse.
pub enum ParseResult<T> {
    /// Parsed successfully.
    Success(T),
    /// Arm failed to match. If the second parameter is `token::Eof`, it indicates an unexpected
    /// end of macro invocation. Otherwise, it indicates that no rules expected the given token.
    Failure(syntax_pos::Span, Token),
    /// Fatal error (malformed macro?). Abort compilation.
    Error(syntax_pos::Span, String),
}

/// A `ParseResult` where the `Success` variant contains a mapping of `Ident`s to `NamedMatch`es.
/// This represents the mapping of metavars to the token trees they bind to.
pub type NamedParseResult = ParseResult<HashMap<Ident, Rc<NamedMatch>>>;

/// Count how many metavars are named in the given matcher `ms`.
pub fn count_names(ms: &[TokenTree]) -> usize {
    ms.iter().fold(0, |count, elt| {
        count + match *elt {
            TokenTree::Sequence(_, ref seq) => seq.num_captures,
            TokenTree::Delimited(_, ref delim) => count_names(&delim.tts),
            TokenTree::MetaVar(..) => 0,
            TokenTree::MetaVarDecl(..) => 1,
            TokenTree::Token(..) => 0,
        }
    })
}

/// Initialize `len` empty shared `Vec`s to be used to store matches of metavars.
fn create_matches(len: usize) -> Vec<Rc<Vec<NamedMatch>>> {
    (0..len).into_iter().map(|_| Rc::new(Vec::new())).collect()
}

/// Generate the top-level matcher position in which the "dot" is before the first token of the
/// matcher `ms` and we are going to start matching at position `lo` in the source.
fn initial_matcher_pos(ms: Vec<TokenTree>, lo: BytePos) -> Box<MatcherPos> {
    let match_idx_hi = count_names(&ms[..]);
    let matches = create_matches(match_idx_hi);
    Box::new(MatcherPos {
        // Start with the top level matcher given to us
        top_elts: TtSeq(ms), // "elts" is an abbr. for "elements"
        // The "dot" is before the first token of the matcher
        idx: 0,
        // We start matching with byte `lo` in the source code
        sp_lo: lo,

        // Initialize `matches` to a bunch of empty `Vec`s -- one for each metavar in `top_elts`.
        // `match_lo` for `top_elts` is 0 and `match_hi` is `matches.len()`. `match_cur` is 0 since
        // we haven't actually matched anything yet.
        matches,
        match_lo: 0,
        match_cur: 0,
        match_hi: match_idx_hi,

        // Haven't descended into any delimiters, so empty stack
        stack: vec![],

        // Haven't descended into any sequences, so both of these are `None`
        sep: None,
        up: None,
    })
}

/// `NamedMatch` is a pattern-match result for a single `token::MATCH_NONTERMINAL`:
/// so it is associated with a single ident in a parse, and all
/// `MatchedNonterminal`s in the `NamedMatch` have the same nonterminal type
/// (expr, item, etc). Each leaf in a single `NamedMatch` corresponds to a
/// single `token::MATCH_NONTERMINAL` in the `TokenTree` that produced it.
///
/// The in-memory structure of a particular `NamedMatch` represents the match
/// that occurred when a particular subset of a matcher was applied to a
/// particular token tree.
///
/// The width of each `MatchedSeq` in the `NamedMatch`, and the identity of
/// the `MatchedNonterminal`s, will depend on the token tree it was applied
/// to: each `MatchedSeq` corresponds to a single `TTSeq` in the originating
/// token tree. The depth of the `NamedMatch` structure will therefore depend
/// only on the nesting depth of `ast::TTSeq`s in the originating
/// token tree it was derived from.
#[derive(Debug, Clone)]
pub enum NamedMatch {
    MatchedSeq(Rc<Vec<NamedMatch>>, syntax_pos::Span),
    MatchedNonterminal(Rc<Nonterminal>),
}

fn nameize<I: Iterator<Item = NamedMatch>>(
    sess: &ParseSess,
    ms: &[TokenTree],
    mut res: I,
) -> NamedParseResult {
    fn n_rec<I: Iterator<Item = NamedMatch>>(
        sess: &ParseSess,
        m: &TokenTree,
        res: &mut I,
        ret_val: &mut HashMap<Ident, Rc<NamedMatch>>,
    ) -> Result<(), (syntax_pos::Span, String)> {
        match *m {
            TokenTree::Sequence(_, ref seq) => for next_m in &seq.tts {
                n_rec(sess, next_m, res.by_ref(), ret_val)?
            },
            TokenTree::Delimited(_, ref delim) => for next_m in &delim.tts {
                n_rec(sess, next_m, res.by_ref(), ret_val)?;
            },
            TokenTree::MetaVarDecl(span, _, id) if id.name == keywords::Invalid.name() => {
                if sess.missing_fragment_specifiers.borrow_mut().remove(&span) {
                    return Err((span, "missing fragment specifier".to_string()));
                }
            }
            TokenTree::MetaVarDecl(sp, bind_name, _) => {
                match ret_val.entry(bind_name) {
                    Vacant(spot) => {
                        // FIXME(simulacrum): Don't construct Rc here
                        spot.insert(Rc::new(res.next().unwrap()));
                    }
                    Occupied(..) => {
                        return Err((sp, format!("duplicated bind name: {}", bind_name)))
                    }
                }
            }
            TokenTree::MetaVar(..) | TokenTree::Token(..) => (),
        }

        Ok(())
    }

    let mut ret_val = HashMap::new();
    for m in ms {
        match n_rec(sess, m, res.by_ref(), &mut ret_val) {
            Ok(_) => {}
            Err((sp, msg)) => return Error(sp, msg),
        }
    }

    Success(ret_val)
}

pub fn parse_failure_msg(tok: Token) -> String {
    match tok {
        token::Eof => "unexpected end of macro invocation".to_string(),
        _ => format!(
            "no rules expected the token `{}`",
            pprust::token_to_string(&tok)
        ),
    }
}

/// Perform a token equality check, ignoring syntax context (that is, an unhygienic comparison)
fn token_name_eq(t1: &Token, t2: &Token) -> bool {
    if let (Some(id1), Some(id2)) = (t1.ident(), t2.ident()) {
        id1.name == id2.name
    } else if let (&token::Lifetime(id1), &token::Lifetime(id2)) = (t1, t2) {
        id1.name == id2.name
    } else {
        *t1 == *t2
    }
}

fn inner_parse_loop(
    sess: &ParseSess,
    cur_items: &mut SmallVector<Box<MatcherPos>>,
    next_items: &mut Vec<Box<MatcherPos>>,
    eof_items: &mut SmallVector<Box<MatcherPos>>,
    bb_items: &mut SmallVector<Box<MatcherPos>>,
    token: &Token,
    span: syntax_pos::Span,
) -> ParseResult<()> {
    while let Some(mut item) = cur_items.pop() {
        // When unzipped trees end, remove them
        while item.idx >= item.top_elts.len() {
            match item.stack.pop() {
                Some(MatcherTtFrame { elts, idx }) => {
                    item.top_elts = elts;
                    item.idx = idx + 1;
                }
                None => break,
            }
        }

        let idx = item.idx;
        let len = item.top_elts.len();

        // at end of sequence
        if idx >= len {
            // We are repeating iff there is a parent
            if item.up.is_some() {
                // Disregarding the separator, add the "up" case to the tokens that should be
                // examined.
                // (remove this condition to make trailing seps ok)
                if idx == len {
                    let mut new_pos = item.up.clone().unwrap();

                    // update matches (the MBE "parse tree") by appending
                    // each tree as a subtree.

                    // Only touch the binders we have actually bound
                    for idx in item.match_lo..item.match_hi {
                        let sub = item.matches[idx].clone();
                        let span = span.with_lo(item.sp_lo);
                        new_pos.push_match(idx, MatchedSeq(sub, span));
                    }

                    new_pos.match_cur = item.match_hi;
                    new_pos.idx += 1;
                    cur_items.push(new_pos);
                }

                // Check if we need a separator
                if idx == len && item.sep.is_some() {
                    // We have a separator, and it is the current token.
                    if item.sep
                        .as_ref()
                        .map(|sep| token_name_eq(token, sep))
                        .unwrap_or(false)
                    {
                        item.idx += 1;
                        next_items.push(item);
                    }
                } else {
                    // we don't need a separator
                    item.match_cur = item.match_lo;
                    item.idx = 0;
                    cur_items.push(item);
                }
            } else {
                // We aren't repeating, so we must be potentially at the end of the input.
                eof_items.push(item);
            }
        } else {
            match item.top_elts.get_tt(idx) {
                /* need to descend into sequence */
                TokenTree::Sequence(sp, seq) => {
                    if seq.op == quoted::KleeneOp::ZeroOrMore {
                        // Examine the case where there are 0 matches of this sequence
                        let mut new_item = item.clone();
                        new_item.match_cur += seq.num_captures;
                        new_item.idx += 1;
                        for idx in item.match_cur..item.match_cur + seq.num_captures {
                            new_item.push_match(idx, MatchedSeq(Rc::new(vec![]), sp));
                        }
                        cur_items.push(new_item);
                    }

                    // Examine the case where there is at least one match of this sequence
                    let matches = create_matches(item.matches.len());
                    cur_items.push(Box::new(MatcherPos {
                        stack: vec![],
                        sep: seq.separator.clone(),
                        idx: 0,
                        matches,
                        match_lo: item.match_cur,
                        match_cur: item.match_cur,
                        match_hi: item.match_cur + seq.num_captures,
                        up: Some(item),
                        sp_lo: sp.lo(),
                        top_elts: Tt(TokenTree::Sequence(sp, seq)),
                    }));
                }
                TokenTree::MetaVarDecl(span, _, id) if id.name == keywords::Invalid.name() => {
                    if sess.missing_fragment_specifiers.borrow_mut().remove(&span) {
                        return Error(span, "missing fragment specifier".to_string());
                    }
                }
                TokenTree::MetaVarDecl(_, _, id) => {
                    // Built-in nonterminals never start with these tokens,
                    // so we can eliminate them from consideration.
                    if may_begin_with(&*id.name.as_str(), token) {
                        bb_items.push(item);
                    }
                }
                seq @ TokenTree::Delimited(..) | seq @ TokenTree::Token(_, DocComment(..)) => {
                    let lower_elts = mem::replace(&mut item.top_elts, Tt(seq));
                    let idx = item.idx;
                    item.stack.push(MatcherTtFrame {
                        elts: lower_elts,
                        idx,
                    });
                    item.idx = 0;
                    cur_items.push(item);
                }
                TokenTree::Token(_, ref t) if token_name_eq(t, token) => {
                    item.idx += 1;
                    next_items.push(item);
                }
                TokenTree::Token(..) | TokenTree::MetaVar(..) => {}
            }
        }
    }

    Success(())
}

/// Use the given sequence of token trees (`ms`) as a matcher. Match the given token stream `tts`
/// against it and return the match.
///
/// # Parameters
///
/// - `sess`: The session into which errors are emitted
/// - `tts`: The tokenstream we are matching against the pattern `ms`
/// - `ms`: A sequence of token trees representing a pattern against which we are matching
/// - `directory`: Information about the file locations (needed for the black-box parser)
/// - `recurse_into_modules`: Whether or not to recurse into modules (needed for the black-box
///   parser)
pub fn parse(
    sess: &ParseSess,
    tts: TokenStream,
    ms: &[TokenTree],
    directory: Option<Directory>,
    recurse_into_modules: bool,
) -> NamedParseResult {
    // Create a parser that can be used for the "black box" parts.
    let mut parser = Parser::new(sess, tts, directory, recurse_into_modules, true);

    // A queue of possible matcher positions. We initialize it with the matcher position in which
    // the "dot" is before the first token of the first token tree in `ms`. `inner_parse_loop` then
    // processes all of these possible matcher positions and produces posible next positions into
    // `next_items`. After some post-processing, the contents of `next_items` replenish `cur_items`
    // and we start over again.
    let mut cur_items = SmallVector::one(initial_matcher_pos(ms.to_owned(), parser.span.lo()));
    let mut next_items = Vec::new();

    loop {
        // Matcher positions black-box parsed by parser.rs (`parser`)
        let mut bb_items = SmallVector::new();

        // Matcher positions that would be valid if the macro invocation was over now
        let mut eof_items = SmallVector::new();
        assert!(next_items.is_empty());

        // Process `cur_items` until either we have finished the input or we need to get some
        // parsing from the black-box parser done. The result is that `next_items` will contain a
        // bunch of possible next matcher positions in `next_items`.
        match inner_parse_loop(
            sess,
            &mut cur_items,
            &mut next_items,
            &mut eof_items,
            &mut bb_items,
            &parser.token,
            parser.span,
        ) {
            Success(_) => {}
            Failure(sp, tok) => return Failure(sp, tok),
            Error(sp, msg) => return Error(sp, msg),
        }

        // inner parse loop handled all cur_items, so it's empty
        assert!(cur_items.is_empty());

        // We need to do some post processing after the `inner_parser_loop`.
        //
        // Error messages here could be improved with links to original rules.

        // If we reached the EOF, check that there is EXACTLY ONE possible matcher. Otherwise,
        // either the parse is ambiguous (which should never happen) or their is a syntax error.
        if token_name_eq(&parser.token, &token::Eof) {
            if eof_items.len() == 1 {
                let matches = eof_items[0]
                    .matches
                    .iter_mut()
                    .map(|dv| Rc::make_mut(dv).pop().unwrap());
                return nameize(sess, ms, matches);
            } else if eof_items.len() > 1 {
                return Error(
                    parser.span,
                    "ambiguity: multiple successful parses".to_string(),
                );
            } else {
                return Failure(parser.span, token::Eof);
            }
        }
        // Another possibility is that we need to call out to parse some rust nonterminal
        // (black-box) parser. However, if there is not EXACTLY ONE of these, something is wrong.
        else if (!bb_items.is_empty() && !next_items.is_empty()) || bb_items.len() > 1 {
            let nts = bb_items
                .iter()
                .map(|item| match item.top_elts.get_tt(item.idx) {
                    TokenTree::MetaVarDecl(_, bind, name) => format!("{} ('{}')", name, bind),
                    _ => panic!(),
                })
                .collect::<Vec<String>>()
                .join(" or ");

            return Error(
                parser.span,
                format!(
                    "local ambiguity: multiple parsing options: {}",
                    match next_items.len() {
                        0 => format!("built-in NTs {}.", nts),
                        1 => format!("built-in NTs {} or 1 other option.", nts),
                        n => format!("built-in NTs {} or {} other options.", nts, n),
                    }
                ),
            );
        }
        // If there are no posible next positions AND we aren't waiting for the black-box parser,
        // then their is a syntax error.
        else if bb_items.is_empty() && next_items.is_empty() {
            return Failure(parser.span, parser.token);
        }
        // Dump all possible `next_items` into `cur_items` for the next iteration.
        else if !next_items.is_empty() {
            // Now process the next token
            cur_items.extend(next_items.drain(..));
            parser.bump();
        }
        // Finally, we have the case where we need to call the black-box parser to get some
        // nonterminal.
        else {
            assert_eq!(bb_items.len(), 1);

            let mut item = bb_items.pop().unwrap();
            if let TokenTree::MetaVarDecl(span, _, ident) = item.top_elts.get_tt(item.idx) {
                let match_cur = item.match_cur;
                item.push_match(
                    match_cur,
                    MatchedNonterminal(Rc::new(parse_nt(&mut parser, span, &ident.name.as_str()))),
                );
                item.idx += 1;
                item.match_cur += 1;
            } else {
                unreachable!()
            }
            cur_items.push(item);
        }

        assert!(!cur_items.is_empty());
    }
}

/// Checks whether a non-terminal may begin with a particular token.
///
/// Returning `false` is a *stability guarantee* that such a matcher will *never* begin with that
/// token. Be conservative (return true) if not sure.
fn may_begin_with(name: &str, token: &Token) -> bool {
    /// Checks whether the non-terminal may contain a single (non-keyword) identifier.
    fn may_be_ident(nt: &token::Nonterminal) -> bool {
        match *nt {
            token::NtItem(_) | token::NtBlock(_) | token::NtVis(_) => false,
            _ => true,
        }
    }

    match name {
        "expr" => token.can_begin_expr(),
        "ty" => token.can_begin_type(),
        "ident" => token.is_ident(),
        "vis" => match *token {
            // The follow-set of :vis + "priv" keyword + interpolated
            Token::Comma | Token::Ident(_) | Token::Interpolated(_) => true,
            _ => token.can_begin_type(),
        },
        "block" => match *token {
            Token::OpenDelim(token::Brace) => true,
            Token::Interpolated(ref nt) => match nt.0 {
                token::NtItem(_)
                | token::NtPat(_)
                | token::NtTy(_)
                | token::NtIdent(_)
                | token::NtMeta(_)
                | token::NtPath(_)
                | token::NtVis(_) => false, // none of these may start with '{'.
                _ => true,
            },
            _ => false,
        },
        "path" | "meta" => match *token {
            Token::ModSep | Token::Ident(_) => true,
            Token::Interpolated(ref nt) => match nt.0 {
                token::NtPath(_) | token::NtMeta(_) => true,
                _ => may_be_ident(&nt.0),
            },
            _ => false,
        },
        "pat" => match *token {
            Token::Ident(_) |               // box, ref, mut, and other identifiers (can stricten)
            Token::OpenDelim(token::Paren) |    // tuple pattern
            Token::OpenDelim(token::Bracket) |  // slice pattern
            Token::BinOp(token::And) |          // reference
            Token::BinOp(token::Minus) |        // negative literal
            Token::AndAnd |                     // double reference
            Token::Literal(..) |                // literal
            Token::DotDot |                     // range pattern (future compat)
            Token::DotDotDot |                  // range pattern (future compat)
            Token::ModSep |                     // path
            Token::Lt |                         // path (UFCS constant)
            Token::BinOp(token::Shl) |          // path (double UFCS)
            Token::Underscore => true,          // placeholder
            Token::Interpolated(ref nt) => may_be_ident(&nt.0),
            _ => false,
        },
        _ => match *token {
            token::CloseDelim(_) => false,
            _ => true,
        },
    }
}

/// A call to the "black-box" parser to parse some rust nonterminal.
///
/// # Parameters
///
/// - `p`: the "black-box" parser to use
/// - `sp`: the `Span` we want to parse
/// - `name`: the name of the metavar _matcher_ we want to match (e.g. `tt`, `ident`, `block`,
///   etc...)
///
/// # Returns
///
/// The parsed nonterminal.
fn parse_nt<'a>(p: &mut Parser<'a>, sp: Span, name: &str) -> Nonterminal {
    if name == "tt" {
        return token::NtTT(p.parse_token_tree());
    }
    // check at the beginning and the parser checks after each bump
    p.process_potential_macro_variable();
    match name {
        "item" => match panictry!(p.parse_item()) {
            Some(i) => token::NtItem(i),
            None => {
                p.fatal("expected an item keyword").emit();
                FatalError.raise();
            }
        },
        "block" => token::NtBlock(panictry!(p.parse_block())),
        "stmt" => match panictry!(p.parse_stmt()) {
            Some(s) => token::NtStmt(s),
            None => {
                p.fatal("expected a statement").emit();
                FatalError.raise();
            }
        },
        "pat" => token::NtPat(panictry!(p.parse_pat())),
        "expr" => token::NtExpr(panictry!(p.parse_expr())),
        "ty" => token::NtTy(panictry!(p.parse_ty())),
        // this could be handled like a token, since it is one
        "ident" => match p.token {
            token::Ident(sn) => {
                p.bump();
                token::NtIdent(Spanned::<Ident> {
                    node: sn,
                    span: p.prev_span,
                })
            }
            _ => {
                let token_str = pprust::token_to_string(&p.token);
                p.fatal(&format!("expected ident, found {}", &token_str[..]))
                    .emit();
                FatalError.raise()
            }
        },
        "path" => token::NtPath(panictry!(p.parse_path_common(PathStyle::Type, false))),
        "meta" => token::NtMeta(panictry!(p.parse_meta_item())),
        "vis" => token::NtVis(panictry!(p.parse_visibility(true))),
        "lifetime" => token::NtLifetime(p.expect_lifetime()),
        // this is not supposed to happen, since it has been checked
        // when compiling the macro.
        _ => p.span_bug(sp, "invalid fragment specifier"),
    }
}
