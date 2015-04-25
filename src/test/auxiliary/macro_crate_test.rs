// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar, quote)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;
extern crate rustc;

use syntax::ast::{self, TokenTree, Item, MetaItem};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::parse::{self, token};
use syntax::ptr::P;
use rustc::plugin::Registry;

#[macro_export]
macro_rules! exported_macro { () => (2) }

macro_rules! unexported_macro { () => (3) }

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("make_a_1", expand_make_a_1);
    reg.register_macro("forged_ident", expand_forged_ident);
    reg.register_macro("identity", expand_identity);
    reg.register_syntax_extension(
        token::intern("into_foo"),
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        Modifier(Box::new(expand_into_foo)));
    reg.register_syntax_extension(
        token::intern("into_multi_foo"),
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        MultiModifier(Box::new(expand_into_foo_multi)));
    reg.register_syntax_extension(
        token::intern("duplicate"),
        MultiDecorator(box expand_duplicate));
}

fn expand_make_a_1(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree])
                   -> Box<MacResult+'static> {
    if !tts.is_empty() {
        cx.span_fatal(sp, "make_a_1 takes no arguments");
    }
    MacEager::expr(quote_expr!(cx, 1))
}

// See Issue #15750
fn expand_identity(cx: &mut ExtCtxt, _span: Span, tts: &[TokenTree])
                   -> Box<MacResult+'static> {
    // Parse an expression and emit it unchanged.
    let mut parser = parse::new_parser_from_tts(cx.parse_sess(),
        cx.cfg(), tts.to_vec());
    let expr = parser.parse_expr();
    MacEager::expr(quote_expr!(&mut *cx, $expr))
}

fn expand_into_foo(cx: &mut ExtCtxt, sp: Span, attr: &MetaItem, it: P<Item>)
                   -> P<Item> {
    P(Item {
        attrs: it.attrs.clone(),
        ..(*quote_item!(cx, enum Foo { Bar, Baz }).unwrap()).clone()
    })
}

fn expand_into_foo_multi(cx: &mut ExtCtxt,
                         sp: Span,
                         attr: &MetaItem,
                         it: Annotatable) -> Annotatable {
    match it {
        Annotatable::Item(it) => {
            Annotatable::Item(P(Item {
                attrs: it.attrs.clone(),
                ..(*quote_item!(cx, enum Foo2 { Bar2, Baz2 }).unwrap()).clone()
            }))
        }
        Annotatable::ImplItem(it) => {
            quote_item!(cx, impl X { fn foo(&self) -> i32 { 42 } }).unwrap().and_then(|i| {
                match i.node {
                    ast::ItemImpl(_, _, _, _, _, mut items) => {
                        Annotatable::ImplItem(items.pop().expect("impl method not found"))
                    }
                    _ => unreachable!("impl parsed to something other than impl")
                }
            })
        }
        Annotatable::TraitItem(it) => {
            quote_item!(cx, trait X { fn foo(&self) -> i32 { 0 } }).unwrap().and_then(|i| {
                match i.node {
                    ast::ItemTrait(_, _, _, mut items) => {
                        Annotatable::TraitItem(items.pop().expect("trait method not found"))
                    }
                    _ => unreachable!("trait parsed to something other than trait")
                }
            })
        }
    }
}

// Create a duplicate of the annotatable, based on the MetaItem
fn expand_duplicate(cx: &mut ExtCtxt,
                    sp: Span,
                    mi: &MetaItem,
                    it: &Annotatable,
                    mut push: Box<FnMut(Annotatable)>)
{
    let copy_name = match mi.node {
        ast::MetaItem_::MetaList(_, ref xs) => {
            if let ast::MetaItem_::MetaWord(ref w) = xs[0].node {
                token::str_to_ident(w.get())
            } else {
                cx.span_err(mi.span, "Expected word");
                return;
            }
        }
        _ => {
            cx.span_err(mi.span, "Expected list");
            return;
        }
    };

    // Duplicate the item but replace its ident by the MetaItem
    match it.clone() {
        Annotatable::Item(it) => {
            let mut new_it = (*it).clone();
            new_it.attrs.clear();
            new_it.ident = copy_name;
            push(Annotatable::Item(P(new_it)));
        }
        Annotatable::ImplItem(it) => {
            match it {
                ImplItem::MethodImplItem(m) => {
                    let mut new_m = (*m).clone();
                    new_m.attrs.clear();
                    replace_method_name(&mut new_m.node, copy_name);
                    push(Annotatable::ImplItem(ImplItem::MethodImplItem(P(new_m))));
                }
                ImplItem::TypeImplItem(t) => {
                    let mut new_t = (*t).clone();
                    new_t.attrs.clear();
                    new_t.ident = copy_name;
                    push(Annotatable::ImplItem(ImplItem::TypeImplItem(P(new_t))));
                }
            }
        }
        Annotatable::TraitItem(it) => {
            match it {
                TraitItem::RequiredMethod(rm) => {
                    let mut new_rm = rm.clone();
                    new_rm.attrs.clear();
                    new_rm.ident = copy_name;
                    push(Annotatable::TraitItem(TraitItem::RequiredMethod(new_rm)));
                }
                TraitItem::ProvidedMethod(pm) => {
                    let mut new_pm = (*pm).clone();
                    new_pm.attrs.clear();
                    replace_method_name(&mut new_pm.node, copy_name);
                    push(Annotatable::TraitItem(TraitItem::ProvidedMethod(P(new_pm))));
                }
                TraitItem::TypeTraitItem(t) => {
                    let mut new_t = (*t).clone();
                    new_t.attrs.clear();
                    new_t.ty_param.ident = copy_name;
                    push(Annotatable::TraitItem(TraitItem::TypeTraitItem(P(new_t))));
                }
            }
        }
    }

    fn replace_method_name(m: &mut ast::Method_, i: ast::Ident) {
        if let &mut ast::Method_::MethDecl(ref mut ident, _, _, _, _, _, _, _) = m {
            *ident = i
        }
    }
}

fn expand_forged_ident(cx: &mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> Box<MacResult+'static> {
    use syntax::ext::quote::rt::*;

    if !tts.is_empty() {
        cx.span_fatal(sp, "forged_ident takes no arguments");
    }

    // Most of this is modelled after the expansion of the `quote_expr!`
    // macro ...
    let parse_sess = cx.parse_sess();
    let cfg = cx.cfg();

    // ... except this is where we inject a forged identifier,
    // and deliberately do not call `cx.parse_tts_with_hygiene`
    // (because we are testing that this will be *rejected*
    //  by the default parser).

    let expr = {
        let tt = cx.parse_tts("\x00name_2,ctxt_0\x00".to_string());
        let mut parser = new_parser_from_tts(parse_sess, cfg, tt);
        parser.parse_expr()
    };
    MacEager::expr(expr)
}

pub fn foo() {}
