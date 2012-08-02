import option::*;
import pat_util::*;
import syntax::ast::*;
import syntax::ast_util::*;
import syntax::{visit, codemap};
import codemap::span;
import std::map::{hashmap, int_hash};
import syntax::print::pprust::path_to_str;
import tstate::ann::{pre_and_post, pre_and_post_state, empty_ann, prestate,
                     poststate, precond, postcond,
                     set_prestate, set_poststate, set_in_poststate_,
                     extend_prestate, extend_poststate, set_precondition,
                     set_postcondition, ts_ann,
                     clear_in_postcond,
                     clear_in_poststate_};
import driver::session::session;
import dvec::{dvec, extensions};
import tritv::{trit, tfalse, ttrue, dont_care, t};

import syntax::print::pprust::{constr_args_to_str, lit_to_str};

// Used to communicate which operands should be invalidated
// to helper functions
enum oper_type {
    oper_move,
    oper_swap,
    oper_assign,
    oper_assign_op,
    oper_pure,
}

/* logging funs */
fn def_id_to_str(d: def_id) -> ~str {
    return int::str(d.crate) + ~"," + int::str(d.node);
}

fn comma_str(args: ~[@constr_arg_use]) -> ~str {
    let mut rslt = ~"";
    let mut comma = false;
    for args.each |a| {
        if comma { rslt += ~", "; } else { comma = true; }
        alt a.node {
          carg_base { rslt += ~"*"; }
          carg_ident(i) { rslt += *i.ident; }
          carg_lit(l) { rslt += lit_to_str(l); }
        }
    }
    return rslt;
}

fn constraint_to_str(tcx: ty::ctxt, c: sp_constr) -> ~str {
    return fmt!{"%s(%s) - arising from %s",
             path_to_str(c.node.path),
             comma_str(c.node.args),
             codemap::span_to_str(c.span, tcx.sess.codemap)};
}

fn tritv_to_str(fcx: fn_ctxt, v: tritv::t) -> ~str {
    let mut s = ~"";
    let mut comma = false;
    for constraints(fcx).each |p| {
        alt v.get(p.bit_num) {
          dont_care { }
          tt {
            s +=
                if comma { ~", " } else { comma = true; ~"" } +
                    if tt == tfalse { ~"!" } else { ~"" } +
                    constraint_to_str(fcx.ccx.tcx, p.c);
          }
        }
    }
    return s;
}

fn log_tritv(fcx: fn_ctxt, v: tritv::t) {
    log(debug, tritv_to_str(fcx, v));
}

fn first_difference_string(fcx: fn_ctxt, expected: tritv::t, actual: tritv::t)
   -> ~str {
    let mut s = ~"";
    for constraints(fcx).each |c| {
      if expected.get(c.bit_num) == ttrue &&
         actual.get(c.bit_num) != ttrue {
            s = constraint_to_str(fcx.ccx.tcx, c.c);
            break;
        }
    }
    return s;
}

fn log_tritv_err(fcx: fn_ctxt, v: tritv::t) {
    log(error, tritv_to_str(fcx, v));
}

fn tos(v: ~[uint]) -> ~str {
    let mut rslt = ~"";
    for v.each |i| {
        if i == 0u {
            rslt += ~"0";
        } else if i == 1u { rslt += ~"1"; } else { rslt += ~"?"; }
    }
    return rslt;
}

fn log_cond(v: ~[uint]) { log(debug, tos(v)); }

fn log_cond_err(v: ~[uint]) { log(error, tos(v)); }

fn log_pp(pp: pre_and_post) {
    let p1 = pp.precondition.to_vec();
    let p2 = pp.postcondition.to_vec();
    debug!{"pre:"};
    log_cond(p1);
    debug!{"post:"};
    log_cond(p2);
}

fn log_pp_err(pp: pre_and_post) {
    let p1 = pp.precondition.to_vec();
    let p2 = pp.postcondition.to_vec();
    error!{"pre:"};
    log_cond_err(p1);
    error!{"post:"};
    log_cond_err(p2);
}

fn log_states(pp: pre_and_post_state) {
    let p1 = pp.prestate.to_vec();
    let p2 = pp.poststate.to_vec();
    debug!{"prestate:"};
    log_cond(p1);
    debug!{"poststate:"};
    log_cond(p2);
}

fn log_states_err(pp: pre_and_post_state) {
    let p1 = pp.prestate.to_vec();
    let p2 = pp.poststate.to_vec();
    error!{"prestate:"};
    log_cond_err(p1);
    error!{"poststate:"};
    log_cond_err(p2);
}

fn print_ident(i: ident) { log(debug, ~" " + *i + ~" "); }

fn print_idents(&idents: ~[ident]) {
    if vec::len::<ident>(idents) == 0u { return; }
    log(debug, ~"an ident: " + *vec::pop::<ident>(idents));
    print_idents(idents);
}


/* data structures */

/**********************************************************************/

/* Two different data structures represent constraints in different
 contexts: constraint and norm_constraint.

constraint gets used to record constraints in a table keyed by def_ids.  Each
constraint has a single operator but a list of possible argument lists, and
thus represents several constraints at once, one for each possible argument
list.

norm_constraint, in contrast, gets used when handling an instance of a
constraint rather than a definition of a constraint. It has only a single
argument list.

The representation of constraints, where multiple instances of the
same predicate are collapsed into one entry in the table, makes it
easier to look up a specific instance.

Both types are in constrast with the constraint type defined in
syntax::ast, which is for predicate constraints only, and is what
gets generated by the parser. aux and ast share the same type
to represent predicate *arguments* however. This type
(constr_arg_general) is parameterized (see comments in syntax::ast).

Both types store an ident and span, for error-logging purposes.
*/
type pred_args_ = {args: ~[@constr_arg_use], bit_num: uint};

type pred_args = spanned<pred_args_>;

// The attached node ID is the *defining* node ID
// for this local.
type constr_arg_use = spanned<constr_arg_general_<inst>>;

/* Predicate constraints refer to the truth value of a predicate on variables
(definitely false, maybe true, or definitely true).  The <path> field (and the
<def_id> field in the npred constructor) names a user-defined function that
may be the operator in a "check" expression in the source.  */

type constraint = {
    path: @path,
    // FIXME (#2539): really only want it to be mut during
    // collect_locals.  freeze it after that.
    descs: @dvec<pred_args>
};

type tsconstr = {
    path: @path,
    def_id: def_id,
    args: ~[@constr_arg_use]
};

type sp_constr = spanned<tsconstr>;

type norm_constraint = {bit_num: uint, c: sp_constr};

type constr_map = std::map::hashmap<def_id, constraint>;

/* Contains stuff that has to be computed up front */
/* For easy access, the fn_info stores two special constraints for each
function.  So we need context. And so it seems clearer to just have separate
constraints. */
type fn_info =
    /* list, accumulated during pre/postcondition
    computation, of all local variables that may be
    used */
    // Doesn't seem to work without the @ -- bug
    {constrs: constr_map,
     num_constraints: uint,
     cf: ret_style,
     used_vars: @mut ~[node_id],
     ignore: bool};

/* mapping from node ID to typestate annotation */
type node_ann_table = @mut ~[mut ts_ann];


/* mapping from function name to fn_info map */
type fn_info_map = std::map::hashmap<node_id, fn_info>;

type fn_ctxt =
    {enclosing: fn_info, id: node_id, name: ident, ccx: crate_ctxt};

type crate_ctxt = {tcx: ty::ctxt, node_anns: node_ann_table, fm: fn_info_map};

fn get_fn_info(ccx: crate_ctxt, id: node_id) -> fn_info {
    assert (ccx.fm.contains_key(id));
    return ccx.fm.get(id);
}

fn add_node(ccx: crate_ctxt, i: node_id, a: ts_ann) {
    let sz = vec::len(*ccx.node_anns);
    if sz <= i as uint {
        vec::grow(*ccx.node_anns, (i as uint) - sz + 1u, empty_ann(0u));
    }
    ccx.node_anns[i] = a;
}

fn get_ts_ann(ccx: crate_ctxt, i: node_id) -> option<ts_ann> {
    if i as uint < vec::len(*ccx.node_anns) {
        return some::<ts_ann>(ccx.node_anns[i]);
    } else { return none::<ts_ann>; }
}


/********* utils ********/
fn node_id_to_ts_ann(ccx: crate_ctxt, id: node_id) -> ts_ann {
    alt get_ts_ann(ccx, id) {
      none {
        error!{"node_id_to_ts_ann: no ts_ann for node_id %d", id};
        fail;
      }
      some(tt) { return tt; }
    }
}

fn node_id_to_poststate(ccx: crate_ctxt, id: node_id) -> poststate {
    debug!{"node_id_to_poststate"};
    return node_id_to_ts_ann(ccx, id).states.poststate;
}

fn stmt_to_ann(ccx: crate_ctxt, s: stmt) -> ts_ann {
    debug!{"stmt_to_ann"};
    alt s.node {
      stmt_decl(_, id) | stmt_expr(_, id) | stmt_semi(_, id) {
        return node_id_to_ts_ann(ccx, id);
      }
    }
}


/* fails if e has no annotation */
fn expr_states(ccx: crate_ctxt, e: @expr) -> pre_and_post_state {
    debug!{"expr_states"};
    return node_id_to_ts_ann(ccx, e.id).states;
}


/* fails if e has no annotation */
fn expr_pp(ccx: crate_ctxt, e: @expr) -> pre_and_post {
    debug!{"expr_pp"};
    return node_id_to_ts_ann(ccx, e.id).conditions;
}

fn stmt_pp(ccx: crate_ctxt, s: stmt) -> pre_and_post {
    return stmt_to_ann(ccx, s).conditions;
}


/* fails if b has no annotation */
fn block_pp(ccx: crate_ctxt, b: blk) -> pre_and_post {
    debug!{"block_pp"};
    return node_id_to_ts_ann(ccx, b.node.id).conditions;
}

fn clear_pp(pp: pre_and_post) {
    ann::clear(pp.precondition);
    ann::clear(pp.postcondition);
}

fn clear_precond(ccx: crate_ctxt, id: node_id) {
    let pp = node_id_to_ts_ann(ccx, id);
    ann::clear(pp.conditions.precondition);
}

fn block_states(ccx: crate_ctxt, b: blk) -> pre_and_post_state {
    debug!{"block_states"};
    return node_id_to_ts_ann(ccx, b.node.id).states;
}

fn stmt_states(ccx: crate_ctxt, s: stmt) -> pre_and_post_state {
    return stmt_to_ann(ccx, s).states;
}

fn expr_precond(ccx: crate_ctxt, e: @expr) -> precond {
    return expr_pp(ccx, e).precondition;
}

fn expr_postcond(ccx: crate_ctxt, e: @expr) -> postcond {
    return expr_pp(ccx, e).postcondition;
}

fn expr_prestate(ccx: crate_ctxt, e: @expr) -> prestate {
    return expr_states(ccx, e).prestate;
}

fn expr_poststate(ccx: crate_ctxt, e: @expr) -> poststate {
    return expr_states(ccx, e).poststate;
}

fn stmt_precond(ccx: crate_ctxt, s: stmt) -> precond {
    return stmt_pp(ccx, s).precondition;
}

fn stmt_postcond(ccx: crate_ctxt, s: stmt) -> postcond {
    return stmt_pp(ccx, s).postcondition;
}

fn states_to_poststate(ss: pre_and_post_state) -> poststate {
    return ss.poststate;
}

fn stmt_prestate(ccx: crate_ctxt, s: stmt) -> prestate {
    return stmt_states(ccx, s).prestate;
}

fn stmt_poststate(ccx: crate_ctxt, s: stmt) -> poststate {
    return stmt_states(ccx, s).poststate;
}

fn block_precond(ccx: crate_ctxt, b: blk) -> precond {
    return block_pp(ccx, b).precondition;
}

fn block_postcond(ccx: crate_ctxt, b: blk) -> postcond {
    return block_pp(ccx, b).postcondition;
}

fn block_prestate(ccx: crate_ctxt, b: blk) -> prestate {
    return block_states(ccx, b).prestate;
}

fn block_poststate(ccx: crate_ctxt, b: blk) -> poststate {
    return block_states(ccx, b).poststate;
}

fn set_prestate_ann(ccx: crate_ctxt, id: node_id, pre: prestate) -> bool {
    debug!{"set_prestate_ann"};
    return set_prestate(node_id_to_ts_ann(ccx, id), pre);
}

fn extend_prestate_ann(ccx: crate_ctxt, id: node_id, pre: prestate) -> bool {
    debug!{"extend_prestate_ann"};
    return extend_prestate(node_id_to_ts_ann(ccx, id).states.prestate, pre);
}

fn set_poststate_ann(ccx: crate_ctxt, id: node_id, post: poststate) -> bool {
    debug!{"set_poststate_ann"};
    return set_poststate(node_id_to_ts_ann(ccx, id), post);
}

fn extend_poststate_ann(ccx: crate_ctxt, id: node_id, post: poststate) ->
   bool {
    debug!{"extend_poststate_ann"};
    return extend_poststate(
        node_id_to_ts_ann(ccx, id).states.poststate, post);
}

fn set_pre_and_post(ccx: crate_ctxt, id: node_id, pre: precond,
                    post: postcond) {
    debug!{"set_pre_and_post"};
    let tt = node_id_to_ts_ann(ccx, id);
    set_precondition(tt, pre);
    set_postcondition(tt, post);
}

fn copy_pre_post(ccx: crate_ctxt, id: node_id, sub: @expr) {
    debug!{"set_pre_and_post"};
    let p = expr_pp(ccx, sub);
    copy_pre_post_(ccx, id, p.precondition, p.postcondition);
}

fn copy_pre_post_(ccx: crate_ctxt, id: node_id, pre: prestate,
                  post: poststate) {
    debug!{"set_pre_and_post"};
    let tt = node_id_to_ts_ann(ccx, id);
    set_precondition(tt, pre);
    set_postcondition(tt, post);
}

/* sets all bits to *1* */
fn set_postcond_false(ccx: crate_ctxt, id: node_id) {
    let p = node_id_to_ts_ann(ccx, id);
    ann::set(p.conditions.postcondition);
}

fn pure_exp(ccx: crate_ctxt, id: node_id, p: prestate) -> bool {
    return set_prestate_ann(ccx, id, p) | set_poststate_ann(ccx, id, p);
}

fn num_constraints(m: fn_info) -> uint { return m.num_constraints; }

fn new_crate_ctxt(cx: ty::ctxt) -> crate_ctxt {
    let na: ~[mut ts_ann] = ~[mut];
    return {tcx: cx, node_anns: @mut na, fm: int_hash::<fn_info>()};
}

/* Use e's type to determine whether it returns.
 If it has a function type with a ! annotation,
the answer is noreturn. */
fn controlflow_expr(ccx: crate_ctxt, e: @expr) -> ret_style {
    alt ty::get(ty::node_id_to_type(ccx.tcx, e.id)).struct {
      ty::ty_fn(f) { return f.ret_style; }
      _ { return return_val; }
    }
}

fn constraints_expr(cx: ty::ctxt, e: @expr) -> ~[@ty::constr] {
    alt ty::get(ty::node_id_to_type(cx, e.id)).struct {
      ty::ty_fn(f) { return f.constraints; }
      _ { return ~[]; }
    }
}

fn node_id_to_def_strict(cx: ty::ctxt, id: node_id) -> def {
    alt cx.def_map.find(id) {
      none {
        error!{"node_id_to_def: node_id %d has no def", id};
        fail;
      }
      some(d) { return d; }
    }
}

fn node_id_to_def(ccx: crate_ctxt, id: node_id) -> option<def> {
    return ccx.tcx.def_map.find(id);
}

fn norm_a_constraint(id: def_id, c: constraint) -> ~[norm_constraint] {
    let mut rslt: ~[norm_constraint] = ~[];
    for (*c.descs).each |pd| {
        vec::push(rslt,
                  {bit_num: pd.node.bit_num,
                   c: respan(pd.span, {path: c.path,
                                       def_id: id,
                                       args: pd.node.args})});
    }
    return rslt;
}


// Tried to write this as an iterator, but I got a
// non-exhaustive match in trans.
fn constraints(fcx: fn_ctxt) -> ~[norm_constraint] {
    let mut rslt: ~[norm_constraint] = ~[];
    for fcx.enclosing.constrs.each |key, val| {
        vec::push_all(rslt, norm_a_constraint(key, val));
    };
    return rslt;
}

// FIXME (#2539): Would rather take an immutable vec as an argument,
// should freeze it at some earlier point.
fn match_args(fcx: fn_ctxt, occs: @dvec<pred_args>,
              occ: ~[@constr_arg_use]) -> uint {
    debug!{"match_args: looking at %s",
           constr_args_to_str(fn@(i: inst) -> ~str { *i.ident }, occ)};
    for (*occs).each |pd| {
        log(debug,
                 ~"match_args: candidate " + pred_args_to_str(pd));
        fn eq(p: inst, q: inst) -> bool { return p.node == q.node; }
        if ty::args_eq(eq, pd.node.args, occ) { return pd.node.bit_num; }
    }
    fcx.ccx.tcx.sess.bug(~"match_args: no match for occurring args");
}

fn def_id_for_constr(tcx: ty::ctxt, t: node_id) -> def_id {
    alt tcx.def_map.find(t) {
      none {
        tcx.sess.bug(~"node_id_for_constr: bad node_id " + int::str(t));
      }
      some(def_fn(i, _)) { return i; }
      _ { tcx.sess.bug(~"node_id_for_constr: pred is not a function"); }
    }
}

fn expr_to_constr_arg(tcx: ty::ctxt, e: @expr) -> @constr_arg_use {
    alt e.node {
      expr_path(p) {
        alt tcx.def_map.find(e.id) {
          some(def_local(nid, _)) | some(def_arg(nid, _)) |
          some(def_binding(nid, _)) | some(def_upvar(nid, _, _)) {
            return @respan(p.span,
                        carg_ident({ident: p.idents[0], node: nid}));
          }
          some(what) {
              tcx.sess.span_bug(e.span,
                 fmt!{"exprs_to_constr_args: non-local variable %? \
                                     as pred arg", what});
          }
          none {
              tcx.sess.span_bug(e.span,
                 ~"exprs_to_constr_args: unbound id as pred arg");

          }
        }
      }
      expr_lit(l) { return @respan(e.span, carg_lit(l)); }
      _ {
        tcx.sess.span_fatal(e.span,
                            ~"arguments to constrained functions must be " +
                                ~"literals or local variables");
      }
    }
}

fn exprs_to_constr_args(tcx: ty::ctxt,
                        args: ~[@expr]) -> ~[@constr_arg_use] {
    let f = |a| expr_to_constr_arg(tcx, a);
    let mut rslt: ~[@constr_arg_use] = ~[];
    for args.each |e| { vec::push(rslt, f(e)); }
    rslt
}

fn expr_to_constr(tcx: ty::ctxt, e: @expr) -> sp_constr {
    alt e.node {
      expr_call(operator, args, _) {
        alt operator.node {
          expr_path(p) {
            return respan(e.span,
                       {path: p,
                        def_id: def_id_for_constr(tcx, operator.id),
                        args: exprs_to_constr_args(tcx, args)});
          }
          _ {
            tcx.sess.span_bug(operator.span,
                              ~"ill-formed operator in predicate");
          }
        }
      }
      _ {
        tcx.sess.span_bug(e.span, ~"ill-formed predicate");
      }
    }
}

fn pred_args_to_str(p: pred_args) -> ~str {
    ~"<" + uint::str(p.node.bit_num) + ~", " +
        constr_args_to_str(fn@(i: inst) -> ~str {return *i.ident; },
                           p.node.args)
        + ~">"
}

fn substitute_constr_args(cx: ty::ctxt, actuals: ~[@expr], c: @ty::constr) ->
   tsconstr {
    let mut rslt: ~[@constr_arg_use] = ~[];
    for c.node.args.each |a| {
        vec::push(rslt, substitute_arg(cx, actuals, a));
    }
    return {path: c.node.path,
         def_id: c.node.id,
         args: rslt};
}

fn substitute_arg(cx: ty::ctxt, actuals: ~[@expr], a: @constr_arg) ->
   @constr_arg_use {
    let num_actuals = vec::len(actuals);
    alt a.node {
      carg_ident(i) {
        if i < num_actuals {
            return expr_to_constr_arg(cx, actuals[i]);
        } else {
            cx.sess.span_fatal(a.span, ~"constraint argument out of bounds");
        }
      }
      carg_base { return @respan(a.span, carg_base); }
      carg_lit(l) { return @respan(a.span, carg_lit(l)); }
    }
}

fn pred_args_matches(pattern: ~[constr_arg_general_<inst>],
                     desc: pred_args) ->
   bool {
    let mut i = 0u;
    for desc.node.args.each |c| {
        let n = pattern[i];
        alt c.node {
          carg_ident(p) {
            alt n {
              carg_ident(q) { if p.node != q.node { return false; } }
              _ { return false; }
            }
          }
          carg_base { if n != carg_base { return false; } }
          carg_lit(l) {
            alt n {
              carg_lit(m) { if !const_eval::lit_eq(l, m) { return false; } }
              _ { return false; }
            }
          }
        }
        i += 1u;
    }
    return true;
}

fn find_instance_(pattern: ~[constr_arg_general_<inst>],
                  descs: ~[pred_args]) ->
   option<uint> {
    for descs.each |d| {
        if pred_args_matches(pattern, d) { return some(d.node.bit_num); }
    }
    return none;
}

type inst = {ident: ident, node: node_id};

enum dest {
    local_dest(inst), // RHS is assigned to a local variable
    call                        // RHS is passed to a function
}

type subst = ~[{from: inst, to: inst}];

fn find_instances(_fcx: fn_ctxt, subst: subst,
                  c: constraint) -> ~[{from: uint, to: uint}] {

    if vec::len(subst) == 0u { return ~[]; }
    let mut res = ~[];
    do (*c.descs).swap |v| {
        let v <- vec::from_mut(v);
        for v.each |d| {
            if args_mention(d.node.args, find_in_subst_bool, subst) {
                let old_bit_num = d.node.bit_num;
                let newv = replace(subst, d);
                alt find_instance_(newv, v) {
                  some(d1) {vec::push(res, {from: old_bit_num, to: d1})}
                  _ {}
                }
            } else {}
        }
        vec::to_mut(v)
    }
    return res;
}

fn find_in_subst(id: node_id, s: subst) -> option<inst> {
    for s.each |p| {
        if id == p.from.node { return some(p.to); }
    }
    return none;
}

fn find_in_subst_bool(s: subst, id: node_id) -> bool {
    is_some(find_in_subst(id, s))
}

fn insts_to_str(stuff: ~[constr_arg_general_<inst>]) -> ~str {
    let mut rslt = ~"<";
    for stuff.each |i| {
        rslt +=
            ~" " +
                alt i {
                  carg_ident(p) { *p.ident }
                  carg_base { ~"*" }
                  carg_lit(_) { ~"~[lit]" }
                } + ~" ";
    }
    rslt += ~">";
    rslt
}

fn replace(subst: subst, d: pred_args) -> ~[constr_arg_general_<inst>] {
    let mut rslt: ~[constr_arg_general_<inst>] = ~[];
    for d.node.args.each |c| {
        alt c.node {
          carg_ident(p) {
            alt find_in_subst(p.node, subst) {
              some(newv) { vec::push(rslt, carg_ident(newv)); }
              _ { vec::push(rslt, c.node); }
            }
          }
          _ {
            vec::push(rslt, c.node);
          }
        }
    }

    return rslt;
}

enum if_ty { if_check, plain_if, }

fn for_constraints_mentioning(fcx: fn_ctxt, id: node_id,
                              f: fn(norm_constraint)) {
    for constraints(fcx).each |c| {
        if constraint_mentions(fcx, c, id) { f(c); }
    };
}

fn local_node_id_to_def_id_strict(fcx: fn_ctxt, sp: span, i: node_id) ->
   def_id {
    alt local_node_id_to_def(fcx, i) {
      some(def_local(nid, _)) | some(def_arg(nid, _)) |
      some(def_upvar(nid, _, _)) {
        return local_def(nid);
      }
      some(_) {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    ~"local_node_id_to_def_id: id \
               isn't a local");
      }
      none {
        // should really be bug. span_bug()?
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    ~"local_node_id_to_def_id: id \
               is unbound");
      }
    }
}

fn local_node_id_to_def(fcx: fn_ctxt, i: node_id) -> option<def> {
    fcx.ccx.tcx.def_map.find(i)
}

fn local_node_id_to_def_id(fcx: fn_ctxt, i: node_id) -> option<def_id> {
    alt local_node_id_to_def(fcx, i) {
      some(def_local(nid, _)) | some(def_arg(nid, _)) |
      some(def_binding(nid, _)) | some(def_upvar(nid, _, _)) {
        some(local_def(nid))
      }
      _ { none }
    }
}

fn local_node_id_to_local_def_id(fcx: fn_ctxt, i: node_id) ->
   option<node_id> {
    alt local_node_id_to_def_id(fcx, i) {
      some(did) { some(did.node) }
      _ { none }
    }
}

fn copy_in_postcond(fcx: fn_ctxt, parent_exp: node_id, dest: inst, src: inst,
                    ty: oper_type) {
    let post =
        node_id_to_ts_ann(fcx.ccx, parent_exp).conditions.postcondition;
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

fn copy_in_poststate(fcx: fn_ctxt, post: poststate, dest: inst, src: inst,
                     ty: oper_type) {
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

// In target_post, set the bits corresponding to copies of any
// constraints mentioning src that are set in src_post, with
// dest substituted for src.
// (This doesn't create any new constraints. If a new, substituted
// constraint isn't already in the bit vector, it's ignored.)
fn copy_in_poststate_two(fcx: fn_ctxt, src_post: poststate,
                         target_post: poststate, dest: inst, src: inst,
                         ty: oper_type) {
    let mut subst;
    alt ty {
      oper_swap { subst = ~[{from: dest, to: src}, {from: src, to: dest}]; }
      oper_assign_op {
        return; // Don't do any propagation
      }
      _ { subst = ~[{from: src, to: dest}]; }
    }


    for fcx.enclosing.constrs.each_value |val| {
        // replace any occurrences of the src def_id with the
        // dest def_id
        let insts = find_instances(fcx, subst, val);
        for insts.each |p| {
            if bitvectors::promises_(p.from, src_post) {
                set_in_poststate_(p.to, target_post);
            }
        }
    };
}

fn forget_in_postcond(fcx: fn_ctxt, parent_exp: node_id, dead_v: node_id) {
    // In the postcondition given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    do option::iter(d) |d_id| {
        do for_constraints_mentioning(fcx, d_id) |c| {
                debug!{"clearing constraint %u %s",
                       c.bit_num,
                       constraint_to_str(fcx.ccx.tcx, c.c)};
                clear_in_postcond(c.bit_num,
                                  node_id_to_ts_ann(fcx.ccx,
                                                    parent_exp).conditions);
        }
    };
}

fn forget_in_poststate(fcx: fn_ctxt, p: poststate, dead_v: node_id) -> bool {
    // In the poststate given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    let mut changed = false;
    do option::iter(d) |d_id| {
        do for_constraints_mentioning(fcx, d_id) |c| {
                changed |= clear_in_poststate_(c.bit_num, p);
        }
    }
    return changed;
}

fn any_eq(v: ~[node_id], d: node_id) -> bool {
    for v.each |i| { if i == d { return true; } }
    false
}

fn constraint_mentions(_fcx: fn_ctxt, c: norm_constraint, v: node_id) ->
   bool {
    return args_mention(c.c.node.args, any_eq, ~[v]);
}

fn args_mention<T>(args: ~[@constr_arg_use],
                   q: fn(~[T], node_id) -> bool,
                   s: ~[T]) -> bool {

    for args.each |a| {
        alt a.node {
          carg_ident(p1) { if q(s, p1.node) { return true; } } _ { }
        }
    }
    return false;
}

fn use_var(fcx: fn_ctxt, v: node_id) {
    vec::push(*fcx.enclosing.used_vars, v);
}

fn op_to_oper_ty(io: init_op) -> oper_type {
    alt io { init_move { oper_move } _ { oper_assign } }
}

// default function visitor
fn do_nothing<T>(_fk: visit::fn_kind, _decl: fn_decl, _body: blk,
                 _sp: span, _id: node_id,
                 _t: T, _v: visit::vt<T>) {
}


fn args_to_constr_args(tcx: ty::ctxt, args: ~[arg],
                       indices: ~[@sp_constr_arg<uint>])
    -> ~[@constr_arg_use] {
    let mut actuals: ~[@constr_arg_use] = ~[];
    let num_args = vec::len(args);
    for indices.each |a| {
        vec::push(
            actuals,
            @respan(a.span,
                    alt a.node {
                        carg_base { carg_base }
                        carg_ident(i) {
                            if i < num_args {
                                carg_ident({ident: args[i].ident,
                                            node: args[i].id})
                            } else {
                                tcx.sess.span_bug(a.span,
                                                  ~"index out of bounds in \
                                                   constraint arg");
                            }
                        }
                        carg_lit(l) { carg_lit(l) }
                    }));
    }
    return actuals;
}

fn ast_constr_to_ts_constr(tcx: ty::ctxt, args: ~[arg], c: @constr) ->
   tsconstr {
    let tconstr = ty::ast_constr_to_constr(tcx, c);
    return {path: tconstr.node.path,
         def_id: tconstr.node.id,
         args: args_to_constr_args(tcx, args, tconstr.node.args)};
}

fn ast_constr_to_sp_constr(tcx: ty::ctxt, args: ~[arg], c: @constr) ->
   sp_constr {
    let tconstr = ast_constr_to_ts_constr(tcx, args, c);
    return respan(c.span, tconstr);
}

type binding = {lhs: ~[dest], rhs: option<initializer>};

fn local_to_bindings(tcx: ty::ctxt, loc: @local) -> binding {
    let mut lhs = ~[];
    do pat_bindings(tcx.def_map, loc.node.pat) |p_id, _s, name| {
      vec::push(lhs, local_dest({ident: path_to_ident(name), node: p_id}));
    };
    {lhs: lhs, rhs: loc.node.init}
}

fn locals_to_bindings(tcx: ty::ctxt, locals: ~[@local]) -> ~[binding] {
    let mut rslt = ~[];
    for locals.each |loc| { vec::push(rslt, local_to_bindings(tcx, loc)); }
    return rslt;
}

fn callee_modes(fcx: fn_ctxt, callee: node_id) -> ~[mode] {
    let ty = ty::type_autoderef(fcx.ccx.tcx,
                                ty::node_id_to_type(fcx.ccx.tcx, callee));
    alt ty::get(ty).struct {
      ty::ty_fn({inputs: args, _}) {
        let mut modes = ~[];
        for args.each |arg| { vec::push(modes, arg.mode); }
        return modes;
      }
      _ {
        // Shouldn't happen; callee should be ty_fn.
        fcx.ccx.tcx.sess.bug(~"non-fn callee type in callee_modes: " +
                                 util::ppaux::ty_to_str(fcx.ccx.tcx, ty));
      }
    }
}

fn callee_arg_init_ops(fcx: fn_ctxt, callee: node_id) -> ~[init_op] {
    do vec::map(callee_modes(fcx, callee)) |m| {
        alt ty::resolved_mode(fcx.ccx.tcx, m) {
          by_move { init_move }
          by_copy | by_ref | by_val | by_mutbl_ref { init_assign }
        }
    }
}

fn arg_bindings(ops: ~[init_op], es: ~[@expr]) -> ~[binding] {
    let mut bindings: ~[binding] = ~[];
    let mut i = 0u;
    for ops.each |op| {
        vec::push(bindings,
                  {lhs: ~[call], rhs: some({op: op, expr: es[i]})});
        i += 1u;
    }
    return bindings;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
