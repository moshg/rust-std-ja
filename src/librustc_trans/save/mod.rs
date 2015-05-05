// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::Session;
use middle::ty;

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use syntax::{attr, visit};
use syntax::ast::{self, NodeId, DefId};
use syntax::parse::token::keywords;
use syntax::codemap::*;

use self::span_utils::SpanUtils;

mod span_utils;
mod recorder;

mod dump_csv;

pub struct SaveContext<'l, 'tcx: 'l> {
    sess: &'l Session,
    analysis: &'l ty::CrateAnalysis<'tcx>,
    span_utils: SpanUtils<'l>,
}

pub struct CrateData {
    pub name: String,
    pub number: u32,
}

pub enum Data {
    FunctionData(FunctionData),
}

pub struct FunctionData {
    pub id: NodeId,
    pub qualname: String,
    pub declaration: Option<DefId>,
    pub span: Span,
    pub scope: NodeId,
}

impl<'l, 'tcx: 'l> SaveContext<'l, 'tcx> {
    pub fn new(sess: &'l Session,
               analysis: &'l ty::CrateAnalysis<'tcx>,
               span_utils: SpanUtils<'l>)
               -> SaveContext<'l, 'tcx> {
        SaveContext {
            sess: sess,
            analysis: analysis,
            span_utils: span_utils,
        }
    }

    // List external crates used by the current crate.
    pub fn get_external_crates(&self) -> Vec<CrateData> {
        let mut result = Vec::new();

        self.sess.cstore.iter_crate_data(|n, cmd| {
            result.push(CrateData { name: cmd.name.clone(), number: n });
        });

        result
    }

    pub fn get_item_data(&self, item: &ast::Item) -> Data {
        match item.node {
            ast::Item_::ItemFn(..) => {
                let qualname = format!("::{}", self.analysis.ty_cx.map.path_to_string(item.id));
                let sub_span = self.span_utils.sub_span_after_keyword(item.span, keywords::Fn);

                Data::FunctionData(FunctionData {
                    id: item.id,
                    qualname: qualname,
                    declaration: None,
                    span: sub_span.unwrap(),
                    scope: self.analysis.ty_cx.map.get_parent(item.id),
                })
            }
            _ => {
                unimplemented!();
            }
        }
    }

    pub fn get_data_for_id(&self, id: &NodeId) -> Data {
        unimplemented!();        
    }
}

#[allow(deprecated)]
pub fn process_crate(sess: &Session,
                     krate: &ast::Crate,
                     analysis: &ty::CrateAnalysis,
                     odir: Option<&Path>) {
    if generated_code(krate.span) {
        return;
    }

    assert!(analysis.glob_map.is_some());
    let cratename = match attr::find_crate_name(&krate.attrs) {
        Some(name) => name.to_string(),
        None => {
            info!("Could not find crate name, using 'unknown_crate'");
            String::from_str("unknown_crate")
        },
    };

    info!("Dumping crate {}", cratename);

    // find a path to dump our data to
    let mut root_path = match env::var_os("DXR_RUST_TEMP_FOLDER") {
        Some(val) => PathBuf::from(val),
        None => match odir {
            Some(val) => val.join("dxr"),
            None => PathBuf::from("dxr-temp"),
        },
    };

    match fs::create_dir_all(&root_path) {
        Err(e) => sess.err(&format!("Could not create directory {}: {}",
                           root_path.display(), e)),
        _ => (),
    }

    {
        let disp = root_path.display();
        info!("Writing output to {}", disp);
    }

    // Create output file.
    let mut out_name = cratename.clone();
    out_name.push_str(".csv");
    root_path.push(&out_name);
    let output_file = match File::create(&root_path) {
        Ok(f) => box f,
        Err(e) => {
            let disp = root_path.display();
            sess.fatal(&format!("Could not open {}: {}", disp, e));
        }
    };
    root_path.pop();

    let mut visitor = dump_csv::DumpCsvVisitor::new(sess, analysis, output_file);

    visitor.dump_crate_info(&cratename[..], krate);
    visit::walk_crate(&mut visitor, krate);
}

// Utility functions for the module.

// Helper function to escape quotes in a string
fn escape(s: String) -> String {
    s.replace("\"", "\"\"")
}

// If the expression is a macro expansion or other generated code, run screaming
// and don't index.
fn generated_code(span: Span) -> bool {
    span.expn_id != NO_EXPANSION || span  == DUMMY_SP
}
