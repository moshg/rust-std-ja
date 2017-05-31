// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::hir::svh::Svh;
use rustc::ich::Fingerprint;
use rustc::middle::cstore::EncodedMetadataHashes;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc_data_structures::fx::FxHashMap;
use rustc_serialize::Encodable as RustcEncodable;
use rustc_serialize::opaque::Encoder;
use std::io::{self, Cursor, Write};
use std::fs::{self, File};
use std::path::PathBuf;

use IncrementalHashesMap;
use super::data::*;
use super::hash::*;
use super::preds::*;
use super::fs::*;
use super::dirty_clean;
use super::file_format;
use super::work_product;

pub fn save_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                incremental_hashes_map: &IncrementalHashesMap,
                                metadata_hashes: &EncodedMetadataHashes,
                                svh: Svh) {
    debug!("save_dep_graph()");
    let _ignore = tcx.dep_graph.in_ignore();
    let sess = tcx.sess;
    if sess.opts.incremental.is_none() {
        return;
    }

    let query = tcx.dep_graph.query();

    if tcx.sess.opts.debugging_opts.incremental_info {
        println!("incremental: {} nodes in dep-graph", query.graph.len_nodes());
        println!("incremental: {} edges in dep-graph", query.graph.len_edges());
    }

    let mut hcx = HashContext::new(tcx, incremental_hashes_map);
    let preds = Predecessors::new(&query, &mut hcx);
    let mut current_metadata_hashes = FxHashMap();

    // IMPORTANT: We are saving the metadata hashes *before* the dep-graph,
    //            since metadata-encoding might add new entries to the
    //            DefIdDirectory (which is saved in the dep-graph file).
    if sess.opts.debugging_opts.incremental_cc ||
       sess.opts.debugging_opts.query_dep_graph {
        save_in(sess,
                metadata_hash_export_path(sess),
                |e| encode_metadata_hashes(tcx,
                                           svh,
                                           metadata_hashes,
                                           &mut current_metadata_hashes,
                                           e));
    }

    save_in(sess,
            dep_graph_path(sess),
            |e| encode_dep_graph(tcx, &preds, e));

    let prev_metadata_hashes = incremental_hashes_map.prev_metadata_hashes.borrow();
    dirty_clean::check_dirty_clean_metadata(tcx,
                                            &*prev_metadata_hashes,
                                            &current_metadata_hashes);
}

pub fn save_work_products(sess: &Session) {
    if sess.opts.incremental.is_none() {
        return;
    }

    debug!("save_work_products()");
    let _ignore = sess.dep_graph.in_ignore();
    let path = work_products_path(sess);
    save_in(sess, path, |e| encode_work_products(sess, e));

    // We also need to clean out old work-products, as not all of them are
    // deleted during invalidation. Some object files don't change their
    // content, they are just not needed anymore.
    let new_work_products = sess.dep_graph.work_products();
    let previous_work_products = sess.dep_graph.previous_work_products();

    for (id, wp) in previous_work_products.iter() {
        if !new_work_products.contains_key(id) {
            work_product::delete_workproduct_files(sess, wp);
            debug_assert!(wp.saved_files.iter().all(|&(_, ref file_name)| {
                !in_incr_comp_dir_sess(sess, file_name).exists()
            }));
        }
    }

    // Check that we did not delete one of the current work-products:
    debug_assert!({
        new_work_products.iter()
                         .flat_map(|(_, wp)| wp.saved_files
                                               .iter()
                                               .map(|&(_, ref name)| name))
                         .map(|name| in_incr_comp_dir_sess(sess, name))
                         .all(|path| path.exists())
    });
}

fn save_in<F>(sess: &Session, path_buf: PathBuf, encode: F)
    where F: FnOnce(&mut Encoder) -> io::Result<()>
{
    debug!("save: storing data in {}", path_buf.display());

    // delete the old dep-graph, if any
    // Note: It's important that we actually delete the old file and not just
    // truncate and overwrite it, since it might be a shared hard-link, the
    // underlying data of which we don't want to modify
    if path_buf.exists() {
        match fs::remove_file(&path_buf) {
            Ok(()) => {
                debug!("save: remove old file");
            }
            Err(err) => {
                sess.err(&format!("unable to delete old dep-graph at `{}`: {}",
                                  path_buf.display(),
                                  err));
                return;
            }
        }
    }

    // generate the data in a memory buffer
    let mut wr = Cursor::new(Vec::new());
    file_format::write_file_header(&mut wr).unwrap();
    match encode(&mut Encoder::new(&mut wr)) {
        Ok(()) => {}
        Err(err) => {
            sess.err(&format!("could not encode dep-graph to `{}`: {}",
                              path_buf.display(),
                              err));
            return;
        }
    }

    // write the data out
    let data = wr.into_inner();
    match File::create(&path_buf).and_then(|mut file| file.write_all(&data)) {
        Ok(_) => {
            debug!("save: data written to disk successfully");
        }
        Err(err) => {
            sess.err(&format!("failed to write dep-graph to `{}`: {}",
                              path_buf.display(),
                              err));
            return;
        }
    }
}

pub fn encode_dep_graph(tcx: TyCtxt,
                        preds: &Predecessors,
                        encoder: &mut Encoder)
                        -> io::Result<()> {
    // First encode the commandline arguments hash
    tcx.sess.opts.dep_tracking_hash().encode(encoder)?;

    let to_hash_based_node = |dep_node: &DepNode<DefId>| {
        dep_node.map_def(|&def_id| Some(tcx.def_path_hash(def_id))).unwrap()
    };

    // Create a flat list of (Input, WorkProduct) edges for
    // serialization.
    let mut edges = FxHashMap();
    for edge in preds.reduced_graph.all_edges() {
        let source = *preds.reduced_graph.node_data(edge.source());
        let target = *preds.reduced_graph.node_data(edge.target());
        match *target {
            DepNode::MetaData(ref def_id) => {
                // Metadata *targets* are always local metadata nodes. We have
                // already handled those in `encode_metadata_hashes`.
                assert!(def_id.is_local());
                continue;
            }
            _ => (),
        }
        debug!("serialize edge: {:?} -> {:?}", source, target);
        let source = to_hash_based_node(source);
        let target = to_hash_based_node(target);
        edges.entry(source).or_insert(vec![]).push(target);
    }

    if tcx.sess.opts.debugging_opts.incremental_dump_hash {
        for (dep_node, hash) in &preds.hashes {
            println!("HIR hash for {:?} is {}", dep_node, hash);
        }
    }

    // Create the serialized dep-graph.
    let bootstrap_outputs = preds.bootstrap_outputs
                                 .iter()
                                 .map(|n| to_hash_based_node(n))
                                 .collect();
    let edges = edges.into_iter()
                     .map(|(k, v)| SerializedEdgeSet { source: k, targets: v })
                     .collect();
    let graph = SerializedDepGraph {
        bootstrap_outputs,
        edges,
        hashes: preds.hashes
            .iter()
            .map(|(&dep_node, &hash)| {
                SerializedHash {
                    dep_node: to_hash_based_node(dep_node),
                    hash: hash,
                }
            })
            .collect(),
    };

    if tcx.sess.opts.debugging_opts.incremental_info {
        println!("incremental: {} nodes in reduced dep-graph", preds.reduced_graph.len_nodes());
        println!("incremental: {} edges in serialized dep-graph", graph.edges.len());
        println!("incremental: {} hashes in serialized dep-graph", graph.hashes.len());
    }

    debug!("graph = {:#?}", graph);

    // Encode the graph data.
    graph.encode(encoder)?;

    Ok(())
}

pub fn encode_metadata_hashes(tcx: TyCtxt,
                              svh: Svh,
                              metadata_hashes: &EncodedMetadataHashes,
                              current_metadata_hashes: &mut FxHashMap<DefId, Fingerprint>,
                              encoder: &mut Encoder)
                              -> io::Result<()> {
    let mut serialized_hashes = SerializedMetadataHashes {
        entry_hashes: metadata_hashes.entry_hashes.to_vec(),
        global_hashes: metadata_hashes.global_hashes.to_vec(),
        index_map: FxHashMap()
    };

    if tcx.sess.opts.debugging_opts.query_dep_graph {
        for serialized_hash in &serialized_hashes.entry_hashes {
            let def_id = DefId::local(serialized_hash.def_index);

            // Store entry in the index_map
            let def_path_hash = tcx.def_path_hash(def_id);
            serialized_hashes.index_map.insert(def_id.index, def_path_hash);

            // Record hash in current_metadata_hashes
            current_metadata_hashes.insert(def_id, serialized_hash.hash);
        }

        debug!("save: stored index_map (len={}) for serialized hashes",
               serialized_hashes.index_map.len());
    }

    // Encode everything.
    svh.encode(encoder)?;
    serialized_hashes.encode(encoder)?;

    Ok(())
}

pub fn encode_work_products(sess: &Session, encoder: &mut Encoder) -> io::Result<()> {
    let work_products: Vec<_> = sess.dep_graph
        .work_products()
        .iter()
        .map(|(id, work_product)| {
            SerializedWorkProduct {
                id: id.clone(),
                work_product: work_product.clone(),
            }
        })
        .collect();

    work_products.encode(encoder)
}
