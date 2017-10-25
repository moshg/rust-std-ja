// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, RegionKind, TyCtxt};
use rustc::mir::{Location, Mir};
use rustc::mir::transform::{MirPass, MirSource};
use rustc::infer::InferCtxt;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use std::collections::BTreeSet;
use std::fmt;
use util::liveness::{self, LivenessMode, LivenessResult, LocalSet};

use util as mir_util;
use self::mir_util::PassWhere;

mod constraint_generation;
mod subtype;

mod region_infer;
use self::region_infer::RegionInferenceContext;

mod renumber;

// MIR Pass for non-lexical lifetimes
pub struct NLL;

impl MirPass for NLL {
    fn run_pass<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        source: MirSource,
        input_mir: &mut Mir<'tcx>,
    ) {
        if !tcx.sess.opts.debugging_opts.nll {
            return;
        }

        tcx.infer_ctxt()
            .enter(|ref infcx| drop(compute_regions(infcx, source, input_mir)));
    }
}

pub struct RegionComputation<'tcx> {
    /// A rewritten version of the input MIR where all the regions are
    /// rewritten to refer to inference variables.
    pub mir: Mir<'tcx>,

    /// The definitions (along with their final values) for all regions.
    pub regioncx: RegionInferenceContext,
}

/// Computes the (non-lexical) regions from the input MIR.
///
/// This may result in errors being reported.
pub fn compute_regions<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    source: MirSource,
    input_mir: &Mir<'tcx>,
) -> RegionComputation<'tcx> {
    // Clone mir so we can mutate it without disturbing the rest of the compiler
    let mut mir = input_mir.clone();

    // Replace all regions with fresh inference variables.
    let num_region_variables = renumber::renumber_mir(infcx, &mut mir);

    // Compute what is live where.
    let liveness = &LivenessResults {
        regular: liveness::liveness_of_locals(
            &mir,
            LivenessMode {
                include_regular_use: true,
                include_drops: false,
            },
        ),

        drop: liveness::liveness_of_locals(
            &mir,
            LivenessMode {
                include_regular_use: false,
                include_drops: true,
            },
        ),
    };

    // Create the region inference context, generate the constraints,
    // and then solve them.
    let mut regioncx = RegionInferenceContext::new(num_region_variables);
    constraint_generation::generate_constraints(infcx, &mut regioncx, &mir, source, liveness);
    let errors = regioncx.solve(infcx, &mir);

    assert!(errors.is_empty(), "FIXME: report region inference failures");

    let computation = RegionComputation { mir, regioncx };

    // Dump MIR results into a file, if that is enabled. This let us
    // write unit-tests.
    dump_mir_results(infcx, liveness, source, &computation);

    computation
}

struct LivenessResults {
    regular: LivenessResult,
    drop: LivenessResult,
}

fn dump_mir_results<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    liveness: &LivenessResults,
    source: MirSource,
    computation: &RegionComputation<'tcx>,
) {
    if !mir_util::dump_enabled(infcx.tcx, "nll", source) {
        return;
    }

    let RegionComputation {
        ref mir,
        ref regioncx,
    } = *computation;

    let regular_liveness_per_location: FxHashMap<_, _> = mir.basic_blocks()
        .indices()
        .flat_map(|bb| {
            let mut results = vec![];
            liveness
                .regular
                .simulate_block(&mir, bb, |location, local_set| {
                    results.push((location, local_set.clone()));
                });
            results
        })
        .collect();

    let drop_liveness_per_location: FxHashMap<_, _> = mir.basic_blocks()
        .indices()
        .flat_map(|bb| {
            let mut results = vec![];
            liveness
                .drop
                .simulate_block(&mir, bb, |location, local_set| {
                    results.push((location, local_set.clone()));
                });
            results
        })
        .collect();

    mir_util::dump_mir(infcx.tcx, None, "nll", &0, source, mir, |pass_where, out| {
        match pass_where {
            // Before the CFG, dump out the values for each region variable.
            PassWhere::BeforeCFG => for region in regioncx.regions() {
                writeln!(
                    out,
                    "| {:?}: {:?}",
                    region,
                    regioncx.region_value(region)
                )?;
            },

            // Before each basic block, dump out the values
            // that are live on entry to the basic block.
            PassWhere::BeforeBlock(bb) => {
                let s = live_variable_set(&liveness.regular.ins[bb], &liveness.drop.ins[bb]);
                writeln!(out, "    | Live variables on entry to {:?}: {}", bb, s)?;
            }

            PassWhere::InCFG(location) => {
                let s = live_variable_set(
                    &regular_liveness_per_location[&location],
                    &drop_liveness_per_location[&location],
                );
                writeln!(out, "            | Live variables at {:?}: {}", location, s)?;
            }

            PassWhere::AfterCFG => {}
        }
        Ok(())
    });
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Region {
    points: BTreeSet<Location>,
}

impl fmt::Debug for Region {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(formatter, "{:?}", self.points)
    }
}

impl Region {
    pub fn add_point(&mut self, point: Location) -> bool {
        self.points.insert(point)
    }

    pub fn may_contain(&self, point: Location) -> bool {
        self.points.contains(&point)
    }
}

newtype_index!(RegionIndex {
    DEBUG_NAME = "R",
});

/// Right now, we piggy back on the `ReVar` to store our NLL inference
/// regions. These are indexed with `RegionIndex`. This method will
/// assert that the region is a `ReVar` and convert the internal index
/// into a `RegionIndex`. This is reasonable because in our MIR we
/// replace all free regions with inference variables.
trait ToRegionIndex {
    fn to_region_index(&self) -> RegionIndex;
}

impl ToRegionIndex for RegionKind {
    fn to_region_index(&self) -> RegionIndex {
        if let &ty::ReVar(vid) = self {
            RegionIndex::new(vid.index as usize)
        } else {
            bug!("region is not an ReVar: {:?}", self)
        }
    }
}

fn live_variable_set(regular: &LocalSet, drops: &LocalSet) -> String {
    // sort and deduplicate:
    let all_locals: BTreeSet<_> = regular.iter().chain(drops.iter()).collect();

    // construct a string with each local, including `(drop)` if it is
    // only dropped, versus a regular use.
    let mut string = String::new();
    for local in all_locals {
        string.push_str(&format!("{:?}", local));

        if !regular.contains(&local) {
            assert!(drops.contains(&local));
            string.push_str(" (drop)");
        }

        string.push_str(", ");
    }

    let len = if string.is_empty() {
        0
    } else {
        string.len() - 2
    };

    format!("[{}]", &string[..len])
}
