extern crate serde;
extern crate ordered_float;
use std::sync::Arc;
use std::sync::Mutex;

use crate::types::*;

use serde::Serialize;
use std::collections::HashMap;

pub struct Config {
    pub jobid: u32,
    pub dataset_fn: String,
    pub output_fn: String,
    pub progress_fn: String,
    pub nrows: usize,
    pub n_workers: usize,
}

#[derive(Debug, Serialize)]
pub struct OutputRowSimilarity {
    pub jobid: u32,
    pub alg_id: u32,
    pub alg_name: String,
    pub similarity: f32
}

#[derive(Debug, Serialize)]
pub struct Progress {
    jobid: u32,
    iteration: u32,
    alg_id: u32,
    alg_name: String,
    //jobs_done: usize,
    //jobs_total: usize,
    delta: i64,
}

pub fn compute_similarities<V, W> (cfg: &Config, wtr: Arc<Mutex<csv::Writer<W>>>, alg: &ALG, codings: &HashMap<Jobid,V>, cluster_func: fn(&V, &V) -> f32) -> Vec<(u32, u32, f32)> 
where W: std::io::Write
{
    let probe_jobid = cfg.jobid;
    let probe_coding = codings.get(&probe_jobid).unwrap();

    //println!("Grouping {:?}, ALG {:?}", min_sim, alg);
    let mut similarities: Vec<(u32, u32, f32)> = Vec::new();
    //let mut avail_codings: Vec<(u32, &V)> = codings.iter().take(cfg.nrows).map(|(k, v)| (*k, v)).collect();
    let avail_codings: Vec<(u32, &V)> = codings.iter().map(|(k, v)| (*k, v)).collect();

    let (alg_id, alg_name) = match alg {
        ALG::Bin(p) => (p.id, p.name.clone()),
        ALG::Hex(p) => (p.id, p.name.clone()),
    };

    for i in 0..3 {
        let mut acs = avail_codings.clone();
        while let Some((jobid, coding)) = acs.pop() {

            let start = chrono::Utc::now();
            let sim = cluster_func(&probe_coding, &coding);
            let stop = chrono::Utc::now();

            let progress_entry = Progress{
                jobid: jobid,
                iteration: i,
                alg_id: alg_id,
                alg_name: alg_name.clone(),
                //jobs_done: codings.len() - acs.len(),
                //jobs_total: codings.len(),
                //elapsed: ((stop_loop - start).num_milliseconds()),
                //delta: ((stop_loop - start_loop).num_milliseconds())
                delta: ((stop - start).num_nanoseconds().unwrap())
            };

            {
                let mut wtr = wtr.lock().unwrap();
                wtr.serialize(progress_entry).unwrap();
                wtr.flush().unwrap();
            }

            similarities.push((jobid, alg_id, sim));
        }
    }

    similarities
}
