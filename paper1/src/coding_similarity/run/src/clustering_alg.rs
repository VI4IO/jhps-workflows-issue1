extern crate serde;
extern crate ordered_float;

use crate::types::*;

use std::sync::Arc;
use std::sync::Mutex;
use serde::Serialize;
use ordered_float::OrderedFloat;
use std::collections::HashMap;

trait Print {
    fn print(&self);
}

pub struct Config {
    pub dataset_fn: String,
    pub output_fn: String,
    pub progress_fn: String,
    pub nrows: usize,
    pub n_workers: usize,
}

impl Print for Config {
    fn print(&self) {
        // TODO
    }
}

#[derive(Debug, Serialize)]
pub struct Progress {
    sim_param: f32,
    alg_id: u32,
    alg_name: String,
    nclusters: usize,
    jobs_done: usize,
    jobs_total: usize,
    elapsed: f64,
    delta: f64,
}

pub fn cluster<V, W> (cfg: &Config, wtr: Arc<Mutex<csv::Writer<W>>>,alg: &ALG, min_sim: OrderedFloat<f32>, codings: &HashMap<Jobid,V>, cluster_func: fn(&V, &V) -> f32) -> Vec<Cluster> 
where W: std::io::Write
{
    //println!("Grouping {:?}, ALG {:?}", min_sim, alg);
    let start = chrono::Utc::now();
    let mut clusters: Vec<ClusterCentroid<&V>> = Vec::new();

    let mut avail_codings: Vec<(u32, &V)>;
    if cfg.nrows > 0 {
        avail_codings = codings.iter().take(cfg.nrows).map(|(k, v)| (*k, v)).collect();
    }
    else {
        avail_codings = codings.iter().map(|(k, v)| (*k, v)).collect();
    }

    let mut found_cluster;
    let mut counter = 0;
    let mut start_loop = chrono::Utc::now();
    while let Some((jobid, coding)) = avail_codings.pop() {
        found_cluster = false;
        //let nclusters = clusters.len();
        for cluster in clusters.iter_mut() {
            let sim = cluster_func(&cluster.centroid_coding, &coding);
            if sim >= min_sim.into_inner() {
                // append to existing cluster
                cluster.entities.push(Entity{jobid: jobid, sim: sim});
                found_cluster = true;
                break;
            }
        }
        // create new cluster
        if !found_cluster {
            //println!("Add cluster jobid = {}, clusters = {}", jobid, clusters.len());
            clusters.push(ClusterCentroid{
                centroid_jobid: jobid, 
                centroid_coding: coding, 
                entities: vec![Entity{jobid: jobid, sim: 1.0}]
            });
        }
        if (counter % 10_000) == 0 {
            let stop_loop = chrono::Utc::now();
            println!("{:?}, alg {:?}, nclusters {:?}, left = {:?}/{:?}, ({:.3} seconds)", 
                     min_sim,
                     alg,
                     clusters.len(),
                     avail_codings.len(), 
                     codings.len(),
                     ((stop_loop - start_loop).num_milliseconds() as f64) / (1000 as f64)
                    );

            let alg_id = match alg {
                ALG::Bin(p) => p.id,
                ALG::Hex(p) => p.id,
            };

            let alg_name = match alg {
                ALG::Bin(p) => p.name.clone(),
                ALG::Hex(p) => p.name.clone(),
            };

            let progress = Progress{
                sim_param: min_sim.into_inner(),
                alg_id: alg_id,
                alg_name: alg_name,
                nclusters: clusters.len(),
                jobs_done: codings.len() - avail_codings.len(),
                jobs_total: codings.len(),
                elapsed: ((stop_loop - start).num_milliseconds() as f64) / (1000 as f64),
                delta: ((stop_loop - start_loop).num_milliseconds() as f64) / (1000 as f64)
            };

            {
                let mut wtr = wtr.lock().unwrap();
                wtr.serialize(progress).unwrap();
                wtr.flush().unwrap();
            }

            start_loop = stop_loop;
        }
        counter += 1;
    }
    // reshaping to common representation
    let clusters: Vec<_> = clusters.iter().map(|x| Cluster{centroid_jobid: x.centroid_jobid, entities: x.entities.clone(),}).collect();
    let stop = chrono::Utc::now();
    println!("Finish grouping {:?}, ALG {:?}, nclusters {:?}, ({:.3} seconds)", 
             min_sim,
             alg,
             clusters.len(),
             ((stop - start).num_milliseconds() as f64) / (1000 as f64)
            );
    clusters
}
