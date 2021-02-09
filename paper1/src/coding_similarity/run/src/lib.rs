extern crate csv;
extern crate serde;
extern crate chrono;
extern crate threadpool;
extern crate ordered_float;

mod types;
pub mod clustering_alg;
pub mod sim_job_finder;


use types::*;
use std::sync::mpsc::channel;
//use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::sync::Mutex;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use serde::Serialize;
use threadpool::ThreadPool;
use std::collections::HashMap;
use ordered_float::OrderedFloat;
//use std::thread;


#[derive(Debug, Deserialize)]
pub struct Record {
    jobid: u32,
    mean_md_file_create: String,
    mean_md_file_delete: String,
    mean_md_mod: String,
    mean_md_other: String,
    mean_md_read: String,
    mean_read_bytes: String,
    mean_read_calls: String,
    mean_write_bytes: String,
    mean_write_calls: String,
    q16_md_file_create: String,
    q16_md_file_delete: String,
    q16_md_mod: String,
    q16_md_other: String,
    q16_md_read: String,
    q16_read_bytes: String,
    q16_read_calls: String,
    q16_write_bytes: String,
    q16_write_calls: String,
    bcoding: String,
}


//impl<T> std::fmt::Debug for Profile<T> {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        f.debug_struct("").finish()
//    }
//}



#[derive(Debug, Serialize)]
pub struct OutputRow {
    pub jobid: u32,
    pub cluster: u32,
    pub alg_id: u32,
    pub alg_name: String,
    pub sim: f32,
    pub sim_param: f32,
}


pub type BCodings = HashMap<Jobid, bin::JobCoding>;
pub type QCodings = HashMap<Jobid, hex::JobCoding>;


pub fn read_codings(data_fn: &String) -> (BCodings, QCodings) {
    let mut b_codings: BCodings = HashMap::new();
    let mut q_codings: QCodings = HashMap::new();

    let file = File::open(&data_fn).expect("Unable to open dataset.");
    let mut rdr = csv::Reader::from_reader(file);

    for result in rdr.deserialize() {
        let record: Record = result.expect("bla bla");
        //let abs_aggzeros_coding = convert_to_coding_2(record.coding_abs_aggzeros);
        let b_coding = bin::convert_to_coding(record.bcoding);
        let q_coding = vec![
			hex::convert_to_coding(record.q16_md_file_create),
			hex::convert_to_coding(record.q16_md_file_delete),
			hex::convert_to_coding(record.q16_md_mod),
			hex::convert_to_coding(record.q16_md_other),
			hex::convert_to_coding(record.q16_md_read),
			hex::convert_to_coding(record.q16_read_bytes),
			hex::convert_to_coding(record.q16_read_calls),
			hex::convert_to_coding(record.q16_write_bytes),
			hex::convert_to_coding(record.q16_write_calls),];

        if b_coding.iter().sum::<bin::Score>() > 0 {
            b_codings.insert(record.jobid, b_coding);
        }

        if q_coding.iter().map(|x| x.iter().sum::<hex::Score>()).sum::<hex::Score>() > (0 as hex::Score) {
           q_codings.insert(record.jobid, q_coding);
        }
    }


    (b_codings, q_codings) 
}


pub fn create_clusters(cfg: clustering_alg::Config) -> Result<(), Box<dyn Error>> {
    let (b_codings, q_codings) = read_codings(&cfg.dataset_fn);

    let min_sims: Vec<OrderedFloat<f32>> = vec![
        OrderedFloat(0.1), 
        OrderedFloat(0.3), 
        OrderedFloat(0.5), 
        OrderedFloat(0.7),
        OrderedFloat(0.9),
        OrderedFloat(0.95),
        OrderedFloat(0.99),
    ];

    let mut algs = Vec::new();
    algs.push(ALG::Bin(Profile{name: String::from("bin_all"), id:1, dataset: b_codings.clone(), func: bin::all::similarity,}));
    algs.push(ALG::Bin(Profile{name: String::from("bin_aggzeros"), id:2, dataset: b_codings.clone(), func: bin::aggzeros::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_lev"), id:3, dataset: q_codings.clone(), func: hex::lev::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_native"), id:5, dataset: q_codings.clone(), func: hex::native::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_phases"), id:4, dataset: q_codings.clone(), func: hex::quant::similarity,}));

    let cfg = Arc::new(cfg);

    let pool = ThreadPool::new(cfg.n_workers);
    //let channel_buf_size = 2000;
    //let (tx, rx) = sync_channel(channel_buf_size);
    let (tx, rx) = channel();
    let file = File::create(&cfg.progress_fn).expect("Unable to open progress file");
    let wtr = csv::Writer::from_writer(file);
    let wtr = Arc::new(Mutex::new(wtr));

    for main_min_sim in min_sims.iter() {
        let min_sim = *main_min_sim;
        let wtr = wtr.clone();

        for main_alg in algs.iter() {
            match main_alg.clone() {
                ALG::Bin(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    let wtr = wtr.clone();
                    pool.execute( move || {
                        let clusters = clustering_alg::cluster(&cfg, wtr, &alg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
                ALG::Hex(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    let wtr = wtr.clone();
                    pool.execute( move || {
                        let clusters = clustering_alg::cluster(&cfg, wtr, &alg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
            };
        }
        wtr.lock().unwrap().flush().unwrap();
    }


    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);
    let mut rx_iter = rx.iter();

    for _ in min_sims.iter() {
        for _ in algs.iter() {
            let (alg, min_sim, clusters) = rx_iter.next().unwrap();
            let (alg_id, alg_name) = match alg {
                ALG::Bin(p) => (p.id, p.name),
                ALG::Hex(p) => (p.id, p.name),
            };

            for cluster in clusters.iter() {
                let cluster_id = cluster.centroid_jobid;
                for entity in cluster.entities.iter() {
                    let output_row = OutputRow {
                        jobid: entity.jobid,
                        cluster: cluster_id,
                        alg_id: alg_id,
                        alg_name: alg_name.clone(),
                        sim: entity.sim,
                        sim_param: min_sim.into_inner(),
                    };

                    wtr.serialize(output_row)?;
                }
            }
            wtr.flush()?;
        }
    }
    Ok(())
}





pub fn find_similar(cfg: sim_job_finder::Config) -> Result<(), Box<dyn Error>> {
    let (b_codings, q_codings) = read_codings(&cfg.dataset_fn);

    let mut algs = Vec::new();
    algs.push(ALG::Bin(Profile{name: String::from("bin_all"), id:1, dataset: b_codings.clone(), func: bin::all::similarity,}));
    algs.push(ALG::Bin(Profile{name: String::from("bin_aggzeros"), id:2, dataset: b_codings.clone(), func: bin::aggzeros::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_lev"), id:3, dataset: q_codings.clone(), func: hex::lev::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_native"), id:5, dataset: q_codings.clone(), func: hex::native::similarity,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex_phases"), id:4, dataset: q_codings.clone(), func: hex::quant::similarity,}));

    let cfg = Arc::new(cfg);

    let pool = ThreadPool::new(cfg.n_workers);
    //let channel_buf_size = 2000;
    //let (tx, rx) = sync_channel(channel_buf_size);
    let (tx, rx) = channel();
    let file = File::create(&cfg.progress_fn).expect("Unable to open progress file");
    let wtr = csv::Writer::from_writer(file);
    let wtr = Arc::new(Mutex::new(wtr));

    let wtr = wtr.clone();

    for main_alg in algs.iter() {
        let alg = main_alg.clone();
        let cfg = cfg.clone();
        let tx = tx.clone();
        match main_alg.clone() {
            ALG::Bin(p)  => {
                let codings = p.dataset.clone();
                let wtr = wtr.clone();
                pool.execute( move || {
                    let clusters = sim_job_finder::compute_similarities(&cfg, wtr, &alg, &codings, p.func);
                    tx.send((alg, clusters)).unwrap();
                });
            }
            ALG::Hex(p) => {
                let codings = p.dataset.clone();
                let wtr = wtr.clone();
                pool.execute( move || {
                    let clusters = sim_job_finder::compute_similarities(&cfg, wtr, &alg, &codings, p.func);
                    tx.send((alg, clusters)).unwrap();
                });
            }
        };
    }
    wtr.lock().unwrap().flush().unwrap();


    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);
    let mut rx_iter = rx.iter();

    for _ in algs.iter() {
        let (alg, clusters) = rx_iter.next().unwrap();
        let (alg_n, alg_name) = match alg {
            ALG::Bin(p) => (p.id, p.name),
            ALG::Hex(p) => (p.id, p.name),
        };

        for cluster in clusters.iter() {
            let cluster_id = cluster;
            let output_row = sim_job_finder::OutputRowSimilarity {
                jobid: cluster_id.0,
                alg_id: alg_n,
                alg_name: alg_name.clone(),
                similarity: cluster_id.2,
            };

            wtr.serialize(output_row)?;
        }
        wtr.flush()?;
    }
    Ok(())
}
