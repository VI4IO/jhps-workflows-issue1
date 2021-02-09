extern crate num_cpus;
extern crate jemallocator;
extern crate clap;

//mod clustering_alg;
//mod sim_job_finder;

use clap::{Arg, App, SubCommand};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

use std::process;


fn main() {
    let matches = App::new("Parallel Job Clustering")
        .version("0.1")
        .author("Eugen B. <eugen.betke@mailbox.org>, Julian. K. <juliankunkel@gmail.com")
        .about("Parallel Job Clustering")
        //.arg(Arg::with_name("config")
        //     .short("c")
        //     .long("config")
        //     .value_name("FILE")
        //     .help("Sets a custom config file")
        //     .takes_value(true))
        //.arg(Arg::with_name("INPUT")
        //     .help("Sets the input file to use")
        //     .required(true)
        //     .index(1))
        .arg(Arg::with_name("v")
             .short("v")
             .multiple(true)
             .help("Sets the level of verbosity"))
        .subcommand(SubCommand::with_name("cluster")
                    .about("controls testing features")
                    .version("1.3")
                    .author("Eugen B. <eugen.betke@mailbox.org>, Julian. K. <juliankunkel@gmail.com>")
                    .arg(Arg::with_name("dataset")
                         .short("d")
                         .long("dataset")
                         .required(true)
                         .takes_value(true)
                         .help("dataset file name"))
                    .arg(Arg::with_name("output")
                         .short("o")
                         .long("output")
                         .required(true)
                         .takes_value(true)
                         .help("output file name"))
                    .arg(Arg::with_name("progress")
                         .short("p")
                         .long("progress")
                         .required(true)
                         .takes_value(true)
                         .help("capture clustering progress"))
                    .arg(Arg::with_name("workers")
                         .short("w")
                         .long("workers")
                         .required(false)
                         .takes_value(true)
                         .default_value("1")
                         .help("number of workers"))
                    .arg(Arg::with_name("nrows")
                         .short("n")
                         .long("nrows")
                         .required(false)
                         .takes_value(true)
                         .help("number of rows"))
                   )
        .subcommand(SubCommand::with_name("find_similar")
                    .about("controls testing features")
                    .version("1.3")
                    .author("Eugen B. <eugen.betke@mailbox.org>, Julian. K. <juliankunkel@gmail.com>")
                    .arg(Arg::with_name("jobid")
                         .short("j")
                         .long("jobid")
                         .required(true)
                         .takes_value(true)
                         .help("jobid"))
                    .arg(Arg::with_name("dataset")
                         .short("d")
                         .long("dataset")
                         .required(true)
                         .takes_value(true)
                         .help("dataset file name"))
                    .arg(Arg::with_name("output")
                         .short("o")
                         .long("output")
                         .required(true)
                         .takes_value(true)
                         .help("output file name"))
                    .arg(Arg::with_name("progress")
                         .short("p")
                         .long("progress")
                         .required(true)
                         .takes_value(true)
                         .help("capture clustering progress"))
                    .arg(Arg::with_name("workers")
                         .short("w")
                         .long("workers")
                         .required(false)
                         .takes_value(true)
                         .default_value("1")
                         .help("number of workers"))
                    .arg(Arg::with_name("nrows")
                         .short("n")
                         .long("nrows")
                         .required(false)
                         .takes_value(true)
                         .help("number of rows"))
                   )
        .get_matches();

    // Gets a value for config if supplied by user, or defaults to "default.conf"
    //let config = matches.value_of("config").unwrap_or("default.conf");
    //println!("Value for config: {}", config);

    // Calling .unwrap() is safe here because "INPUT" is required (if "INPUT" wasn't
    // required we could have used an 'if let' to conditionally get the value)
    //println!("Using input file: {}", matches.value_of("INPUT").unwrap());

    // Vary the output based on how many times the user used the "verbose" flag
    // (i.e. 'myprog -v -v -v' or 'myprog -vvv' vs 'myprog -v'
    match matches.occurrences_of("v") {
       0 => println!("No verbose info"),
       1 => println!("Some verbose info"),
       2 => println!("Tons of verbose info"),
       3 | _ => println!("Don't be crazy"),
    }


    // You can handle information about subcommands by requesting their matches by name
    // (as below), requesting just the name used, or both at the same time
    if let Some(matches) = matches.subcommand_matches("cluster") {
        let mut dataset_fn = String::from("");
        if let Some(value) = matches.value_of("dataset") {
            dataset_fn = String::from(value);
        }
        let mut output_fn = String::from("");
        if let Some(value) = matches.value_of("output") {
            output_fn = String::from(value);
        }
        let mut progress_fn = String::from("");
        if let Some(value) = matches.value_of("progress") {
            progress_fn = String::from(value);
        }

        let nrows = 1000000;
        let n_workers = num_cpus::get();

        let cfg = run::clustering_alg::Config{
            dataset_fn,
            output_fn,
            progress_fn,
            nrows,
            n_workers,
        };
        if let Err(e) = run::create_clusters(cfg) {
            eprintln!("Error occured in run: {}", e);
            process::exit(1);
        }
    }

    if let Some(matches) = matches.subcommand_matches("find_similar") {
        let mut jobid = 0;
        if let Some(value) = matches.value_of("jobid") {
            jobid = value.to_string().parse::<u32>().unwrap();
        }
        let mut dataset_fn = String::from("");
        if let Some(value) = matches.value_of("dataset") {
            dataset_fn = String::from(value);
        }
        let mut output_fn = String::from("");
        if let Some(value) = matches.value_of("output") {
            output_fn = String::from(value);
        }
        let mut progress_fn = String::from("");
        if let Some(value) = matches.value_of("progress") {
            progress_fn = String::from(value);
        }

        let mut nrows = 0;
        if let Some(value) = matches.value_of("nrows") {
            nrows = String::from(value).parse::<usize>().unwrap();
        }

        let n_workers = num_cpus::get();

        let cfg = run::sim_job_finder::Config{
            jobid,
            dataset_fn,
            output_fn,
            progress_fn,
            nrows,
            n_workers,
        };
        if let Err(e) = run::find_similar(cfg) {
            eprintln!("Error occured in run: {}", e);
            process::exit(1);
        }
    }
}
