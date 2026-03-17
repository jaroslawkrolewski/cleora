use std::time::Instant;

use clap::{Arg, Command};
use cleora::configuration;
use cleora::configuration::Configuration;
use cleora::persistence::entity::InMemoryEntityMappingPersistor;
use cleora::pipeline::{build_graphs, train};
use env_logger::Env;
use log::info;
use std::fs;
use std::process;
use std::sync::Arc;

fn main() {
    let env = Env::default()
        .filter_or("MY_LOG_LEVEL", "info")
        .write_style_or("MY_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    let now = Instant::now();

    let matches = Command::new("cleora")
        .version("1.1.0")
        .author("Piotr Babel <piotr.babel@synerise.com> & Jacek Dabrowski <jack.dabrowski@synerise.com>")
        .about("cleora for embeddings calculation")
        .arg(Arg::new("input")
            .short('i')
            .long("input")
            .required(true)
            .help("Input file path"))
        .arg(Arg::new("output-dir")
            .short('o')
            .long("output-dir")
            .help("Output directory for files with embeddings"))
        .arg(Arg::new("dimension")
            .short('d')
            .long("dimension")
            .required(true)
            .help("Embedding dimension size"))
        .arg(Arg::new("number-of-iterations")
            .short('n')
            .long("number-of-iterations")
            .help("Max number of iterations"))
        .arg(Arg::new("columns")
            .short('c')
            .long("columns")
            .required(true)
            .help("Column names (max 12), with modifiers: [transient::, reflexive::, complex::]"))
        .arg(Arg::new("relation-name")
            .short('r')
            .long("relation-name")
            .help("Name of the relation, for output filename generation"))
        .arg(Arg::new("prepend-field-name")
            .short('p')
            .long("prepend-field-name")
            .help("Prepend field name to entity in output"))
        .arg(Arg::new("log-every-n")
            .short('l')
            .long("log-every-n")
            .help("Log output every N lines"))
        .arg(Arg::new("in-memory-embedding-calculation")
            .short('e')
            .long("in-memory-embedding-calculation")
            .value_parser(["0", "1"])
            .help("Calculate embeddings in memory or with memory-mapped files"))
        .get_matches();

    info!("Reading args...");

    let input = matches.get_one::<String>("input").expect("input is required");
    let output_dir = matches.get_one::<String>("output-dir").cloned();
    // try to create output directory for files with embeddings
    if let Some(output_dir) = output_dir.as_ref() {
        if let Err(e) = fs::create_dir_all(output_dir) {
            eprintln!("Error: Can't create output directory '{}': {}", output_dir, e);
            process::exit(1);
        }
    }
    let dimension: u16 = matches
        .get_one::<String>("dimension")
        .expect("dimension is required")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("Error: Invalid dimension value: {}", e);
            process::exit(1);
        });
    let max_iter: u8 = matches
        .get_one::<String>("number-of-iterations")
        .map(|s| s.as_str())
        .unwrap_or("4")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("Error: Invalid number-of-iterations value: {}", e);
            process::exit(1);
        });
    let relation_name = matches
        .get_one::<String>("relation-name")
        .map(|s| s.as_str())
        .unwrap_or("emb");
    let prepend_field_name = {
        let value: u8 = matches
            .get_one::<String>("prepend-field-name")
            .map(|s| s.as_str())
            .unwrap_or("0")
            .parse()
            .unwrap_or_else(|e| {
                eprintln!("Error: Invalid prepend-field-name value: {}", e);
                process::exit(1);
            });
        value == 1
    };
    let log_every: u32 = matches
        .get_one::<String>("log-every-n")
        .map(|s| s.as_str())
        .unwrap_or("10000")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("Error: Invalid log-every-n value: {}", e);
            process::exit(1);
        });
    let in_memory_embedding_calculation = {
        let value: u8 = matches
            .get_one::<String>("in-memory-embedding-calculation")
            .map(|s| s.as_str())
            .unwrap_or("1")
            .parse()
            .unwrap_or_else(|e| {
                eprintln!("Error: Invalid in-memory-embedding-calculation value: {}", e);
                process::exit(1);
            });
        value == 1
    };
    let columns = {
        let cols_str = matches.get_one::<String>("columns").expect("columns is required");
        let cols_str_separated: Vec<&str> = cols_str.split(' ').collect();
        match configuration::extract_fields(cols_str_separated) {
            Ok(cols) => match configuration::validate_fields(cols) {
                Ok(validated_cols) => validated_cols,
                Err(msg) => {
                    eprintln!("Error: Invalid column fields: {}", msg);
                    process::exit(1);
                }
            },
            Err(msg) => {
                eprintln!("Error: Column parsing problem: {}", msg);
                process::exit(1);
            }
        }
    };

    let config = Configuration {
        produce_entity_occurrence_count: true,
        embeddings_dimension: dimension,
        max_number_of_iteration: max_iter,
        prepend_field: prepend_field_name,
        log_every_n: log_every,
        in_memory_embedding_calculation,
        input: input.to_string(),
        output_dir,
        relation_name: relation_name.to_string(),
        columns,
    };
    dbg!(&config);

    info!("Starting calculation...");
    let in_memory_entity_mapping_persistor = Arc::new(InMemoryEntityMappingPersistor::default());

    let sparse_matrices = build_graphs(&config, in_memory_entity_mapping_persistor.clone());
    info!(
        "Finished Sparse Matrices calculation in {} sec",
        now.elapsed().as_secs()
    );

    train(config, in_memory_entity_mapping_persistor, sparse_matrices);
    info!("Finished in {} sec", now.elapsed().as_secs());
}
