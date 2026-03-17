use cleora::configuration::{extract_fields, validate_fields, Column, Configuration};
use cleora::persistence::entity::{EntityMappingPersistor, InMemoryEntityMappingPersistor};
use cleora::persistence::sparse_matrix::{InMemorySparseMatrixPersistor, SparseMatrixPersistor};
use cleora::sparse_matrix::{create_sparse_matrices, SparseMatrix};
use std::sync::Arc;

// ============================================================
// Configuration tests
// ============================================================

#[test]
fn test_extract_fields_simple_columns() {
    let cols = vec!["users", "products", "brands"];
    let result = extract_fields(cols).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].name, "users");
    assert_eq!(result[1].name, "products");
    assert_eq!(result[2].name, "brands");
    assert!(!result[0].transient);
    assert!(!result[0].complex);
    assert!(!result[0].reflexive);
    assert!(!result[0].ignored);
}

#[test]
fn test_extract_fields_with_modifiers() {
    let cols = vec!["transient::users", "complex::reflexive::products", "ignore::brands"];
    let result = extract_fields(cols).unwrap();
    assert_eq!(result.len(), 3);

    assert_eq!(result[0].name, "users");
    assert!(result[0].transient);
    assert!(!result[0].complex);

    assert_eq!(result[1].name, "products");
    assert!(result[1].complex);
    assert!(result[1].reflexive);

    assert_eq!(result[2].name, "brands");
    assert!(result[2].ignored);
}

#[test]
fn test_extract_fields_invalid_modifier() {
    let cols = vec!["invalid_mod::users"];
    let result = extract_fields(cols);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unrecognized column field modifier"));
}

#[test]
fn test_validate_fields_reflexive_and_transient_fails() {
    let cols = vec![Column {
        name: "test".to_string(),
        transient: true,
        complex: true,
        reflexive: true,
        ignored: false,
    }];
    let result = validate_fields(cols);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("REFLEXIVE and simultaneously TRANSIENT"));
}

#[test]
fn test_validate_fields_reflexive_without_complex_fails() {
    let cols = vec![Column {
        name: "test".to_string(),
        transient: false,
        complex: false,
        reflexive: true,
        ignored: false,
    }];
    let result = validate_fields(cols);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("REFLEXIVE but NOT COMPLEX"));
}

#[test]
fn test_validate_fields_valid_reflexive_complex() {
    let cols = vec![Column {
        name: "test".to_string(),
        transient: false,
        complex: true,
        reflexive: true,
        ignored: false,
    }];
    let result = validate_fields(cols);
    assert!(result.is_ok());
}

#[test]
fn test_configuration_not_ignored_columns() {
    let columns = vec![
        Column { name: "a".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
        Column { name: "b".to_string(), transient: false, complex: false, reflexive: false, ignored: true },
        Column { name: "c".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
    ];
    let config = Configuration::default("test.tsv".to_string(), columns);
    let not_ignored = config.not_ignored_columns();
    assert_eq!(not_ignored.len(), 2);
    assert_eq!(not_ignored[0].name, "a");
    assert_eq!(not_ignored[1].name, "c");
}

// ============================================================
// InMemoryEntityMappingPersistor tests
// ============================================================

#[test]
fn test_entity_mapping_persistor_put_and_get() {
    let persistor = InMemoryEntityMappingPersistor::new();
    persistor.put_data(42, "test_entity".to_string());

    assert!(persistor.contains(42));
    assert!(!persistor.contains(43));

    let entity = persistor.get_entity(42);
    assert_eq!(entity, Some("test_entity".to_string()));

    let missing = persistor.get_entity(99);
    assert_eq!(missing, None);
}

#[test]
fn test_entity_mapping_persistor_default() {
    let persistor = InMemoryEntityMappingPersistor::default();
    assert!(!persistor.contains(0));
    assert_eq!(persistor.get_entity(0), None);
}

// ============================================================
// InMemorySparseMatrixPersistor tests
// ============================================================

#[test]
fn test_sparse_matrix_persistor_entity_counter() {
    let mut persistor = InMemorySparseMatrixPersistor::new();
    assert_eq!(persistor.get_entity_counter(), 0);

    persistor.update_entity_counter(5);
    assert_eq!(persistor.get_entity_counter(), 5);
}

#[test]
fn test_sparse_matrix_persistor_hash_id_mapping() {
    let mut persistor = InMemorySparseMatrixPersistor::new();

    assert_eq!(persistor.get_id(100), -1);

    persistor.add_hash_id(100, 0);
    assert_eq!(persistor.get_id(100), 0);

    assert_eq!(persistor.get_hash(0), 100);
    assert_eq!(persistor.get_hash(99), -1);
}

#[test]
fn test_sparse_matrix_persistor_edge_counter() {
    let mut persistor = InMemorySparseMatrixPersistor::new();
    assert_eq!(persistor.increment_edge_counter(), 1);
    assert_eq!(persistor.increment_edge_counter(), 2);
    assert_eq!(persistor.increment_edge_counter(), 3);
}

#[test]
fn test_sparse_matrix_persistor_hash_occurrence() {
    let mut persistor = InMemorySparseMatrixPersistor::new();
    assert_eq!(persistor.increment_hash_occurrence(42), 1);
    assert_eq!(persistor.increment_hash_occurrence(42), 2);
    assert_eq!(persistor.increment_hash_occurrence(42), 3);
    assert_eq!(persistor.get_hash_occurrence(42), 3);
}

#[test]
fn test_sparse_matrix_persistor_entries() {
    use cleora::persistence::sparse_matrix::Entry;

    let mut persistor = InMemorySparseMatrixPersistor::new();
    assert_eq!(persistor.get_amount_of_data(), 0);

    let entry = Entry { row: 0, col: 1, value: 0.5 };
    persistor.add_new_entry(0, entry);
    assert_eq!(persistor.get_amount_of_data(), 1);

    let retrieved = persistor.get_entry(0);
    assert_eq!(retrieved.row, 0);
    assert_eq!(retrieved.col, 1);
    assert!((retrieved.value - 0.5).abs() < f32::EPSILON);

    // update entry value
    let update_entry = Entry { row: 0, col: 1, value: 0.3 };
    persistor.update_entry(0, update_entry);
    let updated = persistor.get_entry(0);
    assert!((updated.value - 0.8).abs() < f32::EPSILON);

    // replace entry
    let replacement = Entry { row: 2, col: 3, value: 1.0 };
    persistor.replace_entry(0, replacement);
    let replaced = persistor.get_entry(0);
    assert_eq!(replaced.row, 2);
    assert_eq!(replaced.col, 3);
    assert!((replaced.value - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_sparse_matrix_persistor_pair_index() {
    let mut persistor = InMemorySparseMatrixPersistor::new();
    assert_eq!(persistor.get_pair_index(999), -1);

    persistor.add_pair_index(999, 42);
    assert_eq!(persistor.get_pair_index(999), 42);
}

#[test]
fn test_sparse_matrix_persistor_row_sum() {
    let mut persistor = InMemorySparseMatrixPersistor::new();
    persistor.update_row_sum(0, 0.5);
    assert!((persistor.get_row_sum(0) - 0.5).abs() < f32::EPSILON);

    persistor.update_row_sum(0, 0.3);
    assert!((persistor.get_row_sum(0) - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_sparse_matrix_persistor_default() {
    let persistor = InMemorySparseMatrixPersistor::default();
    assert_eq!(persistor.get_entity_counter(), 0);
    assert_eq!(persistor.get_amount_of_data(), 0);
}

// ============================================================
// SparseMatrix creation tests
// ============================================================

#[test]
fn test_create_sparse_matrices_simple() {
    let cols = vec![
        Column { name: "a".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
        Column { name: "b".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
        Column { name: "c".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
    ];
    let matrices = create_sparse_matrices(128, &cols);
    // 3 columns -> 3 pairs: (a,b), (a,c), (b,c)
    assert_eq!(matrices.len(), 3);
    assert_eq!(matrices[0].col_a_name, "a");
    assert_eq!(matrices[0].col_b_name, "b");
    assert_eq!(matrices[1].col_a_name, "a");
    assert_eq!(matrices[1].col_b_name, "c");
    assert_eq!(matrices[2].col_a_name, "b");
    assert_eq!(matrices[2].col_b_name, "c");
}

#[test]
fn test_create_sparse_matrices_with_reflexive() {
    let cols = vec![
        Column { name: "a".to_string(), transient: false, complex: true, reflexive: true, ignored: false },
        Column { name: "b".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
    ];
    let matrices = create_sparse_matrices(128, &cols);
    // Pairs: (a,b) + (a,a) reflexive = 2
    assert_eq!(matrices.len(), 2);
}

#[test]
fn test_create_sparse_matrices_transient_pair_skipped() {
    let cols = vec![
        Column { name: "a".to_string(), transient: true, complex: false, reflexive: false, ignored: false },
        Column { name: "b".to_string(), transient: true, complex: false, reflexive: false, ignored: false },
        Column { name: "c".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
    ];
    let matrices = create_sparse_matrices(128, &cols);
    // (a,b) skipped (both transient), (a,c) ok, (b,c) ok = 2
    assert_eq!(matrices.len(), 2);
}

// ============================================================
// SparseMatrix handle_pair and normalize tests
// ============================================================

#[test]
fn test_sparse_matrix_handle_pair() {
    let mut sm = SparseMatrix {
        col_a_id: 0,
        col_a_name: "a".to_string(),
        col_b_id: 1,
        col_b_name: "b".to_string(),
        dimension: 4,
        sparse_matrix_persistor: InMemorySparseMatrixPersistor::new(),
    };

    // hashes[0] = count, hashes[1] = entity_a hash, hashes[2] = entity_b hash
    let hashes: Vec<u64> = vec![2, 100, 200];
    sm.handle_pair(&hashes);

    assert_eq!(sm.sparse_matrix_persistor.get_entity_counter(), 2);
    assert_eq!(sm.sparse_matrix_persistor.get_amount_of_data(), 2); // symmetric pair
}

#[test]
fn test_sparse_matrix_normalize() {
    let mut sm = SparseMatrix {
        col_a_id: 0,
        col_a_name: "a".to_string(),
        col_b_id: 1,
        col_b_name: "b".to_string(),
        dimension: 4,
        sparse_matrix_persistor: InMemorySparseMatrixPersistor::new(),
    };

    let hashes: Vec<u64> = vec![1, 100, 200];
    sm.handle_pair(&hashes);
    sm.normalize();

    // After normalization, values should be divided by row sums
    let entry0 = sm.sparse_matrix_persistor.get_entry(0);
    let entry1 = sm.sparse_matrix_persistor.get_entry(1);
    // Each row has exactly one entry with value 1.0, so normalized should be 1.0/1.0 = 1.0
    assert!((entry0.value - 1.0).abs() < f32::EPSILON);
    assert!((entry1.value - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_sparse_matrix_get_id() {
    let sm = SparseMatrix {
        col_a_id: 2,
        col_a_name: "x".to_string(),
        col_b_id: 5,
        col_b_name: "y".to_string(),
        dimension: 64,
        sparse_matrix_persistor: InMemorySparseMatrixPersistor::new(),
    };
    assert_eq!(sm.get_id(), "2_5");
}

// ============================================================
// TextFileVectorPersistor tests
// ============================================================

#[test]
fn test_text_file_vector_persistor_writes_correctly() {
    use cleora::persistence::embedding::{EmbeddingPersistor, TextFileVectorPersistor};

    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("test_output.out");
    let file_path_str = file_path.to_str().unwrap().to_string();

    {
        let mut persistor = TextFileVectorPersistor::new(file_path_str, true);
        persistor.put_metadata(2, 3);
        persistor.put_data("entity1".to_string(), 5, vec![0.1, 0.2, 0.3]);
        persistor.put_data("entity2".to_string(), 10, vec![0.4, 0.5, 0.6]);
        persistor.finish();
    } // drop flushes BufWriter

    let content = std::fs::read_to_string(&file_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines[0], "2 3");
    assert!(lines[1].starts_with("entity1 5"));
    assert!(lines[2].starts_with("entity2 10"));
}

#[test]
fn test_text_file_vector_persistor_without_occurrence_count() {
    use cleora::persistence::embedding::{EmbeddingPersistor, TextFileVectorPersistor};

    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("test_output_no_count.out");
    let file_path_str = file_path.to_str().unwrap().to_string();

    {
        let mut persistor = TextFileVectorPersistor::new(file_path_str, false);
        persistor.put_metadata(1, 2);
        persistor.put_data("entity1".to_string(), 5, vec![0.1, 0.2]);
        persistor.finish();
    } // drop flushes BufWriter

    let content = std::fs::read_to_string(&file_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines[0], "1 2");
    // Without occurrence count, entity name is followed directly by vector values
    assert!(lines[1].starts_with("entity1 0.1"));
    assert!(!lines[1].contains(" 5 "));
}

// ============================================================
// End-to-end pipeline test
// ============================================================

#[test]
fn test_end_to_end_build_graphs() {
    use cleora::pipeline::build_graphs;

    let dir = tempfile::tempdir().unwrap();
    let input_path = dir.path().join("test_input.tsv");
    std::fs::write(&input_path, "user1\tproduct1\tbrand1\nuser1\tproduct2\tbrand1\nuser2\tproduct1\tbrand2\n").unwrap();

    let columns = vec![
        Column { name: "users".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
        Column { name: "products".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
        Column { name: "brands".to_string(), transient: false, complex: false, reflexive: false, ignored: false },
    ];
    let config = Configuration {
        produce_entity_occurrence_count: true,
        embeddings_dimension: 4,
        max_number_of_iteration: 2,
        prepend_field: false,
        log_every_n: 1000,
        in_memory_embedding_calculation: true,
        input: input_path.to_str().unwrap().to_string(),
        output_dir: None,
        relation_name: "test".to_string(),
        columns,
    };

    let entity_mapping = Arc::new(InMemoryEntityMappingPersistor::new());
    let sparse_matrices = build_graphs(&config, entity_mapping.clone());

    // 3 columns -> 3 sparse matrices
    assert_eq!(sparse_matrices.len(), 3);

    // Verify entities were mapped
    assert!(entity_mapping.contains(
        // At least some entities should exist
        sparse_matrices[0].sparse_matrix_persistor.get_hash(0) as u64
    ));
}
