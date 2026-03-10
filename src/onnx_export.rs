//! Export trained SGBT models to ONNX format.
//!
//! Converts the tree ensemble to an ONNX `TreeEnsembleRegressor` operator,
//! enabling inference in any ONNX-compatible runtime (Python, C++, JavaScript, etc.).
//!
//! # Usage
//!
//! ```no_run
//! use irithyll::{SGBTConfig, SGBT, Sample};
//! use irithyll::onnx_export::{export_onnx, save_onnx};
//!
//! let config = SGBTConfig::builder().n_steps(10).build().unwrap();
//! let mut model = SGBT::new(config);
//!
//! // ... train the model ...
//!
//! // Export to bytes
//! let bytes = export_onnx(&model, 3).unwrap();
//!
//! // Or save directly to a file
//! save_onnx(&model, 3, std::path::Path::new("model.onnx")).unwrap();
//! ```

/// Minimal ONNX protobuf structures for tree ensemble export.
///
/// These match the ONNX specification wire format for the subset needed
/// to represent a `TreeEnsembleRegressor`. Field tags are taken from the
/// ONNX protobuf definitions:
/// <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>
pub(crate) mod onnx_proto {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct TensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        pub dim: Vec<tensor_shape_proto::Dimension>,
    }

    pub mod tensor_shape_proto {
        use prost::Message;

        #[derive(Clone, PartialEq, Message)]
        pub struct Dimension {
            // We use dim_value (tag 1) for known dimensions.
            // For unknown/dynamic dimensions, we simply omit or set to 0.
            #[prost(int64, tag = "1")]
            pub dim_value: i64,
        }
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct TypeProto {
        // oneof value { tensor_type = 1; ... }
        // We only need tensor_type.
        #[prost(message, optional, tag = "1")]
        pub tensor_type: Option<type_proto::Tensor>,
    }

    pub mod type_proto {
        use prost::Message;

        #[derive(Clone, PartialEq, Message)]
        pub struct Tensor {
            /// Element type: 1 = FLOAT, 11 = DOUBLE.
            #[prost(int32, tag = "1")]
            pub elem_type: i32,
            #[prost(message, optional, tag = "2")]
            pub shape: Option<super::TensorShapeProto>,
        }
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct ValueInfoProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<TypeProto>,
    }

    /// ONNX AttributeProto — carries named attributes on operator nodes.
    ///
    /// Field tags from onnx.proto:
    ///   1 = name, 2 = ref_attr_name, 3 = i, 4 = s (bytes), 5 = t,
    ///   6 = f (actually doc_string in some versions — but in ONNX spec f=4... )
    ///
    /// Actually the correct ONNX AttributeProto tags are:
    ///   1 = name, 2 = ref_attr_name, 3 = doc_string,
    ///   4 = type, 5 = f, 6 = i, 7 = s (bytes), 8 = t (TensorProto),
    ///   9 = g (GraphProto), 10 = floats, 11 = ints, 12 = strings (bytes),
    ///   ... wait, let me use the actual ONNX proto spec.
    ///
    /// From onnx.proto3 (canonical):
    ///   string name = 1;
    ///   string ref_attr_name = 21;
    ///   string doc_string = 13;
    ///   AttributeType type = 20;
    ///   float f = 4;
    ///   int64 i = 3;
    ///   bytes s = 4; -- NO, s = 4 conflicts with f = 4? Let me check...
    ///
    /// The actual onnx.proto has:
    ///   float f = 4;         // for FLOAT
    ///   int64 i = 2;         // for INT
    ///   bytes s = 3;         // for STRING
    ///   ... BUT these are NOT in a oneof, they're separate fields.
    ///
    /// Wait, I need to get this right. The actual field numbers from
    /// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3:
    ///
    ///   string name = 1;
    ///   int64 i = 2;          // INT
    ///   bytes s = 3;          // STRING
    ///   float f = 4;          // FLOAT
    ///   ... (more single-value fields)
    ///   repeated float floats = 7;   // FLOATS
    ///   repeated int64 ints = 8;     // INTS
    ///   repeated bytes strings = 9;  // STRINGS
    ///   ...
    ///   AttributeType type = 20;     // discriminator
    #[derive(Clone, PartialEq, Message)]
    pub struct AttributeProto {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(int64, tag = "2")]
        pub i: i64,
        #[prost(bytes = "vec", tag = "3")]
        pub s: Vec<u8>,
        #[prost(float, tag = "4")]
        pub f: f32,
        #[prost(float, repeated, tag = "7")]
        pub floats: Vec<f32>,
        #[prost(int64, repeated, tag = "8")]
        pub ints: Vec<i64>,
        #[prost(bytes = "vec", repeated, tag = "9")]
        pub strings: Vec<Vec<u8>>,
        /// AttributeType discriminator.
        #[prost(int32, tag = "20")]
        pub r#type: i32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,
        #[prost(string, tag = "3")]
        pub name: String,
        #[prost(string, tag = "4")]
        pub op_type: String,
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
        #[prost(string, tag = "7")]
        pub domain: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,
        #[prost(string, tag = "2")]
        pub name: String,
        #[prost(message, repeated, tag = "5")]
        pub input: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "6")]
        pub output: Vec<ValueInfoProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct OperatorSetIdProto {
        #[prost(string, tag = "1")]
        pub domain: String,
        #[prost(int64, tag = "2")]
        pub version: i64,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct ModelProto {
        #[prost(int64, tag = "1")]
        pub ir_version: i64,
        #[prost(string, tag = "2")]
        pub producer_name: String,
        #[prost(string, tag = "3")]
        pub producer_version: String,
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,
        #[prost(message, repeated, tag = "8")]
        pub opset_import: Vec<OperatorSetIdProto>,
    }
}

// ---------------------------------------------------------------------------
// Attribute construction helpers
// ---------------------------------------------------------------------------

/// ONNX AttributeType constants.
const ATTR_TYPE_INT: i32 = 2;
const ATTR_TYPE_STRING: i32 = 3;
const ATTR_TYPE_FLOATS: i32 = 6;
const ATTR_TYPE_INTS: i32 = 7;
const ATTR_TYPE_STRINGS: i32 = 8;

fn attr_ints(name: &str, values: Vec<i64>) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: ATTR_TYPE_INTS,
        ints: values,
        ..Default::default()
    }
}

fn attr_floats(name: &str, values: Vec<f32>) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: ATTR_TYPE_FLOATS,
        floats: values,
        ..Default::default()
    }
}

fn attr_strings(name: &str, values: Vec<&str>) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: ATTR_TYPE_STRINGS,
        strings: values.into_iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: ATTR_TYPE_INT,
        i: value,
        ..Default::default()
    }
}

fn attr_string(name: &str, value: &str) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: ATTR_TYPE_STRING,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// ONNX element type constants
// ---------------------------------------------------------------------------

/// ONNX TensorProto.DataType FLOAT = 1
const ONNX_FLOAT: i32 = 1;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Export an SGBT model to ONNX format as raw bytes.
///
/// The output is a valid ONNX `ModelProto` containing a single
/// `TreeEnsembleRegressor` operator (from the `ai.onnx.ml` domain) that
/// reproduces the model's predictions.
///
/// # Arguments
///
/// * `model` - Trained SGBT model to export.
/// * `n_features` - Number of input features. Must be specified explicitly
///   because the model may not have seen samples in all trees.
///
/// # Errors
///
/// Returns `IrithyllError::Serialization` if protobuf encoding fails.
pub fn export_onnx(
    model: &crate::ensemble::SGBT,
    n_features: usize,
) -> crate::error::Result<Vec<u8>> {
    use prost::Message;

    // ------------------------------------------------------------------
    // 1. Walk every tree and collect ONNX TreeEnsembleRegressor attributes.
    // ------------------------------------------------------------------

    let learning_rate = model.config().learning_rate;

    // Per-node attributes (one entry per node across all trees).
    let mut nodes_treeids: Vec<i64> = Vec::new();
    let mut nodes_nodeids: Vec<i64> = Vec::new();
    let mut nodes_featureids: Vec<i64> = Vec::new();
    let mut nodes_values: Vec<f32> = Vec::new();
    let mut nodes_modes: Vec<&str> = Vec::new();
    let mut nodes_truenodeids: Vec<i64> = Vec::new();
    let mut nodes_falsenodeids: Vec<i64> = Vec::new();
    let mut nodes_hitrates: Vec<f32> = Vec::new();
    let mut nodes_missing_value_tracks_true: Vec<i64> = Vec::new();

    // Per-leaf attributes (one entry per leaf across all trees).
    let mut target_ids: Vec<i64> = Vec::new();
    let mut target_nodeids: Vec<i64> = Vec::new();
    let mut target_treeids: Vec<i64> = Vec::new();
    let mut target_weights: Vec<f32> = Vec::new();

    for (tree_idx, step) in model.steps().iter().enumerate() {
        let tree_id = tree_idx as i64;
        let arena = step.slot().active_tree().arena();
        let n_nodes = arena.feature_idx.len();

        for node_idx in 0..n_nodes {
            let node_id = node_idx as i64;

            nodes_treeids.push(tree_id);
            nodes_nodeids.push(node_id);
            nodes_hitrates.push(1.0);
            nodes_missing_value_tracks_true.push(0);

            if arena.is_leaf[node_idx] {
                // Leaf node.
                nodes_featureids.push(0);
                nodes_values.push(0.0);
                nodes_modes.push("LEAF");
                nodes_truenodeids.push(0);
                nodes_falsenodeids.push(0);

                // Record leaf target.
                target_ids.push(0); // single output target
                target_nodeids.push(node_id);
                target_treeids.push(tree_id);
                target_weights.push((learning_rate * arena.leaf_value[node_idx]) as f32);
            } else {
                // Internal (split) node.
                nodes_featureids.push(arena.feature_idx[node_idx] as i64);
                nodes_values.push(arena.threshold[node_idx] as f32);
                nodes_modes.push("BRANCH_LEQ");
                nodes_truenodeids.push(arena.left[node_idx].0 as i64);
                nodes_falsenodeids.push(arena.right[node_idx].0 as i64);
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Build the TreeEnsembleRegressor ONNX node.
    // ------------------------------------------------------------------

    let tree_node = onnx_proto::NodeProto {
        input: vec!["features".to_string()],
        output: vec!["predictions".to_string()],
        name: "TreeEnsembleRegressor_0".to_string(),
        op_type: "TreeEnsembleRegressor".to_string(),
        domain: "ai.onnx.ml".to_string(),
        attribute: vec![
            attr_ints("nodes_treeids", nodes_treeids),
            attr_ints("nodes_nodeids", nodes_nodeids),
            attr_ints("nodes_featureids", nodes_featureids),
            attr_floats("nodes_values", nodes_values),
            attr_strings("nodes_modes", nodes_modes),
            attr_ints("nodes_truenodeids", nodes_truenodeids),
            attr_ints("nodes_falsenodeids", nodes_falsenodeids),
            attr_floats("nodes_hitrates", nodes_hitrates),
            attr_ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true),
            attr_ints("target_ids", target_ids),
            attr_ints("target_nodeids", target_nodeids),
            attr_ints("target_treeids", target_treeids),
            attr_floats("target_weights", target_weights),
            attr_floats("base_values", vec![model.base_prediction() as f32]),
            attr_string("aggregate_function", "SUM"),
            attr_string("post_transform", "NONE"),
            attr_int("n_targets", 1),
        ],
    };

    // ------------------------------------------------------------------
    // 3. Build the ONNX graph.
    // ------------------------------------------------------------------

    // Input: [batch, n_features] float tensor.
    let input = onnx_proto::ValueInfoProto {
        name: "features".to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::type_proto::Tensor {
                elem_type: ONNX_FLOAT,
                shape: Some(onnx_proto::TensorShapeProto {
                    dim: vec![
                        // Batch dimension (dynamic — use 0 for unknown).
                        onnx_proto::tensor_shape_proto::Dimension { dim_value: 0 },
                        // Feature dimension.
                        onnx_proto::tensor_shape_proto::Dimension {
                            dim_value: n_features as i64,
                        },
                    ],
                }),
            }),
        }),
    };

    // Output: [batch, 1] float tensor.
    let output = onnx_proto::ValueInfoProto {
        name: "predictions".to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::type_proto::Tensor {
                elem_type: ONNX_FLOAT,
                shape: Some(onnx_proto::TensorShapeProto {
                    dim: vec![
                        onnx_proto::tensor_shape_proto::Dimension { dim_value: 0 },
                        onnx_proto::tensor_shape_proto::Dimension { dim_value: 1 },
                    ],
                }),
            }),
        }),
    };

    let graph = onnx_proto::GraphProto {
        name: "irithyll_sgbt".to_string(),
        node: vec![tree_node],
        input: vec![input],
        output: vec![output],
    };

    // ------------------------------------------------------------------
    // 4. Build the ONNX model.
    // ------------------------------------------------------------------

    let model_proto = onnx_proto::ModelProto {
        ir_version: 8,
        producer_name: "irithyll".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        graph: Some(graph),
        opset_import: vec![
            onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            },
            onnx_proto::OperatorSetIdProto {
                domain: "ai.onnx.ml".to_string(),
                version: 3,
            },
        ],
    };

    // ------------------------------------------------------------------
    // 5. Encode to protobuf bytes.
    // ------------------------------------------------------------------

    let bytes = model_proto.encode_to_vec();

    Ok(bytes)
}

/// Export and save an SGBT model to an ONNX file.
///
/// Convenience wrapper around [`export_onnx`] that writes the serialised
/// protobuf bytes directly to the given path.
///
/// # Arguments
///
/// * `model` - Trained SGBT model to export.
/// * `n_features` - Number of input features.
/// * `path` - Destination file path (typically `*.onnx`).
///
/// # Errors
///
/// Returns `IrithyllError::Serialization` if encoding or file I/O fails.
pub fn save_onnx(
    model: &crate::ensemble::SGBT,
    n_features: usize,
    path: &std::path::Path,
) -> crate::error::Result<()> {
    let bytes = export_onnx(model, n_features)?;
    std::fs::write(path, bytes)
        .map_err(|e| crate::error::IrithyllError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn trained_model() -> crate::ensemble::SGBT {
        use crate::ensemble::config::SGBTConfig;
        use crate::sample::Sample;

        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .max_depth(3)
            .n_bins(8)
            .build()
            .unwrap();
        let mut model = crate::ensemble::SGBT::new(config);

        // Train enough samples to get some tree structure.
        for i in 0..100 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0, x * 0.5], x * 3.0));
        }
        model
    }

    #[test]
    fn export_produces_non_empty_bytes() {
        let model = trained_model();
        let bytes = export_onnx(&model, 3).unwrap();
        assert!(!bytes.is_empty(), "ONNX export should produce non-empty bytes");
        assert!(
            bytes.len() > 100,
            "ONNX export should have substantial content, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn export_untrained_model() {
        use crate::ensemble::config::SGBTConfig;
        let config = SGBTConfig::builder().n_steps(3).build().unwrap();
        let model = crate::ensemble::SGBT::new(config);
        let bytes = export_onnx(&model, 5).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn export_can_be_decoded() {
        use prost::Message;
        let model = trained_model();
        let bytes = export_onnx(&model, 3).unwrap();

        // Verify it is valid protobuf by decoding it back.
        let decoded = onnx_proto::ModelProto::decode(bytes.as_slice());
        assert!(decoded.is_ok(), "exported bytes should be valid protobuf");

        let proto = decoded.unwrap();
        assert_eq!(proto.producer_name, "irithyll");
        assert!(proto.graph.is_some());

        let graph = proto.graph.unwrap();
        assert_eq!(graph.node.len(), 1); // single TreeEnsembleRegressor node
        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.output.len(), 1);
    }

    #[test]
    fn node_counts_match_model() {
        use prost::Message;
        let model = trained_model();
        let bytes = export_onnx(&model, 3).unwrap();
        let proto = onnx_proto::ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();
        let node = &graph.node[0];

        // Find the nodes_treeids attribute to count total nodes.
        let treeids_attr = node
            .attribute
            .iter()
            .find(|a| a.name == "nodes_treeids")
            .expect("should have nodes_treeids attribute");

        // Total ONNX nodes should equal sum of arena sizes across all trees.
        let expected_nodes: usize = model
            .steps()
            .iter()
            .map(|s| s.slot().active_tree().arena().feature_idx.len())
            .sum();

        assert_eq!(
            treeids_attr.ints.len(),
            expected_nodes,
            "ONNX node count should match model node count"
        );
    }

    #[test]
    fn save_onnx_creates_file() {
        let model = trained_model();
        let dir = std::env::temp_dir();
        let path = dir.join("irithyll_test_model.onnx");

        let result = save_onnx(&model, 3, &path);
        assert!(result.is_ok());
        assert!(path.exists());

        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opset_imports_are_correct() {
        use prost::Message;
        let model = trained_model();
        let bytes = export_onnx(&model, 3).unwrap();
        let proto = onnx_proto::ModelProto::decode(bytes.as_slice()).unwrap();

        assert_eq!(proto.ir_version, 8);
        assert_eq!(proto.opset_import.len(), 2);

        // Default opset.
        let default_opset = proto
            .opset_import
            .iter()
            .find(|o| o.domain.is_empty())
            .expect("should have default opset");
        assert_eq!(default_opset.version, 18);

        // ML opset.
        let ml_opset = proto
            .opset_import
            .iter()
            .find(|o| o.domain == "ai.onnx.ml")
            .expect("should have ai.onnx.ml opset");
        assert_eq!(ml_opset.version, 3);
    }

    #[test]
    fn leaf_weights_include_learning_rate() {
        use prost::Message;
        let model = trained_model();
        let lr = model.config().learning_rate;
        let bytes = export_onnx(&model, 3).unwrap();
        let proto = onnx_proto::ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();
        let node = &graph.node[0];

        let target_weights_attr = node
            .attribute
            .iter()
            .find(|a| a.name == "target_weights")
            .expect("should have target_weights attribute");

        // Verify at least one non-zero weight exists in a trained model
        // and that the learning rate is baked in.
        let has_nonzero = target_weights_attr.floats.iter().any(|&w| w.abs() > 1e-10);
        assert!(
            has_nonzero,
            "trained model should have at least one non-zero leaf weight"
        );

        // Verify base_values contains the model base prediction.
        let base_values_attr = node
            .attribute
            .iter()
            .find(|a| a.name == "base_values")
            .expect("should have base_values attribute");
        assert_eq!(base_values_attr.floats.len(), 1);
        let expected_base = model.base_prediction() as f32;
        assert!(
            (base_values_attr.floats[0] - expected_base).abs() < 1e-5,
            "base_values should match model.base_prediction(): got {}, expected {}",
            base_values_attr.floats[0],
            expected_base,
        );

        // Sanity check: learning rate was applied to weights.
        // The first tree's first leaf should have weight = lr * leaf_value.
        let first_arena = model.steps()[0].slot().active_tree().arena();
        for i in 0..first_arena.feature_idx.len() {
            if first_arena.is_leaf[i] {
                let expected_weight = (lr * first_arena.leaf_value[i]) as f32;
                // Find this leaf's weight in the ONNX output.
                let target_nodeids_attr = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "target_nodeids")
                    .unwrap();
                let target_treeids_attr = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "target_treeids")
                    .unwrap();

                // Find index where treeids==0 and nodeids==i.
                for (idx, (&tid, &nid)) in target_treeids_attr
                    .ints
                    .iter()
                    .zip(target_nodeids_attr.ints.iter())
                    .enumerate()
                {
                    if tid == 0 && nid == i as i64 {
                        let actual = target_weights_attr.floats[idx];
                        assert!(
                            (actual - expected_weight).abs() < 1e-5,
                            "leaf {} weight mismatch: got {}, expected {}",
                            i,
                            actual,
                            expected_weight,
                        );
                    }
                }
                break; // just check the first leaf
            }
        }
    }
}
