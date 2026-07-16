use serde::{Deserialize, Serialize};

fn default_num_classes() -> usize { 80 }
fn default_input_size() -> usize { 640 }
fn default_backbone_channels() -> Vec<usize> { vec![64, 128, 256, 512, 1024] }
fn default_anchors_per_cell() -> usize { 3 }
fn default_conf_threshold() -> f64 { 0.25 }
fn default_nms_threshold() -> f64 { 0.45 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloConfig {
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_input_size")]
    pub input_size: usize,
    #[serde(default = "default_backbone_channels")]
    pub backbone_channels: Vec<usize>,
    #[serde(default = "default_anchors_per_cell")]
    pub anchors_per_cell: usize,
    #[serde(default = "default_conf_threshold")]
    pub conf_threshold: f64,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f64,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            num_classes: default_num_classes(),
            input_size: default_input_size(),
            backbone_channels: default_backbone_channels(),
            anchors_per_cell: default_anchors_per_cell(),
            conf_threshold: default_conf_threshold(),
            nms_threshold: default_nms_threshold(),
        }
    }
}
