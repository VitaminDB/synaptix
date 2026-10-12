use crate::graph::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Qualcomm,
    Intel,
}

#[derive(Debug)]
pub struct NpuModel {
    pub id: &'static str,
    pub vendor: Vendor,
    pub name: &'static str,
    pub arch: u32,
    pub socs: &'static [&'static str],
    pub activations: &'static [DType],
    pub weights: &'static [DType],
    pub weight_bits: &'static [u8],
    pub fp16_native: bool,
    pub fused_norms: bool,
    pub session_bytes: u64,
}

impl NpuModel {
    pub fn supports_activation(&self, d: DType) -> bool {
        self.activations.contains(&d)
    }

    pub fn supports_weight(&self, d: DType) -> bool {
        self.weights.contains(&d)
    }
}

const GIB: u64 = 1 << 30;

const QC_ACT: &[DType] = &[DType::F16, DType::F32, DType::U8, DType::U16, DType::I32, DType::U32];
const QC_W_V68: &[DType] = &[DType::F16, DType::F32, DType::U8, DType::I8];
const QC_W_V73: &[DType] = &[DType::F16, DType::F32, DType::U8, DType::I8, DType::I4];

pub static MODELS: &[NpuModel] = &[
    NpuModel {
        id: "qualcomm-htp-v68",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v68",
        arch: 68,
        socs: &["SM8350"],
        activations: QC_ACT,
        weights: QC_W_V68,
        weight_bits: &[8, 16],
        fp16_native: false,
        fused_norms: false,
        session_bytes: 2 * GIB,
    },
    NpuModel {
        id: "qualcomm-htp-v69",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v69",
        arch: 69,
        socs: &["SM8450", "SM8475", "SM7475"],
        activations: QC_ACT,
        weights: QC_W_V68,
        weight_bits: &[8, 16],
        fp16_native: true,
        fused_norms: true,
        session_bytes: 2 * GIB,
    },
    NpuModel {
        id: "qualcomm-htp-v73",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v73",
        arch: 73,
        socs: &["SM8550", "SM7550"],
        activations: QC_ACT,
        weights: QC_W_V73,
        weight_bits: &[4, 8, 16],
        fp16_native: true,
        fused_norms: true,
        session_bytes: 3 * GIB,
    },
    NpuModel {
        id: "qualcomm-htp-v75",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v75",
        arch: 75,
        socs: &["SM8650", "SM7675"],
        activations: QC_ACT,
        weights: QC_W_V73,
        weight_bits: &[4, 8, 16],
        fp16_native: true,
        fused_norms: true,
        session_bytes: 3 * GIB,
    },
    NpuModel {
        id: "qualcomm-htp-v79",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v79",
        arch: 79,
        socs: &["SM8750"],
        activations: QC_ACT,
        weights: QC_W_V73,
        weight_bits: &[4, 8, 16],
        fp16_native: true,
        fused_norms: true,
        session_bytes: 3 * GIB,
    },
    NpuModel {
        id: "qualcomm-htp-v81",
        vendor: Vendor::Qualcomm,
        name: "Hexagon HTP v81",
        arch: 81,
        socs: &["SM8850"],
        activations: QC_ACT,
        weights: QC_W_V73,
        weight_bits: &[4, 8, 16],
        fp16_native: true,
        fused_norms: true,
        session_bytes: 3 * GIB,
    },
    // Intel Core Ultra (NPU 37xx/40xx, «AI Boost») через OpenVINO: F16 и int8-веса
    // по каналу; память — общая с системой.
    NpuModel {
        id: "intel-npu",
        vendor: Vendor::Intel,
        name: "Intel AI Boost",
        arch: 37,
        socs: &["Meteor Lake", "Arrow Lake", "Lunar Lake"],
        activations: &[DType::F16, DType::F32, DType::I32, DType::U32, DType::U8],
        weights: &[DType::F16, DType::F32, DType::I8, DType::U8, DType::I4],
        weight_bits: &[4, 8, 16],
        fp16_native: true,
        fused_norms: false,
        session_bytes: 8 * GIB,
    },
];

pub fn by_id(id: &str) -> Option<&'static NpuModel> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn find(vendor: Vendor, arch: u32) -> Option<&'static NpuModel> {
    MODELS.iter().find(|m| m.vendor == vendor && m.arch == arch)
}
