use synaptix_bundle::StDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutDtype {
    /// Квантованные тензоры — как есть (блоб `.qpacked` + манифест
    /// `ggml:<тип>`), плавающие — своим типом. Умолчание конвертации.
    Keep,
    /// Квантованные → F16, плавающие — своим типом (прежнее `auto`).
    Auto,
    F16,
    BF16,
    F32,
}

impl OutDtype {
    /// Оставить ли квантованный тензор в блоках ggml.
    pub fn keeps_quant(self, src: crate::ggml::GgmlType) -> bool {
        self == OutDtype::Keep && src.is_weight_format()
    }

    pub fn resolve(self, src: crate::ggml::GgmlType) -> StDtype {
        use crate::ggml::GgmlType as G;
        match self {
            OutDtype::F16 => StDtype::F16,
            OutDtype::BF16 => StDtype::BF16,
            OutDtype::F32 => StDtype::F32,
            OutDtype::Auto | OutDtype::Keep => match src {
                G::F32 => StDtype::F32,
                G::F16 => StDtype::F16,
                G::BF16 => StDtype::BF16,
                G::F64 => StDtype::F32,
                G::I8 => StDtype::I8,
                G::I16 => StDtype::I16,
                G::I32 => StDtype::I32,
                G::I64 => StDtype::I64,
                _ => StDtype::F16,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    None,

    LogNeg,
    SubOne,
}

#[derive(Debug, Clone)]
pub enum Producer {

    Direct(String),

    Interleave { parts: Vec<String>, block: usize },

    PermuteRows {
        src: String,
        row_elems: usize,
        map: Vec<u32>,
    },

    PermuteCols {
        src: String,
        row_elems: usize,
        block: usize,
        map: Vec<u32>,
    },

    /// Стопка экспертов `[E, ΣNᵢ, K]`: срез `e` — строки среза `e` каждой из
    /// частей `[E, Nᵢ, K]` подряд (`ffn_gate_exps` + `ffn_up_exps` →
    /// `experts.gate_up_proj`). Строки не режутся, квант блоков сохраняется.
    StackConcat { parts: Vec<String> },
}

impl Producer {
    pub fn sources(&self) -> &[String] {
        match self {
            Producer::Interleave { parts, .. } | Producer::StackConcat { parts } => parts,
            _ => std::slice::from_ref(self.first()),
        }
    }
    fn first(&self) -> &String {
        match self {
            Producer::Direct(s) => s,
            Producer::Interleave { parts, .. } | Producer::StackConcat { parts } => &parts[0],
            Producer::PermuteRows { src, .. } => src,
            Producer::PermuteCols { src, .. } => src,
        }
    }
}

pub fn value_head_map(num_value_heads: usize, num_key_heads: usize) -> Vec<u32> {
    // `max(1)`: при голов ключей больше, чем голов значений (битые
    // метаданные), `group` был нулём, и `j % group` делил на ноль.
    let group = (num_value_heads / num_key_heads.max(1)).max(1);
    (0..num_value_heads)
        .map(|j| ((j % group) * num_key_heads + j / group) as u32)
        .collect()
}

#[derive(Debug, Clone)]
pub struct MappedTensor {

    pub hf_name: String,
    pub producer: Producer,

    pub shape: Option<Vec<usize>>,
    pub transform: Transform,
}

impl MappedTensor {
    pub fn direct(hf_name: impl Into<String>, gguf: impl Into<String>) -> Self {
        Self {
            hf_name: hf_name.into(),
            producer: Producer::Direct(gguf.into()),
            shape: None,
            transform: Transform::None,
        }
    }
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }
    pub fn with_transform(mut self, t: Transform) -> Self {
        self.transform = t;
        self
    }
}

#[derive(Debug, Clone)]
pub struct MappedFile {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Component {
    pub name: String,
    pub tensors: Vec<MappedTensor>,
}

#[derive(Debug, Clone)]
pub struct ConversionPlan {
    pub bundle_id: String,
    pub arch: String,
    pub components: Vec<Component>,
    pub files: Vec<MappedFile>,
}

impl ConversionPlan {
    pub fn tensor_count(&self) -> usize {
        self.components.iter().map(|c| c.tensors.len()).sum()
    }
}
