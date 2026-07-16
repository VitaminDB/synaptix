#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EosToken,
    MaxLength,
    StopString,
    UserRequested,
}

pub trait StopCriteria: Send {
    fn should_stop(&self, generated: &[u32], new_token: u32) -> Option<StopReason>;
}

pub struct EosStop {
    pub eos_token_id: u32,
}

impl StopCriteria for EosStop {
    fn should_stop(&self, _: &[u32], new: u32) -> Option<StopReason> {
        if new == self.eos_token_id { Some(StopReason::EosToken) } else { None }
    }
}

pub struct MaxLengthStop {
    pub max_new_tokens: usize,
}

impl StopCriteria for MaxLengthStop {
    fn should_stop(&self, generated: &[u32], _: u32) -> Option<StopReason> {
        if generated.len() >= self.max_new_tokens { Some(StopReason::MaxLength) } else { None }
    }
}

pub struct StopStringStop {
    pub stop_strings: Vec<String>,
    pub decode_fn: Box<dyn Fn(&[u32]) -> String + Send>,
}

impl StopCriteria for StopStringStop {
    fn should_stop(&self, generated: &[u32], _: u32) -> Option<StopReason> {
        let text = (self.decode_fn)(generated);
        for s in &self.stop_strings {
            if text.contains(s.as_str()) { return Some(StopReason::StopString); }
        }
        None
    }
}

pub struct CompoundStop {
    pub criteria: Vec<Box<dyn StopCriteria>>,
}

impl StopCriteria for CompoundStop {
    fn should_stop(&self, generated: &[u32], new: u32) -> Option<StopReason> {
        for c in &self.criteria {
            if let Some(r) = c.should_stop(generated, new) { return Some(r); }
        }
        None
    }
}
