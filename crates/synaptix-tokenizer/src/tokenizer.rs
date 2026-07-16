use crate::encoding::{EncodeOptions, Encoding};
use crate::error::Result;
use crate::special_tokens::SpecialTokens;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding>;

    fn encode_pair(&self, a: &str, b: &str, add_special_tokens: bool) -> Result<Encoding>;

    fn encode_batch(&self, inputs: &[String], add_special_tokens: bool) -> Result<Vec<Encoding>>;

    fn encode_with_options(&self, input: &str, opts: &EncodeOptions) -> Result<Encoding> {
        let mut enc = self.encode(input, opts.add_special_tokens)?;
        apply_truncation(&mut enc, &opts.truncation);
        apply_padding_one(&mut enc, &opts.padding);
        Ok(enc)
    }

    fn encode_batch_with_options(&self, inputs: &[String], opts: &EncodeOptions) -> Result<Vec<Encoding>> {
        let mut out = self.encode_batch(inputs, opts.add_special_tokens)?;
        for enc in out.iter_mut() {
            apply_truncation(enc, &opts.truncation);
        }
        apply_padding_batch(&mut out, &opts.padding);
        Ok(out)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    fn decode_batch(&self, batches: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>>;

    fn vocab_size(&self, with_added: bool) -> usize;

    fn id_to_token(&self, id: u32) -> Option<String>;

    fn token_to_id(&self, token: &str) -> Option<u32>;

    fn special_tokens(&self) -> &SpecialTokens;
}

fn apply_truncation(enc: &mut Encoding, t: &crate::encoding::TruncationStrategy) {
    use crate::encoding::TruncationStrategy::*;
    match t {
        None => {}
        LongestFirst { max_length, stride, direction } => {
            enc.truncate(*max_length, *stride, *direction);
        }
    }
}

fn apply_padding_one(enc: &mut Encoding, p: &crate::encoding::PaddingStrategy) {
    use crate::encoding::PaddingStrategy::*;
    match p {
        None | Longest { .. } => {}
        MaxLength { length, pad_id, pad_token, direction } => {
            enc.pad(*length, *pad_id, 0, pad_token, *direction);
        }
    }
}

fn apply_padding_batch(out: &mut [Encoding], p: &crate::encoding::PaddingStrategy) {
    use crate::encoding::PaddingStrategy::*;
    match p {
        None => {}
        Longest { pad_id, pad_token, direction } => {
            let max = out.iter().map(|e| e.len()).max().unwrap_or(0);
            for enc in out.iter_mut() {
                enc.pad(max, *pad_id, 0, pad_token, *direction);
            }
        }
        MaxLength { length, pad_id, pad_token, direction } => {
            for enc in out.iter_mut() {
                enc.pad(*length, *pad_id, 0, pad_token, *direction);
            }
        }
    }
}
