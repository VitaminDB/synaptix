pub struct TextChunk {
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
}

pub fn chunk_by_chars(text: &str, max_chars: usize, overlap: usize) -> Vec<TextChunk> {
    assert!(overlap < max_chars, "overlap must be less than max_chars");
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }

    let step = max_chars - overlap;
    let mut chunks = Vec::new();
    let mut start_char = 0usize;

    while start_char < n {
        let end_char = (start_char + max_chars).min(n);
        let start_byte = chars[start_char].0;
        let end_byte = if end_char < n { chars[end_char].0 } else { text.len() };
        chunks.push(TextChunk {
            text: text[start_byte..end_byte].to_string(),
            start_byte,
            end_byte,
        });
        if end_char == n {
            break;
        }
        start_char += step;
    }

    chunks
}

pub fn chunk_by_words(text: &str, max_words: usize, overlap: usize) -> Vec<TextChunk> {
    assert!(overlap < max_words, "overlap must be less than max_words");
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n == 0 {
        return Vec::new();
    }

    let word_byte_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(n);
        let mut search_start = 0usize;
        for word in &words {
            let pos = text[search_start..].find(word).unwrap() + search_start;
            offsets.push(pos);
            search_start = pos + word.len();
        }
        offsets
    };

    let step = max_words - overlap;
    let mut chunks = Vec::new();
    let mut start_word = 0usize;

    while start_word < n {
        let end_word = (start_word + max_words).min(n);
        let start_byte = word_byte_offsets[start_word];
        let last_word_start = word_byte_offsets[end_word - 1];
        let last_word_len = words[end_word - 1].len();
        let end_byte = last_word_start + last_word_len;
        chunks.push(TextChunk {
            text: text[start_byte..end_byte].to_string(),
            start_byte,
            end_byte,
        });
        if end_word == n {
            break;
        }
        start_word += step;
    }

    chunks
}

pub fn chunk_by_tokens(
    text: &str,
    max_tokens: usize,
    overlap: usize,
    token_fn: impl Fn(&str) -> usize,
) -> Vec<TextChunk> {
    assert!(overlap < max_tokens, "overlap must be less than max_tokens");
    let sentences: Vec<&str> = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let sentence_byte_starts: Vec<usize> = {
        let mut offsets = Vec::with_capacity(sentences.len());
        let mut search = 0usize;
        for sent in &sentences {
            let pos = text[search..].find(sent).unwrap_or(0) + search;
            offsets.push(pos);
            search = pos + sent.len();
        }
        offsets
    };

    let step_tokens = max_tokens - overlap;
    let mut chunks = Vec::new();
    let mut i = 0usize;

    while i < sentences.len() {
        let mut tokens = 0usize;
        let mut j = i;
        while j < sentences.len() {
            let t = token_fn(sentences[j]);
            if tokens + t > max_tokens && j > i {
                break;
            }
            tokens += t;
            j += 1;
            if tokens >= max_tokens {
                break;
            }
        }
        if j == i {
            j = i + 1;
        }
        let start_byte = sentence_byte_starts[i];
        let end_byte = if j < sentences.len() {
            sentence_byte_starts[j]
        } else {
            text.len()
        };
        chunks.push(TextChunk {
            text: text[start_byte..end_byte].trim().to_string(),
            start_byte,
            end_byte,
        });
        if j >= sentences.len() {
            break;
        }
        let mut advanced = 0usize;
        let mut k = i;
        while k < j && advanced < step_tokens {
            advanced += token_fn(sentences[k]);
            k += 1;
        }
        i = k.max(i + 1);
    }

    chunks
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    while i < len {
        if bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?' {
            let end = i + 1;
            let slice = text[start..end].trim();
            if !slice.is_empty() {
                sentences.push(&text[start..end]);
            }
            start = end;
            while start < len && bytes[start] == b' ' {
                start += 1;
            }
            i = start;
        } else {
            i += 1;
        }
    }
    if start < len {
        let slice = text[start..].trim();
        if !slice.is_empty() {
            sentences.push(&text[start..]);
        }
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text);
    }
    sentences
}
