pub fn pack_sequences(sequences: &[Vec<u32>], max_len: usize, pad_id: u32) -> Vec<Vec<u32>> {
    let mut bins: Vec<Vec<u32>> = Vec::new();
    let mut bin_lens: Vec<usize> = Vec::new();

    for seq in sequences {
        let seq_len = seq.len().min(max_len);
        let truncated = &seq[..seq_len];
        let mut placed = false;
        for (i, bin) in bins.iter_mut().enumerate() {
            if bin_lens[i] + seq_len <= max_len {
                bin.extend_from_slice(truncated);
                bin_lens[i] += seq_len;
                placed = true;
                break;
            }
        }
        if !placed {
            let mut new_bin = Vec::with_capacity(max_len);
            new_bin.extend_from_slice(truncated);
            bin_lens.push(seq_len);
            bins.push(new_bin);
        }
    }

    for (i, bin) in bins.iter_mut().enumerate() {
        while bin_lens[i] < max_len {
            bin.push(pad_id);
            bin_lens[i] += 1;
        }
    }

    bins
}

pub fn pack_with_attention_mask(
    sequences: &[Vec<u32>],
    max_len: usize,
    pad_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    let mut bins: Vec<Vec<u32>> = Vec::new();
    let mut masks: Vec<Vec<u8>> = Vec::new();
    let mut bin_lens: Vec<usize> = Vec::new();

    for seq in sequences {
        let seq_len = seq.len().min(max_len);
        let truncated = &seq[..seq_len];
        let mut placed = false;
        for (i, bin) in bins.iter_mut().enumerate() {
            if bin_lens[i] + seq_len <= max_len {
                bin.extend_from_slice(truncated);
                masks[i].extend(std::iter::repeat(1u8).take(seq_len));
                bin_lens[i] += seq_len;
                placed = true;
                break;
            }
        }
        if !placed {
            let mut new_bin = Vec::with_capacity(max_len);
            let mut new_mask = Vec::with_capacity(max_len);
            new_bin.extend_from_slice(truncated);
            new_mask.extend(std::iter::repeat(1u8).take(seq_len));
            bin_lens.push(seq_len);
            bins.push(new_bin);
            masks.push(new_mask);
        }
    }

    for (i, bin) in bins.iter_mut().enumerate() {
        while bin_lens[i] < max_len {
            bin.push(pad_id);
            masks[i].push(0u8);
            bin_lens[i] += 1;
        }
    }

    (bins, masks)
}

pub fn truncate_and_pad(seq: &[u32], max_len: usize, pad_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(max_len);
    let copy_len = seq.len().min(max_len);
    out.extend_from_slice(&seq[..copy_len]);
    while out.len() < max_len {
        out.push(pad_id);
    }
    out
}
