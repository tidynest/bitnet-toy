//! Char-level data pipeline: text → vocab → sliding-window batches.
//!
//! Tokenisation: one character = one token.  Vocab size = unique-char count.
//! No special tokens (BOS/EOS/PAD) - our fixed-window training regime doesn't
//! need them, and skipping them keeps the vocab as small as possible.
//!
//! Batches are `(input_ids, target_ids)` pairs where target = input shifted by 1.
//! A model trained on these learns "given chars 0..N−1, predict char N."

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// Char ↔ id bidirectional mapping. Sorted lexicographically by char,
/// so the same input text always produces the same id assignment.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Forward mapping. BTreeMap (not HashMap) for deterministic iteration
    /// order - useful when you want to print the vocab and not see it shuffle.
    pub char_to_id: BTreeMap<char, usize>,
    /// Inverse mapping. id is just the index, so a Vec is enough.
    pub id_to_char: Vec<char>,
}

impl Vocab {
    /// Build vocab from the unique characters in `text`.
    /// Multiple occurrences of the same char are dedup'd; the resulting
    /// `id_to_char` is sorted by char ordinal (so id 0 = the first char in Unicode order).
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let id_to_char = chars.clone();
        let char_to_id = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        Self {
            char_to_id,
            id_to_char,
        }
    }

    /// Number of distinct tokens in the vocab.
    pub fn size(&self) -> usize {
        self.id_to_char.len()
    }

    /// True when `c` is in the vocab. Single-char building block for
    /// `can_encode`; public so prompt filters can drop out-of-vocab
    /// chars without rebuilding a char set from the corpus.
    pub fn can_encode_char(&self, c: char) -> bool {
        self.char_to_id.contains_key(&c)
    }

    /// True when every char of `text` is in the vocab, i.e. `encode`
    /// would succeed. Lets callers with *runtime* text (CLI prompts
    /// against an arbitrary training corpus, issue #8) check first
    /// instead of hitting `encode`'s intentional bug-catching panic.
    pub fn can_encode(&self, text: &str) -> bool {
        text.chars().all(|c| self.can_encode_char(c))
    }

    /// Encode a string to a vec of token ids.
    /// Panics if a char isn't in the vocab - a corpus/text mismatch is a bug,
    /// not a runtime condition. Failing loud here saves debugging later.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| {
                *self
                    .char_to_id
                    .get(&c)
                    .unwrap_or_else(|| panic!("char {:?} not in vocab", c))
            })
            .collect()
    }

    /// Decode ids back to a string. Panics on out-of-range id (bug, not runtime).
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                assert!(id < self.size(), "id {} >= vocab size {}", id, self.size());
                self.id_to_char[id]
            })
            .collect()
    }
}

/// All `(input, target)` sliding windows of length `seq_len` over `ids`.
///
/// Window `i` is  `(ids[i .. i+seq_len],  ids[i+1 .. i+seq_len+1])`.
/// Target is input shifted by 1 - the standard next-token-prediction setup.
///
/// Number of windows = `ids.len() - seq_len`. Caller must ensure
/// `ids.len() > seq_len` so at least one window exists.
pub fn make_windows(ids: &[usize], seq_len: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert!(
        ids.len() > seq_len,
        "corpus has {} ids, seq_len = {} - need ids.len() > seq_len",
        ids.len(),
        seq_len
    );

    let n = ids.len() - seq_len;
    let mut windows = Vec::with_capacity(n);
    for i in 0..n {
        let input = ids[i..i + seq_len].to_vec();
        let target = ids[i + 1..i + seq_len + 1].to_vec();
        windows.push((input, target));
    }
    windows
}

/// Read a UTF-8 corpus from disk. Trivial wrapper around `fs::read_to_string`,
/// kept here so callers don't need to remember the standard-library path.
pub fn read_corpus<P: AsRef<Path>>(path: P) -> io::Result<String> {
    fs::read_to_string(path)
}

/// Tiny linear-congruential generator. Shared deterministic randomness for
/// data shuffling, weight init helpers, and anything else that needs a
/// "random enough" seed without pulling the `rand` crate.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    pub fn gen_range(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % (n as u64)) as usize
    }

    /// Uniform sample in [0.0, 1.0). Uses the top 24 bits of `next_u64`
    /// (matching f32's 24-bit significand) and divides by 2^24.
    pub fn next_f01(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32
    }
}

/// Fisher-Yates shuffle in place using the given LCG. Deterministic for a
/// given seed so training runs are reproducible. Currently unused by the
/// training loop (which samples uniformly per step instead) but kept around
/// for callers that want a permuted view of any slice.
#[allow(dead_code)]
pub fn shuffle_in_place<T>(slice: &mut [T], rng: &mut Lcg) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.gen_range(i + 1);
        slice.swap(i, j);
    }
}

/// Tiny embedded corpus for tests + the M9 training run.
/// Four lines of Hamlet - short enough to read by eye, varied enough for a
/// non-trivial vocab (~30 chars: letters, space, newline). The fixed embedded
/// constant means tests don't depend on filesystem state.
pub const TINY_CORPUS: &str = "\
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_dedups_and_sorts_by_char_ordinal() {
        let v = Vocab::from_text("baba");
        assert_eq!(v.size(), 2);
        assert_eq!(v.id_to_char, vec!['a', 'b']);
        assert_eq!(v.char_to_id[&'a'], 0);
        assert_eq!(v.char_to_id[&'b'], 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let v = Vocab::from_text(TINY_CORPUS);
        let s = "to be or not to be";
        let ids = v.encode(s);
        let back = v.decode(&ids);
        assert_eq!(back, s, "encode→decode must be identity");
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_char_outside_vocab() {
        // Vocab has only 'a', 'b', 'c'; encoding 'z' must fail loudly.
        let v = Vocab::from_text("abc");
        let _ = v.encode("z");
    }

    #[test]
    fn can_encode_reports_vocab_membership() {
        let v = Vocab::from_text("abc");
        assert!(v.can_encode("cab"));
        assert!(v.can_encode("")); // vacuously encodable
        assert!(!v.can_encode("abz"));
    }

    #[test]
    fn make_windows_produces_correct_pairs() {
        // ids = [1, 2, 3, 4, 5, 6, 7], seq_len = 3
        //   window 0: ([1, 2, 3], [2, 3, 4])
        //   window 1: ([2, 3, 4], [3, 4, 5])
        //   window 2: ([3, 4, 5], [4, 5, 6])
        //   window 3: ([4, 5, 6], [5, 6, 7])
        let ids = vec![1, 2, 3, 4, 5, 6, 7];
        let w = make_windows(&ids, 3);
        assert_eq!(w.len(), 4);
        assert_eq!(w[0], (vec![1, 2, 3], vec![2, 3, 4]));
        assert_eq!(w[3], (vec![4, 5, 6], vec![5, 6, 7]));
    }

    #[test]
    #[should_panic]
    fn make_windows_panics_when_corpus_too_short() {
        // ids.len() == seq_len → 0 windows → assertion fires.
        let _ = make_windows(&[1, 2, 3], 3);
    }

    #[test]
    fn tiny_corpus_yields_reasonable_vocab_and_encoding() {
        let v = Vocab::from_text(TINY_CORPUS);

        // Letters + space + newline → at least 20 distinct chars.
        assert!(
            v.size() >= 20,
            "vocab size {} too small for tiny corpus",
            v.size()
        );

        // Round-trip the whole corpus through encode/decode.
        let ids = v.encode(TINY_CORPUS);
        let back = v.decode(&ids);
        assert_eq!(
            back, TINY_CORPUS,
            "full-corpus encode→decode must be identity"
        );
        assert_eq!(ids.len(), TINY_CORPUS.chars().count());
    }
}
