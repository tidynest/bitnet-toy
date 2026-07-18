//! Hand-rolled byte-pair encoding (issue #24).
//!
//! Byte-level BPE, the GPT-2 family's core idea minus the regex
//! pre-splitter: the initial vocabulary is the 256 raw bytes, and
//! training greedily merges the most frequent adjacent token pair
//! into a new token until the requested vocabulary size is reached.
//! Because every token expands to a byte sequence, ANY input encodes
//! (no out-of-vocab), and `decode(encode(x)) == x` holds for
//! arbitrary UTF-8 - the round-trip property the tests pin.
//!
//! Determinism: ties on pair frequency break towards the smallest
//! `(left, right)` pair, so the same corpus and vocab size always
//! produce byte-identical merge lists (a DoD requirement - the `.bpe`
//! artefact must be reproducible).
//!
//! Encoding applies the merges in learned order over the whole token
//! stream - exactly the transformation training performed, so the
//! corpus tokenisation is identical to the final training-time state.
//! O(merges x len): fine for a one-time corpus pass and trivial for
//! prompts.

use std::collections::HashMap;
use std::io::{self, Read, Write};

/// Serialised artefact magic. u32 merge count follows, then the merge
/// pairs as little-endian u32 pairs in learned order.
const BPE_MAGIC: &[u8; 4] = b"BPE1";

/// Number of base tokens: one per byte value.
pub const BPE_BASE: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bpe {
    /// Merge list in learned order; merge `rank` fuses `(left, right)`
    /// into token id `BPE_BASE + rank`.
    pub merges: Vec<(u32, u32)>,
    /// Byte expansion of every token id (256 singletons, then one
    /// entry per merge). Derived from `merges`; kept for O(1) decode.
    token_bytes: Vec<Vec<u8>>,
}

impl Bpe {
    /// Learn `vocab_size - 256` merges from `corpus`. Stops early if no
    /// pair occurs at least twice (merging a singleton pair gains
    /// nothing and would make the tail of the merge list corpus-order
    /// noise).
    pub fn train(corpus: &[u8], vocab_size: usize) -> Bpe {
        assert!(
            vocab_size > BPE_BASE,
            "BPE vocab_size must exceed {BPE_BASE} (the byte alphabet)"
        );
        let mut stream: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();
        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(vocab_size - BPE_BASE);
        for rank in 0..(vocab_size - BPE_BASE) {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for w in stream.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
            // Most frequent pair; ties break towards the SMALLEST pair
            // for determinism.
            let best = counts
                .into_iter()
                .max_by(|(pa, ca), (pb, cb)| ca.cmp(cb).then_with(|| pb.cmp(pa)));
            let Some((pair, count)) = best else { break };
            if count < 2 {
                break;
            }
            let new_id = (BPE_BASE + rank) as u32;
            merges.push(pair);
            stream = merge_pass(&stream, pair, new_id);
        }
        Bpe::from_merges(merges)
    }

    /// Rebuild the full tokeniser (including the decode table) from a
    /// merge list - the deserialisation entry point.
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Bpe {
        let mut token_bytes: Vec<Vec<u8>> = (0..BPE_BASE as u32).map(|b| vec![b as u8]).collect();
        for &(a, b) in &merges {
            let mut bytes = token_bytes[a as usize].clone();
            bytes.extend_from_slice(&token_bytes[b as usize]);
            token_bytes.push(bytes);
        }
        Bpe {
            merges,
            token_bytes,
        }
    }

    /// Total vocabulary size: 256 byte tokens + one per merge.
    pub fn size(&self) -> usize {
        BPE_BASE + self.merges.len()
    }

    /// Encode arbitrary text: bytes, then every merge in learned order.
    /// Never fails - byte-level BPE has no out-of-vocab input.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut stream: Vec<u32> = text.bytes().map(u32::from).collect();
        for (rank, &pair) in self.merges.iter().enumerate() {
            stream = merge_pass(&stream, pair, (BPE_BASE + rank) as u32);
        }
        stream.into_iter().map(|t| t as usize).collect()
    }

    /// Decode ids back to text. Invalid UTF-8 (possible when sampling
    /// splits a multibyte char across generation boundaries) is
    /// replaced, not panicked on - generation output is display-only.
    /// Panics on out-of-range ids: that is a bug, not a runtime
    /// condition.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes[id]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Serialise: magic, u32 merge count, u32 LE pairs.
    pub fn save<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(BPE_MAGIC)?;
        w.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for &(a, b) in &self.merges {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialise a `save`d artefact.
    pub fn load<R: Read>(r: &mut R) -> io::Result<Bpe> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != BPE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a BPE1 tokeniser file (bad magic)",
            ));
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        let mut merges = Vec::with_capacity(n);
        for _ in 0..n {
            r.read_exact(&mut buf4)?;
            let a = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let b = u32::from_le_bytes(buf4);
            // A merge may only reference tokens that already exist.
            let limit = (BPE_BASE + merges.len()) as u32;
            if a >= limit || b >= limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("merge {} references future token ({a}, {b})", merges.len()),
                ));
            }
            merges.push((a, b));
        }
        Ok(Bpe::from_merges(merges))
    }
}

/// One left-to-right, non-overlapping merge pass: every adjacent
/// `(pair.0, pair.1)` becomes `new_id`.
fn merge_pass(stream: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(stream.len());
    let mut i = 0;
    while i < stream.len() {
        if i + 1 < stream.len() && stream[i] == pair.0 && stream[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(stream[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-merge fixture: "abab ab" - the dominant pair is (a, b).
    #[test]
    fn learns_the_obvious_merge_first() {
        let bpe = Bpe::train(b"ababab ababab", 258);
        assert_eq!(bpe.merges[0], (b'a' as u32, b'b' as u32));
        // "ab" now encodes to the single merged token.
        assert_eq!(bpe.encode("ab"), vec![BPE_BASE]);
        assert_eq!(bpe.decode(&[BPE_BASE]), "ab");
    }

    /// The DoD property: decode(encode(x)) == x for arbitrary UTF-8,
    /// including multibyte chars the training corpus never saw.
    #[test]
    fn encode_decode_round_trips_arbitrary_utf8() {
        let bpe = Bpe::train(b"the quick brown fox jumps over the lazy dog. ", 300);
        let mut lcg = 0x2545F4914F6CDD1Du64;
        let mut cases: Vec<String> = vec![
            String::new(),
            "the the the".into(),
            "completely unseen WORDS \n\t 123".into(),
            "multibyte: \u{00e9}\u{4e16}\u{754c} \u{1F600} caf\u{00e9}".into(),
        ];
        for _ in 0..50 {
            let mut s = String::new();
            for _ in 0..40 {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let c = char::from_u32((lcg >> 40) as u32 % 0x2FFF).unwrap_or('x');
                s.push(c);
            }
            cases.push(s);
        }
        for case in &cases {
            let ids = bpe.encode(case);
            assert_eq!(&bpe.decode(&ids), case, "round-trip failed");
            assert!(ids.iter().all(|&i| i < bpe.size()), "id out of range");
        }
    }

    /// Determinism: same corpus + size -> byte-identical artefacts.
    #[test]
    fn training_is_deterministic() {
        let corpus = b"to be or not to be that is the question ";
        let a = Bpe::train(corpus, 280);
        let b = Bpe::train(corpus, 280);
        assert_eq!(a.merges, b.merges);
        let (mut buf_a, mut buf_b) = (Vec::new(), Vec::new());
        a.save(&mut buf_a).unwrap();
        b.save(&mut buf_b).unwrap();
        assert_eq!(buf_a, buf_b);
    }

    /// Serialisation round-trip + corrupt-input rejection.
    #[test]
    fn save_load_round_trips_and_validates() {
        let bpe = Bpe::train(b"mississippi mississippi", 262);
        let mut buf = Vec::new();
        bpe.save(&mut buf).unwrap();
        let back = Bpe::load(&mut buf.as_slice()).unwrap();
        assert_eq!(back, bpe);
        assert!(Bpe::load(&mut &b"NOPE"[..]).is_err());
        // A merge referencing a not-yet-created token must be rejected.
        let mut evil = Vec::new();
        evil.extend_from_slice(b"BPE1");
        evil.extend_from_slice(&1u32.to_le_bytes());
        evil.extend_from_slice(&300u32.to_le_bytes());
        evil.extend_from_slice(&0u32.to_le_bytes());
        assert!(Bpe::load(&mut evil.as_slice()).is_err());
    }

    /// Merging is capped by corpus content: a request for more merges
    /// than useful pairs stops early rather than emitting noise.
    #[test]
    fn training_stops_when_no_pair_repeats() {
        let bpe = Bpe::train(b"abcdefg", 512);
        assert!(bpe.size() < 512, "cannot mint merges from singleton pairs");
    }
}
