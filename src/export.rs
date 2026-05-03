//! Binary export and import of trained model weights.
//!
//! Two formats:
//!   - `Format::Float32`  every weight as raw f32 (4 bytes per value). Baseline.
//!   - `Format::Ternary`  embeddings stay f32; every BitLinear weight (block
//!                        weights and the LM head) becomes `(γ f32, ternary i8/value)`.
//!
//! On-disk layout starts with a 33-byte header so importers can sanity-check
//! the file and reconstruct a `ModelConfig` without external metadata.
//!
//! ```text
//! Header (29 bytes):
//!   4 B   magic "BNT1"
//!   1 B   format byte: 0 = Float32, 1 = Ternary
//!   4 B   vocab_size  (u32 LE)
//!   4 B   hidden_dim  (u32 LE)
//!   4 B   head_dim    (u32 LE)
//!   4 B   ffn_dim     (u32 LE)
//!   4 B   max_seq_len (u32 LE)
//!   4 B   n_blocks    (u32 LE)
//!
//! Payload (positional, in this order):
//!   token_embed, pos_embed              (always f32)
//!   per block: attn Q, K, V, O; ffn up, down
//!   lm_head
//! ```
//!
// Many functions in this module are called only from main() (which doesn't
// exist in the cargo-test binary) plus the tests in this file. The dead-code
// lint can't see the cross-binary usage; allow at module level.
#![allow(dead_code)]

//! Round-trip caveat: the importer reads dequantised values
//! (`γ · W_q`) into the master tensors. The first forward through STE
//! quantisation will scale these slightly because `mean(|γ · W_q|) = γ · f`
//! where `f` is the fraction of non-zero ternary entries. The output magnitude
//! shifts by a factor of `f` compared to the export-time forward. The structural
//! information (which weights are non-zero, signs) is preserved exactly.

use crate::bitlinear::absmean_ternary;
use crate::model::{BlockMasters, Model, ModelConfig};
use crate::tensor::Tensor;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"BNT1";
const HEADER_SIZE: usize = 4 + 1 + 6 * 4; // magic + format byte + 6 u32 fields = 29

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Every weight as raw f32 (4 bytes per value).
    Float32,
    /// BitLinear weights as `(γ f32, one i8 per ternary)`. Embeddings stay f32.
    /// Round-trippable via `xxd` for debugging.
    Ternary,
    /// BitLinear weights as `(γ f32, base-3 packed bytes)`. 5 ternary values
    /// fit in a single byte (3^5 = 243 < 256). Closest to the "1.58 bits per
    /// weight" theoretical optimum with a fixed-byte container.
    TernaryPacked,
}

impl Format {
    fn byte(self) -> u8 {
        match self {
            Format::Float32 => 0,
            Format::Ternary => 1,
            Format::TernaryPacked => 2,
        }
    }
    fn from_byte(b: u8) -> io::Result<Format> {
        match b {
            0 => Ok(Format::Float32),
            1 => Ok(Format::Ternary),
            2 => Ok(Format::TernaryPacked),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown format byte: {}", other),
            )),
        }
    }
}

// ---- header ----

fn write_header<W: Write>(w: &mut W, cfg: &ModelConfig, fmt: Format) -> io::Result<usize> {
    w.write_all(MAGIC)?;
    w.write_all(&[fmt.byte()])?;
    for v in [
        cfg.vocab_size as u32,
        cfg.hidden_dim as u32,
        cfg.head_dim as u32,
        cfg.ffn_dim as u32,
        cfg.max_seq_len as u32,
        cfg.n_blocks as u32,
    ] {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(HEADER_SIZE)
}

fn read_header<R: Read>(r: &mut R) -> io::Result<(Format, ModelConfig)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a BNT1 file (bad magic)",
        ));
    }
    let mut fb = [0u8; 1];
    r.read_exact(&mut fb)?;
    let fmt = Format::from_byte(fb[0])?;

    let mut buf = [0u8; 4];
    let mut read_u32 = |r: &mut R| -> io::Result<u32> {
        r.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    };

    let vocab_size = read_u32(r)? as usize;
    let hidden_dim = read_u32(r)? as usize;
    let head_dim = read_u32(r)? as usize;
    let ffn_dim = read_u32(r)? as usize;
    let max_seq_len = read_u32(r)? as usize;
    let n_blocks = read_u32(r)? as usize;

    Ok((
        fmt,
        ModelConfig {
            vocab_size,
            hidden_dim,
            head_dim,
            ffn_dim,
            max_seq_len,
            n_blocks,
        },
    ))
}

// ---- per-tensor helpers ----

fn write_f32_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<usize> {
    let mut bytes = 0;
    for &v in &t.data {
        w.write_all(&v.to_le_bytes())?;
        bytes += 4;
    }
    Ok(bytes)
}

fn write_ternary_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<usize> {
    let (w_q, gamma) = absmean_ternary(t);
    let mut bytes = 0;
    w.write_all(&gamma.to_le_bytes())?;
    bytes += 4;
    for &v in &w_q.data {
        let i = v as i8;
        w.write_all(&[i as u8])?;
        bytes += 1;
    }
    Ok(bytes)
}

fn read_f32_tensor<R: Read>(r: &mut R, shape: Vec<usize>) -> io::Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        data.push(f32::from_le_bytes(buf));
    }
    Ok(Tensor { data, shape })
}

/// Pack 5 ternary values (each in {-1, 0, +1}) into one byte using base-3:
///   byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4
/// where each `dk` is 0, 1, or 2 corresponding to -1, 0, +1.
/// Max byte value is 5 * 2 * (1 + 3 + 9 + 27 + 81) ... actually
/// max packed = 2 * (1 + 3 + 9 + 27 + 81) = 242, fits in u8.
fn pack_ternary_chunk(chunk: &[i8]) -> u8 {
    let to_d = |v: i8| -> u8 {
        // -1 -> 0, 0 -> 1, +1 -> 2
        (v + 1) as u8
    };
    let mut byte: u32 = 0;
    let mut place: u32 = 1;
    for &v in chunk.iter().take(5) {
        byte += u32::from(to_d(v)) * place;
        place *= 3;
    }
    byte as u8
}

fn unpack_ternary_byte(b: u8) -> [i8; 5] {
    let mut out = [0i8; 5];
    let mut x = u32::from(b);
    for slot in &mut out {
        let d = (x % 3) as i8;
        x /= 3;
        *slot = d - 1; // 0,1,2 -> -1,0,+1
    }
    out
}

fn write_ternary_packed_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<usize> {
    let (w_q, gamma) = absmean_ternary(t);
    let mut bytes = 0;
    w.write_all(&gamma.to_le_bytes())?;
    bytes += 4;
    let i8_data: Vec<i8> = w_q.data.iter().map(|&v| v as i8).collect();
    for chunk in i8_data.chunks(5) {
        let b = pack_ternary_chunk(chunk);
        w.write_all(&[b])?;
        bytes += 1;
    }
    Ok(bytes)
}

fn read_ternary_packed_tensor<R: Read>(r: &mut R, shape: Vec<usize>) -> io::Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    let gamma = f32::from_le_bytes(buf);

    let n_bytes = n.div_ceil(5);
    let mut data = Vec::with_capacity(n);
    let mut byte = [0u8; 1];
    for _ in 0..n_bytes {
        r.read_exact(&mut byte)?;
        let unpacked = unpack_ternary_byte(byte[0]);
        for &v in &unpacked {
            if data.len() < n {
                data.push(f32::from(v) * gamma);
            }
        }
    }
    Ok(Tensor { data, shape })
}

fn read_ternary_tensor<R: Read>(r: &mut R, shape: Vec<usize>) -> io::Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    let gamma = f32::from_le_bytes(buf);
    let mut data = Vec::with_capacity(n);
    let mut byte = [0u8; 1];
    for _ in 0..n {
        r.read_exact(&mut byte)?;
        let qi = byte[0] as i8;
        // Reconstruct dequantised value γ · W_q. See module docs for the
        // re-quantisation caveat.
        data.push(f32::from(qi) * gamma);
    }
    Ok(Tensor { data, shape })
}

// ---- public export ----

pub fn export_f32<W: Write>(model: &Model, w: &mut W) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::Float32)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        total += write_f32_tensor(w, &b.attn_w_q)?;
        total += write_f32_tensor(w, &b.attn_w_k)?;
        total += write_f32_tensor(w, &b.attn_w_v)?;
        total += write_f32_tensor(w, &b.attn_w_o)?;
        total += write_f32_tensor(w, &b.ffn_up_w)?;
        total += write_f32_tensor(w, &b.ffn_down_w)?;
    }
    total += write_f32_tensor(w, &model.lm_head)?;
    Ok(total)
}

pub fn export_ternary<W: Write>(model: &Model, w: &mut W) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::Ternary)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        total += write_ternary_tensor(w, &b.attn_w_q)?;
        total += write_ternary_tensor(w, &b.attn_w_k)?;
        total += write_ternary_tensor(w, &b.attn_w_v)?;
        total += write_ternary_tensor(w, &b.attn_w_o)?;
        total += write_ternary_tensor(w, &b.ffn_up_w)?;
        total += write_ternary_tensor(w, &b.ffn_down_w)?;
    }
    // LM head is also a BitLinear, so it's ternary too.
    total += write_ternary_tensor(w, &model.lm_head)?;
    Ok(total)
}

/// Most compact format. BitLinear weights packed 5 ternaries per byte (base-3).
pub fn export_ternary_packed<W: Write>(model: &Model, w: &mut W) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::TernaryPacked)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        total += write_ternary_packed_tensor(w, &b.attn_w_q)?;
        total += write_ternary_packed_tensor(w, &b.attn_w_k)?;
        total += write_ternary_packed_tensor(w, &b.attn_w_v)?;
        total += write_ternary_packed_tensor(w, &b.attn_w_o)?;
        total += write_ternary_packed_tensor(w, &b.ffn_up_w)?;
        total += write_ternary_packed_tensor(w, &b.ffn_down_w)?;
    }
    total += write_ternary_packed_tensor(w, &model.lm_head)?;
    Ok(total)
}

// ---- public import ----

/// Read a model file (either format) and reconstruct the `Model` plus the
/// `Format` it was stored in.
pub fn import<R: Read>(r: &mut R) -> io::Result<(Model, Format)> {
    let (fmt, cfg) = read_header(r)?;
    let h = cfg.hidden_dim;
    let d = cfg.head_dim;
    let f = cfg.ffn_dim;

    let token_embed = read_f32_tensor(r, vec![cfg.vocab_size, h])?;
    let pos_embed = read_f32_tensor(r, vec![cfg.max_seq_len, h])?;

    let mut blocks = Vec::with_capacity(cfg.n_blocks);
    for _ in 0..cfg.n_blocks {
        let read_w = |r: &mut R, shape: Vec<usize>| match fmt {
            Format::Float32 => read_f32_tensor(r, shape),
            Format::Ternary => read_ternary_tensor(r, shape),
            Format::TernaryPacked => read_ternary_packed_tensor(r, shape),
        };
        blocks.push(BlockMasters {
            attn_w_q: read_w(r, vec![h, d])?,
            attn_w_k: read_w(r, vec![h, d])?,
            attn_w_v: read_w(r, vec![h, d])?,
            attn_w_o: read_w(r, vec![d, h])?,
            ffn_up_w: read_w(r, vec![h, f])?,
            ffn_down_w: read_w(r, vec![f, h])?,
        });
    }

    let lm_head = match fmt {
        Format::Float32 => read_f32_tensor(r, vec![h, cfg.vocab_size])?,
        Format::Ternary => read_ternary_tensor(r, vec![h, cfg.vocab_size])?,
        Format::TernaryPacked => read_ternary_packed_tensor(r, vec![h, cfg.vocab_size])?,
    };

    Ok((
        Model {
            token_embed,
            pos_embed,
            blocks,
            lm_head,
            config: cfg,
        },
        fmt,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_model() -> Model {
        let cfg = ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            head_dim: 4,
            ffn_dim: 8,
            max_seq_len: 4,
            n_blocks: 2,
        };
        Model::new(&cfg, 0)
    }

    #[test]
    fn f32_export_size_matches_total_param_count() {
        // Header (33) + 336 floats × 4 = 1344 + 33 = 1377 bytes.
        let expected = HEADER_SIZE + 336 * 4;
        let mut buf = Vec::new();
        let bytes = export_f32(&tiny_model(), &mut buf).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn ternary_export_size_matches_expected_layout() {
        // Header (33) + embeddings (192) + blocks (304) + lm_head ternary (4 + 32 = 36) = 565.
        let expected = HEADER_SIZE + 192 + 304 + 36;
        let mut buf = Vec::new();
        let bytes = export_ternary(&tiny_model(), &mut buf).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn ternary_export_is_smaller_than_f32() {
        let model = tiny_model();
        let mut f32_buf = Vec::new();
        let mut ter_buf = Vec::new();
        export_f32(&model, &mut f32_buf).unwrap();
        export_ternary(&model, &mut ter_buf).unwrap();
        assert!(
            ter_buf.len() < f32_buf.len(),
            "ternary export ({} B) must be smaller than f32 ({} B)",
            ter_buf.len(),
            f32_buf.len()
        );
    }

    #[test]
    fn f32_round_trip_preserves_every_weight_exactly() {
        let model = tiny_model();
        let mut buf = Vec::new();
        export_f32(&model, &mut buf).unwrap();

        let (loaded, fmt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Float32);
        assert_eq!(loaded.config.vocab_size, model.config.vocab_size);
        assert_eq!(loaded.token_embed.data, model.token_embed.data);
        assert_eq!(loaded.lm_head.data, model.lm_head.data);
        assert_eq!(loaded.blocks.len(), model.blocks.len());
        for (lb, mb) in loaded.blocks.iter().zip(&model.blocks) {
            assert_eq!(lb.attn_w_q.data, mb.attn_w_q.data);
            assert_eq!(lb.ffn_up_w.data, mb.ffn_up_w.data);
        }
    }

    #[test]
    fn ternary_round_trip_preserves_block_weight_signs() {
        // Round-trip discards f32 master precision in BitLinear weights but
        // must preserve the (γ, W_q) decomposition exactly.
        let model = tiny_model();
        let mut buf = Vec::new();
        export_ternary(&model, &mut buf).unwrap();

        let (loaded, fmt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Ternary);

        // Embeddings are f32, exact match.
        assert_eq!(loaded.token_embed.data, model.token_embed.data);

        // Block weights and lm_head: every loaded value must equal γ · w_q where
        // (w_q, γ) = absmean_ternary(original master). That's the exact contract.
        for (loaded_b, master_b) in loaded.blocks.iter().zip(&model.blocks) {
            let (w_q, gamma) = absmean_ternary(&master_b.attn_w_q);
            for i in 0..loaded_b.attn_w_q.data.len() {
                let expected = w_q.data[i] * gamma;
                assert!(
                    (loaded_b.attn_w_q.data[i] - expected).abs() < 1e-5,
                    "round-trip drift at attn_w_q[{}]: {} vs {}",
                    i,
                    loaded_b.attn_w_q.data[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn ternary_packed_round_trip() {
        let model = tiny_model();
        let mut buf = Vec::new();
        export_ternary_packed(&model, &mut buf).unwrap();

        let (loaded, fmt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::TernaryPacked);

        // Embeddings stay f32, exact match.
        assert_eq!(loaded.token_embed.data, model.token_embed.data);

        // Block weights: every loaded value must equal γ · w_q.
        for (loaded_b, master_b) in loaded.blocks.iter().zip(&model.blocks) {
            let (w_q, gamma) = absmean_ternary(&master_b.attn_w_q);
            for i in 0..loaded_b.attn_w_q.data.len() {
                let expected = w_q.data[i] * gamma;
                assert!(
                    (loaded_b.attn_w_q.data[i] - expected).abs() < 1e-5,
                    "packed round-trip drift at attn_w_q[{}]: {} vs {}",
                    i,
                    loaded_b.attn_w_q.data[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn ternary_packed_smaller_than_unpacked_ternary() {
        let model = tiny_model();
        let mut t_buf = Vec::new();
        let mut p_buf = Vec::new();
        export_ternary(&model, &mut t_buf).unwrap();
        export_ternary_packed(&model, &mut p_buf).unwrap();
        assert!(
            p_buf.len() < t_buf.len(),
            "packed ({} B) must be smaller than unpacked ternary ({} B)",
            p_buf.len(),
            t_buf.len()
        );
    }

    #[test]
    fn pack_unpack_roundtrips_each_5tuple() {
        // Pack every possible 5-tuple of ternary values, unpack, verify identity.
        // 3^5 = 243 distinct possibilities, exhaustive.
        for a in -1..=1i8 {
            for b in -1..=1i8 {
                for c in -1..=1i8 {
                    for d in -1..=1i8 {
                        for e in -1..=1i8 {
                            let chunk = [a, b, c, d, e];
                            let byte = pack_ternary_chunk(&chunk);
                            let back = unpack_ternary_byte(byte);
                            assert_eq!(back, chunk, "roundtrip failed for {:?}", chunk);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn import_rejects_bad_magic() {
        let mut bad = Vec::new();
        bad.extend_from_slice(b"NOPE");
        bad.extend_from_slice(&[0u8; HEADER_SIZE - 4]);
        let err = import(&mut Cursor::new(&bad)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
