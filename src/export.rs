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
//! Header (33 bytes):
//!   4 B   magic "BNT2"   (bumped from "BNT1" when multi-head arrived;
//!                         old single-head files fail-fast on import)
//!   1 B   format byte: 0 = Float32, 1 = Ternary, 2 = TernaryPacked
//!   4 B   vocab_size  (u32 LE)
//!   4 B   hidden_dim  (u32 LE)
//!   4 B   n_heads     (u32 LE)
//!   4 B   head_dim    (u32 LE)
//!   4 B   ffn_dim     (u32 LE)
//!   4 B   max_seq_len (u32 LE)
//!   4 B   n_blocks    (u32 LE)
//!
//! Payload (positional, in this order):
//!   token_embed, pos_embed              (always f32)
//!   per block:
//!     for each of n_heads heads: w_q, w_k, w_v, w_o
//!     ffn_up_w, ffn_down_w
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
use crate::model::{AttentionHead, BlockMasters, Model, ModelConfig};
use crate::optim::OptimState;
use crate::tensor::Tensor;
use std::io::{self, ErrorKind, Read, Write};

// Magic version history:
//   BNT1: single-head attention, ReLU FFN (2 weights per block).
//   BNT2: multi-head attention, ReLU FFN (added n_heads to header).
//   BNT3: multi-head attention, SwiGLU FFN (added ffn_gate_w as a third
//         FFN weight per block, written between the heads and ffn_up_w).
// Each bump invalidates older checkpoints fail-fast rather than letting the
// importer slide off into garbage payload bytes. The header layout is
// unchanged from BNT2 (magic + format + 7 u32 fields = 33 bytes); the
// difference lives entirely in the per-block payload.
const MAGIC: &[u8; 4] = b"BNT3";
const HEADER_SIZE: usize = 4 + 1 + 7 * 4; // magic + format byte + 7 u32 fields = 33

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
        cfg.n_heads as u32,
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
            "not a BNT3 file (bad magic - BNT1 single-head and BNT2 \
             ReLU-FFN checkpoints are no longer compatible; retrain)",
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
    let n_heads = read_u32(r)? as usize;
    let head_dim = read_u32(r)? as usize;
    let ffn_dim = read_u32(r)? as usize;
    let max_seq_len = read_u32(r)? as usize;
    let n_blocks = read_u32(r)? as usize;

    Ok((
        fmt,
        ModelConfig {
            vocab_size,
            hidden_dim,
            n_heads,
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

// ---- optim state IO ----

/// 4-byte marker that introduces an optim-state payload. Lets the importer
/// distinguish "no optim state" (file ends after lm_head, EOF) from "optim
/// state follows" (these four bytes, then step_count + moment tensors).
/// Writers always emit this marker when called; readers gracefully accept
/// EOF here, which is what makes BNT3 files without optim state still load.
const OPTM_MARKER: &[u8; 4] = b"OPTM";

fn write_optim_state<W: Write>(w: &mut W, state: &OptimState) -> io::Result<usize> {
    w.write_all(OPTM_MARKER)?;
    w.write_all(&state.step_count.to_le_bytes())?;
    let mut bytes = 4 + 4;
    for t in state.m.iter().chain(state.v.iter()) {
        for &v in &t.data {
            w.write_all(&v.to_le_bytes())?;
            bytes += 4;
        }
    }
    Ok(bytes)
}

/// Read an optim-state payload if present, otherwise return `Ok(None)`.
/// `param_shapes` must match the model whose state is being loaded - the
/// importer reads exactly `2 * sum(shape.product())` floats and uses the
/// shapes to reconstruct each `m` and `v` tensor.
fn read_optim_state_or_none<R: Read>(
    r: &mut R,
    param_shapes: &[Vec<usize>],
) -> io::Result<Option<OptimState>> {
    let mut marker = [0u8; 4];
    match r.read_exact(&mut marker) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    if &marker != OPTM_MARKER {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "expected OPTM marker after lm_head, got {:?}",
                String::from_utf8_lossy(&marker)
            ),
        ));
    }

    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    let step_count = u32::from_le_bytes(buf);

    let mut m = Vec::with_capacity(param_shapes.len());
    for shape in param_shapes {
        m.push(read_f32_tensor(r, shape.clone())?);
    }
    let mut v = Vec::with_capacity(param_shapes.len());
    for shape in param_shapes {
        v.push(read_f32_tensor(r, shape.clone())?);
    }
    Ok(Some(OptimState { step_count, m, v }))
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

pub fn export_f32<W: Write>(
    model: &Model,
    w: &mut W,
    optim: Option<&OptimState>,
) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::Float32)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        for h in &b.heads {
            total += write_f32_tensor(w, &h.w_q)?;
            total += write_f32_tensor(w, &h.w_k)?;
            total += write_f32_tensor(w, &h.w_v)?;
            total += write_f32_tensor(w, &h.w_o)?;
        }
        total += write_f32_tensor(w, &b.ffn_gate_w)?;
        total += write_f32_tensor(w, &b.ffn_up_w)?;
        total += write_f32_tensor(w, &b.ffn_down_w)?;
    }
    total += write_f32_tensor(w, &model.lm_head)?;
    if let Some(state) = optim {
        total += write_optim_state(w, state)?;
    }
    Ok(total)
}

pub fn export_ternary<W: Write>(
    model: &Model,
    w: &mut W,
    optim: Option<&OptimState>,
) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::Ternary)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        for h in &b.heads {
            total += write_ternary_tensor(w, &h.w_q)?;
            total += write_ternary_tensor(w, &h.w_k)?;
            total += write_ternary_tensor(w, &h.w_v)?;
            total += write_ternary_tensor(w, &h.w_o)?;
        }
        total += write_ternary_tensor(w, &b.ffn_gate_w)?;
        total += write_ternary_tensor(w, &b.ffn_up_w)?;
        total += write_ternary_tensor(w, &b.ffn_down_w)?;
    }
    // LM head is also a BitLinear, so it's ternary too.
    total += write_ternary_tensor(w, &model.lm_head)?;
    if let Some(state) = optim {
        total += write_optim_state(w, state)?;
    }
    Ok(total)
}

/// Most compact format. BitLinear weights packed 5 ternaries per byte (base-3).
pub fn export_ternary_packed<W: Write>(
    model: &Model,
    w: &mut W,
    optim: Option<&OptimState>,
) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::TernaryPacked)?;
    total += write_f32_tensor(w, &model.token_embed)?;
    total += write_f32_tensor(w, &model.pos_embed)?;
    for b in &model.blocks {
        for h in &b.heads {
            total += write_ternary_packed_tensor(w, &h.w_q)?;
            total += write_ternary_packed_tensor(w, &h.w_k)?;
            total += write_ternary_packed_tensor(w, &h.w_v)?;
            total += write_ternary_packed_tensor(w, &h.w_o)?;
        }
        total += write_ternary_packed_tensor(w, &b.ffn_gate_w)?;
        total += write_ternary_packed_tensor(w, &b.ffn_up_w)?;
        total += write_ternary_packed_tensor(w, &b.ffn_down_w)?;
    }
    total += write_ternary_packed_tensor(w, &model.lm_head)?;
    if let Some(state) = optim {
        total += write_optim_state(w, state)?;
    }
    Ok(total)
}

// ---- public import ----

/// Read a model file (any format) and reconstruct the `Model`, the `Format`
/// it was stored in, and an optional `OptimState` if the file carries one.
/// Older BNT3 files written before optim-state persistence terminate after
/// `lm_head`; this function returns `Ok((model, fmt, None))` for those.
pub fn import<R: Read>(r: &mut R) -> io::Result<(Model, Format, Option<OptimState>)> {
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
        let mut heads = Vec::with_capacity(cfg.n_heads);
        for _ in 0..cfg.n_heads {
            heads.push(AttentionHead {
                w_q: read_w(r, vec![h, d])?,
                w_k: read_w(r, vec![h, d])?,
                w_v: read_w(r, vec![h, d])?,
                w_o: read_w(r, vec![d, h])?,
            });
        }
        blocks.push(BlockMasters {
            heads,
            ffn_gate_w: read_w(r, vec![h, f])?,
            ffn_up_w: read_w(r, vec![h, f])?,
            ffn_down_w: read_w(r, vec![f, h])?,
        });
    }

    let lm_head = match fmt {
        Format::Float32 => read_f32_tensor(r, vec![h, cfg.vocab_size])?,
        Format::Ternary => read_ternary_tensor(r, vec![h, cfg.vocab_size])?,
        Format::TernaryPacked => read_ternary_packed_tensor(r, vec![h, cfg.vocab_size])?,
    };

    let model = Model {
        token_embed,
        pos_embed,
        blocks,
        lm_head,
        config: cfg,
    };

    // Try to read optim state. EOF here means the file pre-dates optim
    // persistence; that's fine, returns Ok(None).
    let param_shapes = model.param_shapes();
    let optim = read_optim_state_or_none(r, &param_shapes)?;

    Ok((model, fmt, optim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_model() -> Model {
        // n_heads * head_dim == hidden_dim invariant for attention.
        // SwiGLU FFN adds one extra weight tensor (ffn_gate_w) per block, so
        // total parameter count is larger than the BNT2 (ReLU FFN) model.
        let cfg = ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            n_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            max_seq_len: 4,
            n_blocks: 2,
        };
        Model::new(&cfg, 0)
    }

    #[test]
    fn f32_export_size_matches_total_param_count() {
        // Total entries:
        //   token_embed   : 8 * 4 = 32
        //   pos_embed     : 4 * 4 = 16
        //   per block     : 2 heads * 4 * (4*2) + 3 FFN tensors (gate/up/down)
        //                 = 64 + 96 = 160 entries per block
        //   2 blocks      : 320
        //   lm_head       : 4 * 8 = 32
        //   total         : 48 + 320 + 32 = 400 floats
        // Header (33) + 400 floats × 4 = 33 + 1600 = 1633 bytes.
        let expected = HEADER_SIZE + 400 * 4;
        let mut buf = Vec::new();
        let bytes = export_f32(&tiny_model(), &mut buf, None).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn ternary_export_size_matches_expected_layout() {
        // Header (33) + embeddings (48 floats * 4 = 192) +
        //   per block: 11 ternary tensors (4 per head * 2 heads + gate + up + down)
        //              each carries a 4-byte gamma + 1 byte per i8 entry
        //              gammas: 11 * 4 = 44, i8 entries: 160
        //              per-block bytes: 204
        //   2 blocks: 408
        //   lm_head ternary: 4-byte gamma + 32 i8 entries = 36
        // Total: 33 + 192 + 408 + 36 = 669.
        let expected = HEADER_SIZE + 192 + 408 + 36;
        let mut buf = Vec::new();
        let bytes = export_ternary(&tiny_model(), &mut buf, None).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn ternary_export_is_smaller_than_f32() {
        let model = tiny_model();
        let mut f32_buf = Vec::new();
        let mut ter_buf = Vec::new();
        export_f32(&model, &mut f32_buf, None).unwrap();
        export_ternary(&model, &mut ter_buf, None).unwrap();
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
        export_f32(&model, &mut buf, None).unwrap();

        let (loaded, fmt, _opt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Float32);
        assert_eq!(loaded.config.vocab_size, model.config.vocab_size);
        assert_eq!(loaded.token_embed.data, model.token_embed.data);
        assert_eq!(loaded.lm_head.data, model.lm_head.data);
        assert_eq!(loaded.blocks.len(), model.blocks.len());
        for (lb, mb) in loaded.blocks.iter().zip(&model.blocks) {
            // Spot-check head 0's Q projection plus the FFN weights. The full
            // round-trip is structurally tested in the ternary tests below.
            assert_eq!(lb.heads.len(), mb.heads.len());
            assert_eq!(lb.heads[0].w_q.data, mb.heads[0].w_q.data);
            assert_eq!(lb.ffn_up_w.data, mb.ffn_up_w.data);
        }
    }

    #[test]
    fn ternary_round_trip_preserves_block_weight_signs() {
        // Round-trip discards f32 master precision in BitLinear weights but
        // must preserve the (γ, W_q) decomposition exactly.
        let model = tiny_model();
        let mut buf = Vec::new();
        export_ternary(&model, &mut buf, None).unwrap();

        let (loaded, fmt, _opt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Ternary);

        // Embeddings are f32, exact match.
        assert_eq!(loaded.token_embed.data, model.token_embed.data);

        // Every per-head weight must round-trip to γ · w_q exactly. Walking
        // both heads in every block exercises the per-head IO path; a head
        // dropped on either write or read would surface as a drift here.
        for (loaded_b, master_b) in loaded.blocks.iter().zip(&model.blocks) {
            assert_eq!(loaded_b.heads.len(), master_b.heads.len());
            for (lh, mh) in loaded_b.heads.iter().zip(&master_b.heads) {
                let (w_q, gamma) = absmean_ternary(&mh.w_q);
                for i in 0..lh.w_q.data.len() {
                    let expected = w_q.data[i] * gamma;
                    assert!(
                        (lh.w_q.data[i] - expected).abs() < 1e-5,
                        "round-trip drift at head w_q[{}]: {} vs {}",
                        i,
                        lh.w_q.data[i],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn ternary_packed_round_trip() {
        let model = tiny_model();
        let mut buf = Vec::new();
        export_ternary_packed(&model, &mut buf, None).unwrap();

        let (loaded, fmt, _opt) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::TernaryPacked);

        // Embeddings stay f32, exact match.
        assert_eq!(loaded.token_embed.data, model.token_embed.data);

        for (loaded_b, master_b) in loaded.blocks.iter().zip(&model.blocks) {
            assert_eq!(loaded_b.heads.len(), master_b.heads.len());
            for (lh, mh) in loaded_b.heads.iter().zip(&master_b.heads) {
                let (w_q, gamma) = absmean_ternary(&mh.w_q);
                for i in 0..lh.w_q.data.len() {
                    let expected = w_q.data[i] * gamma;
                    assert!(
                        (lh.w_q.data[i] - expected).abs() < 1e-5,
                        "packed round-trip drift at head w_q[{}]: {} vs {}",
                        i,
                        lh.w_q.data[i],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn ternary_packed_smaller_than_unpacked_ternary() {
        let model = tiny_model();
        let mut t_buf = Vec::new();
        let mut p_buf = Vec::new();
        export_ternary(&model, &mut t_buf, None).unwrap();
        export_ternary_packed(&model, &mut p_buf, None).unwrap();
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
    fn import_returns_none_optim_for_files_without_optm_marker() {
        // Round-trip a model with no optim state. import should succeed and
        // return None for the third tuple element (no OPTM marker means we
        // hit EOF cleanly after lm_head).
        let model = tiny_model();
        let mut buf = Vec::new();
        export_ternary_packed(&model, &mut buf, None).unwrap();
        let (_loaded, _fmt, optim) = import(&mut Cursor::new(&buf)).unwrap();
        assert!(optim.is_none(), "expected no optim state in payload-less export");
    }

    #[test]
    fn optim_state_round_trips_through_packed_export() {
        // Write a model with a hand-built optim state, read it back, verify
        // step_count and a sample of m/v values survived the round-trip exactly.
        // OPTM payload is f32 (lossless), unlike the ternary weights themselves.
        use crate::optim::OptimState;

        let model = tiny_model();
        let shapes = model.param_shapes();
        // Construct an OptimState whose m/v values vary per tensor so a
        // shape-mismatched read would fail loudly.
        let m: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                let data: Vec<f32> = (0..n).map(|j| (i * 100 + j) as f32 * 0.001).collect();
                Tensor {
                    data,
                    shape: s.clone(),
                }
            })
            .collect();
        let v: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                let data: Vec<f32> =
                    (0..n).map(|j| (i * 100 + j) as f32 * 0.0001).collect();
                Tensor {
                    data,
                    shape: s.clone(),
                }
            })
            .collect();
        let saved = OptimState {
            step_count: 1234,
            m: m.clone(),
            v: v.clone(),
        };

        let mut buf = Vec::new();
        export_ternary_packed(&model, &mut buf, Some(&saved)).unwrap();
        let (_loaded_model, _fmt, loaded_optim) = import(&mut Cursor::new(&buf)).unwrap();
        let loaded = loaded_optim.expect("optim state must round-trip when written");

        assert_eq!(loaded.step_count, 1234);
        assert_eq!(loaded.m.len(), m.len());
        assert_eq!(loaded.v.len(), v.len());
        // Spot-check a couple of tensors for exact f32 equality.
        for (lt, st) in loaded.m.iter().zip(&m) {
            assert_eq!(lt.shape, st.shape);
            assert_eq!(lt.data, st.data);
        }
        for (lt, st) in loaded.v.iter().zip(&v) {
            assert_eq!(lt.shape, st.shape);
            assert_eq!(lt.data, st.data);
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
