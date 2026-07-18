//! Binary export and import of trained model weights.
//!
//! Three formats:
//!   - `Format::Float32`: every weight as raw f32 (4 bytes per value).
//!     Lossless: BitLinear masters survive byte-identical
//!     through save+load, so resuming training picks up
//!     exactly where it left off (no `step 0 val_ppl`
//!     spike from a re-quantised master). This is the
//!     recommended format for `--resume` checkpoints.
//!   - `Format::Ternary`: embeddings stay f32; every BitLinear weight (block
//!     weights and the LM head) becomes `(γ f32, ternary i8/value)`.
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
//   BNT4: RoPE replaces learned absolute position embedding. Top-level
//         payload no longer carries `pos_embed`; only `token_embed`,
//         then per block, then `lm_head`. `max_seq_len` stays in the header
//         (still useful for the assertion in `Model::forward`).
//   BNT5: tied input-output embeddings. The trailing `lm_head` payload
//         entry is gone; the LM-head matmul reads `token_embed` (transposed
//         at op-build time) directly. Visitor / param_shape order drops the
//         trailing lm_head slot; AdamW state in the OPTM payload is
//         correspondingly one slot shorter.
// Each bump invalidates older checkpoints fail-fast rather than letting the
// importer slide off into garbage payload bytes. The header layout is
// unchanged (magic + format + 7 u32 fields = 33 bytes); the difference at
// each version is the payload shape.
const MAGIC: &[u8; 4] = b"BNT5";
const HEADER_SIZE: usize = 4 + 1 + 7 * 4; // magic + format byte + 7 u32 fields = 33

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Every weight as raw f32 (4 bytes per value). The "with masters" format:
    /// BitLinear master weights are saved verbatim instead of being projected
    /// onto `γ · W_q`, so a save+load cycle is a true identity on the masters.
    /// Use this for resume checkpoints; pair with `Ternary` / `TernaryPacked`
    /// for the compact deployment artefact.
    Float32,
    /// BitLinear weights as `(γ f32, one i8 per ternary)`. Embeddings stay f32.
    /// Round-trippable via `xxd` for debugging.
    Ternary,
    /// BitLinear weights as `(γ f32, base-3 packed bytes)`. 5 ternary values
    /// fit in a single byte (3^5 = 243 < 256). Closest to the "1.58 bits per
    /// weight" theoretical optimum with a fixed-byte container.
    TernaryPacked,
    /// Issue #23: every weight as a bf16 half (2 bytes per value) - the
    /// top 16 bits of the f32 pattern. Masters are stored at exactly the
    /// precision the optimiser maintains them at, so a save+load cycle
    /// is still a true identity. The OPTM payload (AdamW moments) stays
    /// f32. The resume-checkpoint format since #23.
    MastersBf16,
}

impl Format {
    fn byte(self) -> u8 {
        match self {
            Format::Float32 => 0,
            Format::Ternary => 1,
            Format::TernaryPacked => 2,
            Format::MastersBf16 => 3,
        }
    }
    fn from_byte(b: u8) -> io::Result<Format> {
        match b {
            0 => Ok(Format::Float32),
            1 => Ok(Format::Ternary),
            2 => Ok(Format::TernaryPacked),
            3 => Ok(Format::MastersBf16),
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
            "not a BNT4 file (bad magic - BNT1/BNT2/BNT3 checkpoints \
             are no longer compatible; v0.12 retired learned pos_embed \
             in favour of RoPE, so the payload layout changed; retrain)",
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

/// Issue #23: bf16 halves, 2 bytes per value. The writer narrows with
/// RNE - a no-op for masters the optimiser already keeps on the bf16
/// grid, and the correct projection for anything else (e.g. a model
/// trained pre-#23 being re-saved).
fn write_bf16_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<usize> {
    let mut bytes = 0;
    for &v in &t.data {
        w.write_all(&crate::tensor::f32_to_bf16_bits(v).to_le_bytes())?;
        bytes += 2;
    }
    Ok(bytes)
}

fn read_bf16_tensor<R: Read>(r: &mut R, shape: Vec<usize>) -> io::Result<Tensor> {
    let len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut buf = [0u8; 2];
    for _ in 0..len {
        r.read_exact(&mut buf)?;
        data.push(crate::tensor::bf16_bits_to_f32(u16::from_le_bytes(buf)));
    }
    Ok(Tensor { data, shape })
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
/// Body of the OPTM section (marker already consumed by the trailer
/// dispatch in `import`).
fn read_optim_state_body<R: Read>(
    r: &mut R,
    param_shapes: &[Vec<usize>],
) -> io::Result<OptimState> {
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
    Ok(OptimState { step_count, m, v })
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
    // No trailing lm_head: tied to token_embed since BNT5 / v0.17.
    if let Some(state) = optim {
        total += write_optim_state(w, state)?;
    }
    Ok(total)
}

/// Issue #23: the resume-checkpoint export - bf16 masters (2 bytes per
/// weight, exactly the precision the optimiser maintains), f32 OPTM
/// payload. Identity round-trip for post-#23 models.
pub fn export_bf16<W: Write>(
    model: &Model,
    w: &mut W,
    optim: Option<&OptimState>,
) -> io::Result<usize> {
    let mut total = write_header(w, &model.config, Format::MastersBf16)?;
    total += write_bf16_tensor(w, &model.token_embed)?;
    for b in &model.blocks {
        for h in &b.heads {
            total += write_bf16_tensor(w, &h.w_q)?;
            total += write_bf16_tensor(w, &h.w_k)?;
            total += write_bf16_tensor(w, &h.w_v)?;
            total += write_bf16_tensor(w, &h.w_o)?;
        }
        total += write_bf16_tensor(w, &b.ffn_gate_w)?;
        total += write_bf16_tensor(w, &b.ffn_up_w)?;
        total += write_bf16_tensor(w, &b.ffn_down_w)?;
    }
    // No trailing lm_head: tied to token_embed since BNT5 / v0.17.
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
    // No trailing lm_head: tied to token_embed since BNT5 / v0.17. The LM-head
    // matmul reads the (f32) token_embed at inference and ternary-quantises it
    // on the fly, identical to the training-time forward.
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
    // No trailing lm_head: tied to token_embed since BNT5 / v0.17.
    if let Some(state) = optim {
        total += write_optim_state(w, state)?;
    }
    Ok(total)
}

// ---- BPE tokeniser section (issue #24) ----
//
// Optional trailing section AFTER the OPTM payload: marker "BPEM" then
// the `Bpe::save` bytes. A BPE-trained model is unusable without its
// merges, so the tokeniser travels inside the checkpoint - `sample`
// needs no corpus and no side-channel file. Char-vocab checkpoints
// simply omit the section (fully backward compatible).

const BPE_MARKER: &[u8; 4] = b"BPEM";

pub fn append_bpe_section<W: Write>(w: &mut W, bpe: &crate::bpe::Bpe) -> io::Result<()> {
    w.write_all(BPE_MARKER)?;
    bpe.save(w)
}

// ---- public import ----

/// Read a model file (any format) and reconstruct the `Model`, the `Format`
/// it was stored in, and an optional `OptimState` if the file carries one.
/// Older BNT3 files written before optim-state persistence terminate after
/// `lm_head`; this function returns `Ok((model, fmt, None))` for those.
#[allow(clippy::type_complexity)]
pub fn import<R: Read>(
    r: &mut R,
) -> io::Result<(Model, Format, Option<OptimState>, Option<crate::bpe::Bpe>)> {
    let (fmt, cfg) = read_header(r)?;
    let h = cfg.hidden_dim;
    let d = cfg.head_dim;
    let f = cfg.ffn_dim;

    // Ternary formats keep the embedding at f32 (it is not a BitLinear
    // weight); the bf16 format (#23) narrows it like every other master.
    let token_embed = match fmt {
        Format::MastersBf16 => read_bf16_tensor(r, vec![cfg.vocab_size, h])?,
        _ => read_f32_tensor(r, vec![cfg.vocab_size, h])?,
    };

    let mut blocks = Vec::with_capacity(cfg.n_blocks);
    for _ in 0..cfg.n_blocks {
        let read_w = |r: &mut R, shape: Vec<usize>| match fmt {
            Format::Float32 => read_f32_tensor(r, shape),
            Format::Ternary => read_ternary_tensor(r, shape),
            Format::TernaryPacked => read_ternary_packed_tensor(r, shape),
            Format::MastersBf16 => read_bf16_tensor(r, shape),
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

    // No lm_head payload to read: tied to token_embed since BNT5 / v0.17.
    let model = Model {
        token_embed,
        blocks,
        config: cfg,
    };

    // Try to read optim state. EOF here means the file pre-dates optim
    // persistence; that's fine, returns Ok(None).
    // Trailing sections, each announced by a 4-byte marker: OPTM
    // (AdamW state) and/or BPEM (issue #24 tokeniser), any order, both
    // optional. EOF ends the trailer; an unknown marker is corruption.
    let param_shapes = model.param_shapes();
    let mut optim = None;
    let mut bpe = None;
    let mut marker = [0u8; 4];
    loop {
        match r.read_exact(&mut marker) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            other => other?,
        }
        match &marker {
            m if m == OPTM_MARKER => optim = Some(read_optim_state_body(r, &param_shapes)?),
            m if m == BPE_MARKER => bpe = Some(crate::bpe::Bpe::load(r)?),
            other => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "unknown trailer marker {:?}",
                        String::from_utf8_lossy(other)
                    ),
                ));
            }
        }
    }

    Ok((model, fmt, optim, bpe))
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
        // Total entries (BNT5: pos_embed dropped via RoPE, lm_head dropped via
        // tied embeddings; the LM-head matmul reads token_embed at op-build):
        //   token_embed   : 8 * 4 = 32
        //   per block     : 2 heads * 4 * (4*2) + 3 FFN tensors (gate/up/down)
        //                 = 64 + 96 = 160 entries per block
        //   2 blocks      : 320
        //   total         : 32 + 320 = 352 floats
        // Header (33) + 352 floats × 4 = 33 + 1408 = 1441 bytes.
        let expected = HEADER_SIZE + 352 * 4;
        let mut buf = Vec::new();
        let bytes = export_f32(&tiny_model(), &mut buf, None).unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn ternary_export_size_matches_expected_layout() {
        // BNT5 layout: token_embed only (no pos_embed) + ternary blocks
        // (lm_head dropped via tied embeddings). Header (33) +
        //   token_embed f32 (32 floats * 4 = 128) +
        //   per block: 11 ternary tensors (4 per head * 2 heads + gate + up + down)
        //              each carries a 4-byte gamma + 1 byte per i8 entry
        //              gammas: 11 * 4 = 44, i8 entries: 160
        //              per-block bytes: 204
        //   2 blocks: 408
        // Total: 33 + 128 + 408 = 569.
        let expected = HEADER_SIZE + 128 + 408;
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

        let (loaded, fmt, _opt, _) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Float32);
        assert_eq!(loaded.config.vocab_size, model.config.vocab_size);
        assert_eq!(loaded.token_embed.data, model.token_embed.data);
        // No separate lm_head field since BNT5 / v0.17; the LM-head weight
        // is `token_embed` itself, transposed at op-build time, and is
        // checked above.
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

        let (loaded, fmt, _opt, _) = import(&mut Cursor::new(&buf)).unwrap();
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

        let (loaded, fmt, _opt, _) = import(&mut Cursor::new(&buf)).unwrap();
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
        let (_loaded, _fmt, optim, _) = import(&mut Cursor::new(&buf)).unwrap();
        assert!(
            optim.is_none(),
            "expected no optim state in payload-less export"
        );
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
                let data: Vec<f32> = (0..n).map(|j| (i * 100 + j) as f32 * 0.0001).collect();
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
        let (_loaded_model, _fmt, loaded_optim, _) = import(&mut Cursor::new(&buf)).unwrap();
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
    fn f32_round_trip_preserves_every_master_and_optim_byte_identical() {
        // The Finding-A guarantee: a Float32 save+load is a true identity on
        // every master tensor AND on the AdamW optim state. If this test ever
        // breaks, resuming from a `.f32.bin` checkpoint will silently
        // re-introduce the "step 0 val_ppl spike" failure mode. The previous
        // `f32_round_trip_preserves_every_weight_exactly` only spot-checked
        // a single head and one FFN tensor; this one walks every parameter.
        use crate::optim::OptimState;

        let model = tiny_model();
        let shapes = model.param_shapes();
        let m: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let n: usize = s.iter().product();
                let data: Vec<f32> = (0..n).map(|j| (i * 7 + j) as f32 * 1.234e-3).collect();
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
                let data: Vec<f32> = (0..n).map(|j| (i * 11 + j) as f32 * 5.678e-5).collect();
                Tensor {
                    data,
                    shape: s.clone(),
                }
            })
            .collect();
        let saved = OptimState {
            step_count: 9_999,
            m: m.clone(),
            v: v.clone(),
        };

        let mut buf = Vec::new();
        export_f32(&model, &mut buf, Some(&saved)).unwrap();
        let (loaded_model, fmt, loaded_optim, _) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(fmt, Format::Float32);
        let loaded_state = loaded_optim.expect("optim state must round-trip");

        // Token embedding is full f32 in this format. (No pos_embed since
        // v0.12: RoPE replaced it. No lm_head since v0.17: tied to
        // token_embed.)
        assert_eq!(loaded_model.token_embed.data, model.token_embed.data);

        // Every block's master weights survive byte-identical. Walking every
        // head's Q/K/V/O and every FFN tensor catches any silent drop or
        // dequantisation in the f32 path.
        assert_eq!(loaded_model.blocks.len(), model.blocks.len());
        for (lb, mb) in loaded_model.blocks.iter().zip(&model.blocks) {
            assert_eq!(lb.heads.len(), mb.heads.len());
            for (lh, mh) in lb.heads.iter().zip(&mb.heads) {
                assert_eq!(lh.w_q.data, mh.w_q.data);
                assert_eq!(lh.w_k.data, mh.w_k.data);
                assert_eq!(lh.w_v.data, mh.w_v.data);
                assert_eq!(lh.w_o.data, mh.w_o.data);
            }
            assert_eq!(lb.ffn_gate_w.data, mb.ffn_gate_w.data);
            assert_eq!(lb.ffn_up_w.data, mb.ffn_up_w.data);
            assert_eq!(lb.ffn_down_w.data, mb.ffn_down_w.data);
        }

        // Optim state is identical too: step_count plus every m and v cell.
        assert_eq!(loaded_state.step_count, 9_999);
        assert_eq!(loaded_state.m.len(), m.len());
        assert_eq!(loaded_state.v.len(), v.len());
        for (lt, st) in loaded_state.m.iter().zip(&m) {
            assert_eq!(lt.shape, st.shape);
            assert_eq!(lt.data, st.data);
        }
        for (lt, st) in loaded_state.v.iter().zip(&v) {
            assert_eq!(lt.shape, st.shape);
            assert_eq!(lt.data, st.data);
        }
    }

    /// Issue #23: bf16 export -> import is an identity for models whose
    /// masters sit on the bf16 grid (i.e. anything the post-#23
    /// optimiser produced), and the OPTM payload survives at f32.
    #[test]
    fn bf16_export_import_round_trips_grid_masters() {
        let mut model = tiny_model();
        // Force every master onto the bf16 grid, as the optimiser does.
        model.for_each_param_mut(|t| {
            for v in t.data.iter_mut() {
                *v = crate::tensor::narrow_to_bf16(*v);
            }
        });
        let optim = OptimState {
            step_count: 42,
            m: model_shaped_tensors(&model, 0.31),
            v: model_shaped_tensors(&model, 0.77),
        };
        let mut buf = Vec::new();
        export_bf16(&model, &mut buf, Some(&optim)).unwrap();

        let (back, fmt, optim_back, _) = import(&mut buf.as_slice()).unwrap();
        assert_eq!(fmt, Format::MastersBf16);
        let mut expect: Vec<Tensor> = Vec::new();
        model.for_each_param_mut(|t| expect.push(t.clone()));
        let mut i = 0;
        let mut back_m = back;
        back_m.for_each_param_mut(|t| {
            assert_eq!(t.data, expect[i].data, "master {i} did not round-trip");
            i += 1;
        });
        let ob = optim_back.expect("OPTM payload lost");
        assert_eq!(ob.step_count, 42);
        for (a, b) in ob.m.iter().zip(&optim.m) {
            assert_eq!(a.data, b.data, "moment m did not round-trip at f32");
        }
        // Size: masters at 2 bytes each. Compare against the f32 export.
        let mut f32_buf = Vec::new();
        export_f32(&model, &mut f32_buf, None).unwrap();
        let mut bf16_buf = Vec::new();
        export_bf16(&model, &mut bf16_buf, None).unwrap();
        let header = 33;
        assert_eq!(bf16_buf.len() - header, (f32_buf.len() - header) / 2);
    }

    /// Helper: tensors shaped like the model's params, deterministic fill.
    fn model_shaped_tensors(model: &Model, salt: f32) -> Vec<Tensor> {
        model
            .param_shapes()
            .into_iter()
            .map(|shape| {
                let len: usize = shape.iter().product();
                Tensor {
                    data: (0..len).map(|i| (i as f32 * salt).sin()).collect(),
                    shape,
                }
            })
            .collect()
    }

    /// Issue #24: the BPEM trailing section round-trips the tokeniser
    /// through a checkpoint, and files without one read back None.
    #[test]
    fn bpe_section_round_trips_through_checkpoint() {
        let model = tiny_model();
        let bpe = crate::bpe::Bpe::train(b"the theme thereof there ", 270);
        let mut buf = Vec::new();
        export_bf16(&model, &mut buf, None).unwrap();
        append_bpe_section(&mut buf, &bpe).unwrap();
        let (_m, _fmt, _opt, loaded) = import(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.expect("BPEM section lost"), bpe);

        let mut plain = Vec::new();
        export_bf16(&model, &mut plain, None).unwrap();
        let (_m, _fmt, _opt, none) = import(&mut Cursor::new(&plain)).unwrap();
        assert!(none.is_none(), "phantom tokeniser from a plain checkpoint");
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
