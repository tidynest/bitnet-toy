//! Full BitNet language model - token embedding + N transformer blocks +
//! tied LM head.
//!
//! Structure:
//!     Model { token_embed, blocks: Vec<BlockMasters>, config }
//!
//! Position information arrives via RoPE inside attention (see `autograd::rope`
//! and `attention::head_output`) - there is no learned absolute position
//! embedding tensor. RoPE is parameter-free, so the model carries one fewer
//! tensor than a typical transformer.
//!
//! **Tied embeddings (v0.17 / BNT5):** the LM head re-uses `token_embed` as
//! its weight matrix, transposed at op-build time. The same master tensor is
//! gradient-updated from both the embed-lookup path (gather backward) and
//! the LM-head path (matmul backward through STE quantisation). This drops
//! `vocab_size * hidden_dim` parameters (e.g. 65 * 256 = ~17k for the
//! shakespeare_large config) and acts as a strong regulariser - input and
//! output now share a single semantic vocabulary representation. Standard
//! practice in GPT-2 / LLaMA / etc; the BitNet paper itself doesn't tie,
//! but at small scales the regularisation win usually beats the rare cases
//! where decoupled paths help.
//!
//! Training cycle (one step):
//!     tape   = Tape::new()
//!     leaves = model.register_leaves(&tape)        # master Tensors -> tape leaves
//!     logits = model.forward(&leaves, &ids)        # build the graph
//!     loss   = logits.cross_entropy(&targets)
//!     tape.backward(loss.id)
//!     model.apply_grads(&leaves, lr)               # SGD update on masters
//!     # tape + leaves drop here - graph released, master tensors retained
//!
//! No bias on any layer, no learnable RMSNorm gain - same constraints as the
//! BitNet paper, kept here for paper-faithfulness and to minimise param count.

use crate::autograd::{Tape, Var};
use crate::block::{BlockWeights, transformer_block};
use crate::tensor::Tensor;

/// Hyperparameters. Cheap to clone - fields are all `usize`.
///
/// Multi-head attention: each block runs `n_heads` independent attention paths
/// in parallel; their outputs are summed (sum-of-projections form, mathematically
/// equivalent to concat-then-project). Conventional sizing: `n_heads * head_dim`
/// equals `hidden_dim`, which keeps total attention parameter count identical
/// to single-head while splitting the representation into orthogonal subspaces.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub n_blocks: usize,
}

/// Master weights for one attention head. Each head holds its own Q/K/V/O
/// projections; an `n_heads`-long `Vec<AttentionHead>` lives inside each block.
/// Sum-of-projections form: each head's W_o brings its `head_dim` context
/// straight back to `hidden_dim`, so the block's attention output is just the
/// element-wise sum of all head outputs (no concat op needed).
#[derive(Debug, Clone)]
pub struct AttentionHead {
    pub w_q: Tensor, // [hidden, head_dim]
    pub w_k: Tensor, // [hidden, head_dim]
    pub w_v: Tensor, // [hidden, head_dim]
    pub w_o: Tensor, // [head_dim, hidden]
}

/// Per-block master weights as plain Tensors. Mirror of `BlockWeights<'t>`
/// (which holds Vars on a tape); these live across training steps.
///
/// FFN form: SwiGLU. `ffn_gate_w` and `ffn_up_w` share the same `[hidden, ffn_dim]`
/// shape; the gate goes through SiLU and is then element-wise multiplied with
/// the up projection. `ffn_down_w` brings the gated product back to hidden.
#[derive(Debug, Clone)]
pub struct BlockMasters {
    pub heads: Vec<AttentionHead>, // length = config.n_heads
    pub ffn_gate_w: Tensor,        // [hidden, ffn_dim]
    pub ffn_up_w: Tensor,          // [hidden, ffn_dim]
    pub ffn_down_w: Tensor,        // [ffn_dim, hidden]
}

/// Top-level model. Owns every trainable tensor.
///
/// Since v0.17 the LM head is tied to `token_embed` (transposed at op-build
/// time inside `forward`), so there is no separate `lm_head` tensor.
#[derive(Debug, Clone)]
pub struct Model {
    pub token_embed: Tensor, // [vocab, hidden]; doubles as the LM-head weight
    pub blocks: Vec<BlockMasters>,
    pub config: ModelConfig,
}

/// Tape-side mirror of `Model`: every master tensor registered as a leaf.
/// Lifetime `'t` ties this to a specific `Tape` instance. No `lm_head` field
/// since v0.17 - the LM-head matmul reads `token_embed` (transposed) directly.
pub struct ModelLeaves<'t> {
    pub token_embed: Var<'t>,
    pub blocks: Vec<BlockWeights<'t>>,
}

/// Tiny linear congruential generator - Numerical Recipes constants.
/// Not cryptographic, not statistically great; just good enough to break
/// initialisation symmetry without adding a `rand` crate dependency.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
    /// Uniform sample in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 41) as f32 / (1u32 << 22) as f32) - 1.0
    }
    fn fill_tensor(&mut self, shape: Vec<usize>, scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| self.next_f32() * scale).collect();
        Tensor { data, shape }
    }
}

impl Model {
    /// Build a freshly-initialised model. `seed` controls the LCG used for
    /// random init - same seed, same model.
    pub fn new(config: &ModelConfig, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let h = config.hidden_dim;
        let d = config.head_dim;
        let f = config.ffn_dim;

        // Token embedding init: scale `1/√hidden_dim`. This is also the
        // LM-head's effective scale since v0.17 (tied embeddings); a 1/√h
        // init gives decisive output logits at random init even at tiny
        // test scales, where GPT-2's flat 0.02 would produce near-tied
        // logits (the residual-stream depth needed to amplify 0.02-scale
        // weights into a decisive logit gap is several blocks; toy tests
        // use 1-2 blocks). For production-scale configs (h = 192-256)
        // this is 0.06-0.07, very close to 0.02 anyway, so the change
        // doesn't meaningfully shift the trained-model dynamics. No
        // learned positional embedding: RoPE injects position info inside
        // attention (see `autograd::rope`).
        let scale_embed = 1.0 / (h as f32).sqrt();
        let token_embed = rng.fill_tensor(vec![config.vocab_size, h], scale_embed);

        // Linear-layer inits: scale by 1/√fan_in (a poor man's Kaiming).
        let scale_h_d = 1.0 / (h as f32).sqrt();
        let scale_d_h = 1.0 / (d as f32).sqrt();
        let scale_h_f = 1.0 / (h as f32).sqrt();
        let scale_f_h = 1.0 / (f as f32).sqrt();

        let blocks = (0..config.n_blocks)
            .map(|_| {
                let heads = (0..config.n_heads)
                    .map(|_| AttentionHead {
                        w_q: rng.fill_tensor(vec![h, d], scale_h_d),
                        w_k: rng.fill_tensor(vec![h, d], scale_h_d),
                        w_v: rng.fill_tensor(vec![h, d], scale_h_d),
                        w_o: rng.fill_tensor(vec![d, h], scale_d_h),
                    })
                    .collect();
                BlockMasters {
                    heads,
                    ffn_gate_w: rng.fill_tensor(vec![h, f], scale_h_f),
                    ffn_up_w: rng.fill_tensor(vec![h, f], scale_h_f),
                    ffn_down_w: rng.fill_tensor(vec![f, h], scale_f_h),
                }
            })
            .collect();

        Self {
            token_embed,
            blocks,
            config: *config,
        }
    }

    /// Register every master tensor on the given tape as a fresh leaf.
    /// Returns the bundle of leaf handles you'll pass to `forward` / `apply_grads`.
    /// `token_embed` is registered once but used twice in the graph (once for
    /// the embed lookup, once for the LM-head matmul via transpose) - the tape
    /// accumulates gradient contributions from both paths automatically.
    pub fn register_leaves<'t>(&self, tape: &'t Tape) -> ModelLeaves<'t> {
        use crate::block::AttentionHeadVars;
        ModelLeaves {
            token_embed: Var::leaf(tape, self.token_embed.clone()),
            blocks: self
                .blocks
                .iter()
                .map(|b| BlockWeights {
                    heads: b
                        .heads
                        .iter()
                        .map(|h| AttentionHeadVars {
                            w_q: Var::leaf(tape, h.w_q.clone()),
                            w_k: Var::leaf(tape, h.w_k.clone()),
                            w_v: Var::leaf(tape, h.w_v.clone()),
                            w_o: Var::leaf(tape, h.w_o.clone()),
                        })
                        .collect(),
                    ffn_gate_w: Var::leaf(tape, b.ffn_gate_w.clone()),
                    ffn_up_w: Var::leaf(tape, b.ffn_up_w.clone()),
                    ffn_down_w: Var::leaf(tape, b.ffn_down_w.clone()),
                })
                .collect(),
        }
    }

    /// Forward pass: token ids -> logits.  Builds the full graph on the tape.
    /// Returns the logits Var; loss + backward are caller's responsibility.
    pub fn forward<'t>(&self, leaves: &ModelLeaves<'t>, ids: &[usize]) -> Var<'t> {
        let seq = ids.len();
        assert!(
            seq <= self.config.max_seq_len,
            "forward: seq_len {} exceeds max_seq_len {}",
            seq,
            self.config.max_seq_len
        );

        // Embed tokens. Position information is added inside attention via
        // RoPE (`autograd::rope`), not here - so the input to the block stack
        // is just the token-embedding lookup, no learned positional add.
        let mut x = leaves.token_embed.clone().embed(ids);

        // Stack of transformer blocks.
        for bw in &leaves.blocks {
            x = transformer_block(x, bw, self.config.head_dim);
        }

        // Final pre-norm before the LM head, same RMSNorm pattern as inside blocks.
        let x = x.rmsnorm();

        // LM head as BitLinear: tied to `token_embed`, transposed. Backward
        // through transpose + STE accumulates a [vocab, hidden] gradient
        // contribution into `token_embed`'s grad slot, on top of the gather
        // gradient from the embed-lookup path above.
        x.quantise_acts_ste().matmul(
            leaves
                .token_embed
                .clone()
                .transpose_2d()
                .quantise_weights_ste(),
        )
    }

    /// SGD update: `master -= lr · master.grad()`. Reads gradient from each
    /// leaf, applies the update to the corresponding master tensor in-place.
    /// Call AFTER `tape.backward(loss.id)`, BEFORE dropping the tape.
    /// Kept as a baseline next to AdamW; exercised in `model::tests`.
    #[allow(dead_code)]
    pub fn apply_grads(&mut self, leaves: &ModelLeaves<'_>, lr: f32) {
        self.for_each_param_with_grad(leaves, |p, g| {
            for i in 0..p.data.len() {
                p.data[i] -= lr * g.data[i];
            }
        });
    }

    /// Visit every (master parameter, leaf gradient) pair in canonical order.
    /// Optimisers (`SGD`, `AdamW`, etc.) drive their iteration through here so
    /// they don't need to know the model's internal layout.
    ///
    /// Order: token_embed, then per block (each head's q, k, v, o followed by
    /// ffn_gate, ffn_up, ffn_down). Since v0.17 there is no separate lm_head
    /// at the tail - the LM-head matmul reads `token_embed` (transposed), so
    /// its gradient already accumulates into the first slot. The `optim::AdamW`
    /// state vectors are sized to match - changing the order here breaks
    /// resume. (Pre-v0.12 the order included `pos_embed` after `token_embed`;
    /// RoPE retired that tensor and bumped magic to `BNT4`. Tied embeddings
    /// drops the trailing `lm_head` slot and bumps magic to `BNT5`.)
    pub fn for_each_param_with_grad<F>(&mut self, leaves: &ModelLeaves<'_>, mut f: F)
    where
        F: FnMut(&mut Tensor, &Tensor),
    {
        f(&mut self.token_embed, &leaves.token_embed.grad());
        for (mb, lb) in self.blocks.iter_mut().zip(&leaves.blocks) {
            for (mh, lh) in mb.heads.iter_mut().zip(&lb.heads) {
                f(&mut mh.w_q, &lh.w_q.grad());
                f(&mut mh.w_k, &lh.w_k.grad());
                f(&mut mh.w_v, &lh.w_v.grad());
                f(&mut mh.w_o, &lh.w_o.grad());
            }
            f(&mut mb.ffn_gate_w, &lb.ffn_gate_w.grad());
            f(&mut mb.ffn_up_w, &lb.ffn_up_w.grad());
            f(&mut mb.ffn_down_w, &lb.ffn_down_w.grad());
        }
    }

    /// Iterate every master parameter (mutable, no leaf grad). Used by the
    /// batched training path: gradients arrive pre-aggregated as a `Vec<Tensor>`
    /// in visitor order, so the optimiser walks parameters here and pulls the
    /// matching gradient by index instead of going through a tape leaf.
    /// Same canonical order as `for_each_param_with_grad`.
    pub fn for_each_param_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Tensor),
    {
        f(&mut self.token_embed);
        for mb in self.blocks.iter_mut() {
            for mh in mb.heads.iter_mut() {
                f(&mut mh.w_q);
                f(&mut mh.w_k);
                f(&mut mh.w_v);
                f(&mut mh.w_o);
            }
            f(&mut mb.ffn_gate_w);
            f(&mut mb.ffn_up_w);
            f(&mut mb.ffn_down_w);
        }
    }

    /// Iterate every leaf-gradient (read-only). Used by tests + grad-clip
    /// debug printers; not called from the main training loop directly.
    #[allow(dead_code)]
    pub fn for_each_grad<F: FnMut(&Tensor)>(&self, leaves: &ModelLeaves<'_>, mut f: F) {
        f(&leaves.token_embed.grad());
        for lb in &leaves.blocks {
            for lh in &lb.heads {
                f(&lh.w_q.grad());
                f(&lh.w_k.grad());
                f(&lh.w_v.grad());
                f(&lh.w_o.grad());
            }
            f(&lb.ffn_gate_w.grad());
            f(&lb.ffn_up_w.grad());
            f(&lb.ffn_down_w.grad());
        }
    }

    /// Canonical parameter shapes, in the same order as the visitors above.
    /// Used by `AdamW::new_for` to size its momentum / variance buffers.
    pub fn param_shapes(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        out.push(self.token_embed.shape.clone());
        for b in &self.blocks {
            for h in &b.heads {
                out.push(h.w_q.shape.clone());
                out.push(h.w_k.shape.clone());
                out.push(h.w_v.shape.clone());
                out.push(h.w_o.shape.clone());
            }
            out.push(b.ffn_gate_w.shape.clone());
            out.push(b.ffn_up_w.shape.clone());
            out.push(b.ffn_down_w.shape.clone());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        // 2 heads × 2 head_dim = hidden_dim 4. Cleanest invariant for tests:
        // n_heads * head_dim == hidden_dim. Verifies the head-loop sums correctly
        // across multiple heads (1 head would silently allow concat-vs-sum bugs).
        ModelConfig {
            vocab_size: 8,
            hidden_dim: 4,
            n_heads: 2,
            head_dim: 2,
            ffn_dim: 8,
            max_seq_len: 4,
            n_blocks: 2,
        }
    }

    #[test]
    fn model_forward_produces_correct_logits_shape() {
        let cfg = tiny_config();
        let model = Model::new(&cfg, 42);
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);

        let ids = vec![3, 1, 4, 1]; // length 4 = max_seq_len
        let logits = model.forward(&leaves, &ids);

        assert_eq!(logits.value().shape, vec![4, cfg.vocab_size]);
        for &v in logits.value().data.iter() {
            assert!(v.is_finite(), "logit non-finite: {}", v);
        }
    }

    #[test]
    fn model_backward_routes_gradient_to_every_master_weight() {
        // The integration test for the entire model graph: every leaf in
        // every block + the embeddings + LM head must receive non-zero gradient.
        let cfg = tiny_config();
        let model = Model::new(&cfg, 0);
        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);

        let ids = vec![1, 2, 3, 4];
        let targets = vec![2, 3, 4, 5];

        let logits = model.forward(&leaves, &ids);
        let loss = logits.cross_entropy(&targets);
        tape.backward(loss.id);

        let has_nonzero = |t: &Tensor| t.data.iter().any(|&v| v.abs() > 1e-10);
        // token_embed receives gradient from BOTH the embed-lookup path (gather
        // backward, scatter-adds onto rows for tokens used in `ids`) AND the
        // LM-head matmul (transpose backward routes a [vocab, hidden] grad
        // contribution back). Tied since v0.17.
        assert!(has_nonzero(&leaves.token_embed.grad()), "token_embed");
        for (i, b) in leaves.blocks.iter().enumerate() {
            for (j, h) in b.heads.iter().enumerate() {
                assert!(has_nonzero(&h.w_q.grad()), "block {} head {} w_q", i, j);
                assert!(has_nonzero(&h.w_k.grad()), "block {} head {} w_k", i, j);
                assert!(has_nonzero(&h.w_v.grad()), "block {} head {} w_v", i, j);
                assert!(has_nonzero(&h.w_o.grad()), "block {} head {} w_o", i, j);
            }
            assert!(has_nonzero(&b.ffn_gate_w.grad()), "block {} ffn_gate_w", i);
            assert!(has_nonzero(&b.ffn_up_w.grad()), "block {} ffn_up_w", i);
            assert!(has_nonzero(&b.ffn_down_w.grad()), "block {} ffn_down_w", i);
        }
    }

    #[test]
    fn training_steps_reduce_loss_on_a_fixed_window() {
        // Multistep rather than single-step because STE makes the per-step
        // loss path noisy: the gradient is computed on the continuous loss
        // surface (where STE pretends the quantiser is identity), but the
        // forward runs on the ternary loss surface, which has discrete jumps
        // at quant boundaries. Single-step loss can go either way.
        // Over 50 steps the noise averages out and the trend is down.
        //
        // This is also the right shape for the M9 gate (portion 3 will
        // train on the real corpus and assert the same property).
        let cfg = tiny_config();
        let mut model = Model::new(&cfg, 7);

        let ids = vec![1, 2, 3, 4];
        let targets = vec![2, 3, 4, 5];
        let lr = 0.05_f32;
        let n_steps = 50;

        // Helper closure to compute current loss without mutating the model.
        let compute_loss = |m: &Model| -> f32 {
            let tape = Tape::new();
            let leaves = m.register_leaves(&tape);
            let logits = m.forward(&leaves, &ids);
            logits.cross_entropy(&targets).value().data[0]
        };

        let initial_loss = compute_loss(&model);

        for _ in 0..n_steps {
            let tape = Tape::new();
            let leaves = model.register_leaves(&tape);
            let logits = model.forward(&leaves, &ids);
            let loss = logits.cross_entropy(&targets);
            tape.backward(loss.id);
            model.apply_grads(&leaves, lr);
        }

        let final_loss = compute_loss(&model);

        // Significant reduction over 50 steps. Tolerance is loose because the
        // ternary loss surface has irreducible plateaus - for a 4-token, 8-vocab
        // toy the model can't reach zero loss, but it should comfortably halve
        // the initial cross-entropy.
        assert!(
            final_loss < initial_loss * 0.7,
            "loss did not drop enough over {} steps: initial = {}, final = {}",
            n_steps,
            initial_loss,
            final_loss
        );
    }
}
