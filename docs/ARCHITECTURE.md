# Architecture

How the modules compose, top-down.

## Layered view

```
                                 main.rs
                       (TrainConfig, demos, CLI)
                                    |
                ┌───────────────────┼────────────────────┐
                |                   |                    |
            inference.rs        optim.rs            export.rs
       (greedy + temp gen)   (AdamW, clip, LR)    (binary I/O, import)
                |                   |                    |
                └───────────────┐   |   ┌────────────────┘
                                |   |   |
                              model.rs
                  (Model, ModelConfig, leaf register, init)
                                    |
                ┌───────────────────┼───────────────────┐
                |                   |                   |
            block.rs            attention.rs           ffn.rs
       (RMSNorm + attn      (single-head, causal,    (BitLinear up,
        + FFN + residual)    softmax over keys)       ReLU, BitLinear down)
                                    |
                              autograd.rs
            (Tape, Var, all ops: matmul, softmax, rmsnorm,
             quantise_weights_ste, quantise_acts_ste,
             causal_mask, embed, cross_entropy, etc.)
                                    |
                          tensor.rs    bitlinear.rs
                  (raw f32 storage)  (absmean_ternary, absmax_int8)

                              data.rs
              (Vocab, sliding windows, LCG, file reader; standalone)
```

## Why this layering

- **`tensor.rs`** is a pure value type. No autograd awareness, no quantisation.
  Pure linear algebra primitives.

- **`bitlinear.rs`** holds the two quantisation primitives as free functions.
  These are the only places that compute gamma and alpha; everything else
  consumes `(W_q, gamma)` or `(x_q, alpha)` returned from here.

- **`autograd.rs`** is the centerpiece. It owns the `Tape` data structure,
  the `Var` handle, and every op the project needs. STE wrappers
  (`quantise_weights_ste`, `quantise_acts_ste`) live here too, so the
  autograd-aware path stays in one file.

- **`attention.rs`, `ffn.rs`, `block.rs`** are short composition layers built
  on `Var` ops. They contain no state; they're pure functions taking `Var`s
  in and returning a `Var`.

- **`model.rs`** owns the master parameters as `Tensor`s and exposes a
  parameter-visitor API. The `register_leaves` method binds a fresh
  `Var` set to a tape for one forward; `apply_grads` and `for_each_param_with_grad`
  let optimisers iterate the model's tensors uniformly.

- **`optim.rs`** consumes the visitor. AdamW maintains its own `m`, `v`
  buffers indexed by visitor order. Gradient clipping touches the leaves
  directly via `Tape::write_grad`.

- **`export.rs`** owns the on-disk format. Header + payload.
  Round-trippable in all three formats. The format byte plus a 4-byte magic
  let importers detect the right reader.

- **`inference.rs`** is greedy + temperature sampling. Both build the same
  forward graph; greedy takes argmax of the last-position logits, temperature
  samples by inverse CDF.

- **`data.rs`** stands alone. Vocab, sliding windows, file reader, LCG,
  shuffler. No upstream dependencies on anything except `std`.

- **`main.rs`** is the integration layer: CLI dispatch, `TrainConfig` struct
  with its hyperparameters, the M4-M10 demos, and the integration tests
  that gate each milestone.

## The training cycle (one step)

```
master tensors live in Model (outside any tape)
    │
    ▼
Model::register_leaves(&tape)
    Var leaves on the tape, one per master tensor
    │
    ▼
Model::forward(&leaves, ids)
    builds the graph: embed → blocks → final RMSNorm → BitLinear lm_head
    every op records itself on the tape
    │
    ▼
logits.cross_entropy(targets)
    fused softmax + NLL with closed-form (softmax − onehot) backward
    │
    ▼
tape.backward(loss.id)
    seeds output grad with ones, walks tape in reverse
    each leaf accumulates ∂L/∂master in its grad cell
    │
    ▼
clip_grad_norm(&leaves, 1.0)
    rescales every leaf grad in place if global L2 norm exceeds the cap
    │
    ▼
opt.step(&mut model, &leaves)
    visits every (master, grad) pair and applies AdamW update
    │
    ▼
tape drops at end of scope
    every Var, every saved tensor, every backward closure released
    only the master Tensors persist into the next step
```

## STE in one paragraph

Forward through `quantise_weights_ste(w)` returns `gamma * W_q` where
`(W_q, gamma) = absmean_ternary(w)`. The forward value passes through ternary
quantisation. The backward closure ignores the discreteness and returns the
incoming gradient unchanged: `vec![(parent_id, grad.clone())]`. This is the
"straight-through" lie. Empirically it works because gradient *direction*
matters more than its exact magnitude under SGD/AdamW, and the quantiser
preserves direction information (the master moves; the ternary form follows).

## Causal mask

Without causal masking, attention at position `i` could see input row `i+1`,
trivially predicting the target `input[i+1]`. The model would converge to
near-zero loss without learning language. `Var::causal_mask` sets
`scores[i, j] = -inf` for `j > i` before softmax, so positions to the right
get zero attention weight. Backward zeroes gradient on the upper triangle and
passes it through on the lower, matching the forward selection.

## Memory model

The `Tape` owns every recorded value and gradient cell. `Var<'t>` is a `Copy`
handle into a tape; cloning a `Var` is free, the underlying storage is in the
tape's `Vec<Node>`. Building one tape per training step and dropping it after
backward gives bounded memory: the maximum is one step's worth of saved
forward tensors plus their grad cells.

Master parameters in `Model` live across steps; everything else is ephemeral.

## Where the toy stops

Three places this implementation deliberately diverges from production BitNet:

1. **Single-head attention.** Real BitNet uses multi-head. We sum head outputs
   trivially and call it done. See `TODO.md` for the deferred refactor.

2. **f32 throughout.** Real BitNet has BF16 master weights and INT8 activations
   stored as `i8` (not `f32` representations of `i8`). The integer matmul that
   produces the speedup is also missing; we keep all arithmetic in f32. The
   *values* are quantised; the *arithmetic* isn't.

3. **No KV cache.** Generation recomputes the full forward each token. Real
   inference caches K and V across positions to make per-token cost constant.

The first two are listed in `TODO.md`. The third is genuine ML-engineering
territory beyond the curriculum's M10 finale.
