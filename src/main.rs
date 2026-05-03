//! M4 entry point: 1D linear regression by SGD, demonstrating that the
//! hand-rolled autograd in src/autograd.rs actually drives loss downward.
//!
//! Task:   fit  y = 2x  to 8 hand-picked points.
//! Model:  pred = x · w     (single scalar weight, no bias - bias broadcast
//!                            isn't worth the complexity until something else
//!                            forces it; M7 attention will).
//! Loss:    mean((pred − y)²)
//! Update:  w <- w - lr · dL/dw   (vanilla SGD)

// Crate-level clippy allows. Pedantic surfaces a lot of stylistic noise that
// isn't worth fixing in a research / learning codebase; the items left to fail
// are the ones that catch actual bugs.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::float_cmp)]

mod attention;
mod autograd;
mod bitlinear;
mod block;
mod data;
mod export;
mod ffn;
mod inference;
mod model;
mod optim;
mod tensor;

use autograd::{Tape, Var};
use tensor::Tensor;

/// One full training run. Returns `(initial_loss, final_loss, final_w)` so the
/// integration test below can sanity-check convergence without parsing stdout.
fn train_linear_regression(n_steps: usize, lr: f32) -> (f32, f32, f32) {
    // ── Synthetic data: 8 points exactly on  y = 2x  (no noise, no offset). ──
    let x_data = Tensor::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], vec![8, 1]);
    let y_data = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![8, 1]);

    // Master weight: lives OUTSIDE any tape. Each step we will clone its current
    // value into a fresh tape as a leaf, then multiply this master after backward.
    // Started at zero - for 1D regression that's fine because the gradient at
    // zero points unambiguously toward the global minimum.
    let mut w = Tensor::from_vec(vec![0.0], vec![1, 1]);

    let mut initial_loss: Option<f32> = None;
    let mut final_loss = 0.0_f32;

    for step in 0..n_steps {
        // Fresh tape every iteration. The old tape - including all saved input
        // tensors captured by backward closures - is released when this scope
        // ends, so memory is bounded by ONE step's worth of ops, not the full run.
        let tape = Tape::new();

        // Register inputs, target, and the current weight as leaves of this tape.
        // The data is cloned IN; the master `w` continues to live outside.
        let x_var = Var::leaf(&tape, x_data.clone());
        let y_var = Var::leaf(&tape, y_data.clone());
        let w_var = Var::leaf(&tape, w.clone());

        // Forward:
        //   pred = x · w           shape [8, 1]
        //   diff = pred − y        shape [8, 1]
        //   sq   = diff ⊙ diff     shape [8, 1]   (square via reused leaf;
        //                                          accumulation gives ∂sq/∂diff = 2·diff)
        //   loss = mean(sq)        shape [1]      (scalar - clean seed for backward)
        let pred = x_var.matmul(w_var);
        let diff = pred.sub(y_var);
        let sq = diff.mul(diff);
        let loss = sq.mean();

        let loss_value = loss.value().data[0];
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        // Backward - fills `w_var.grad()` with ∂loss/∂w.
        tape.backward(loss.id);

        // SGD update on the MASTER (not on w_var, which is about to die with the tape).
        //   w_master ← w_master − lr · ∂L/∂w
        let g = w_var.grad();
        for i in 0..w.data.len() {
            w.data[i] -= lr * g.data[i];
        }

        // Periodic log so convergence is visible to the eye.
        if step % 20 == 0 || step == n_steps - 1 {
            println!(
                "step {:>4}   loss = {:.6e}   w = {:.6}",
                step, loss_value, w.data[0]
            );
        }
        // `tape` is dropped here - entire computational graph released.
    }

    (initial_loss.unwrap(), final_loss, w.data[0])
}

/// M6 gate: BitLinear-shaped training on `y = 2x₁ − 2x₂`.
///
/// Same training-loop structure as `train_linear_regression`, but the forward
/// graph runs through STE-quantised weights and per-row STE-quantised
/// activations. If the gradient successfully threads back through the quantisers
/// (i.e. STE works), `master_w` converges to ≈ (2, −2). If STE is broken, every
/// gradient through the quantiser is zero and `master_w` stays frozen at zeros.
///
/// Returns `(initial_loss, final_loss, [w0, w1])`.
fn train_bitlinear_regression(n_steps: usize, lr: f32) -> (f32, f32, Vec<f32>) {
    // ── Synthetic data ──
    // 4 rows, 2 features, single output. Rows include both axis-aligned cases
    // (1,0) and (0,1) and "mixed" cases (2,1), (1,2) so the per-row activation
    // quantisation gets exercised at different α magnitudes.
    let x_data = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0], vec![4, 2]);
    // y = 2 · x1  −  2 · x2  (exactly hits the ternary grid at γ=2, W_q=[+1,−1]).
    let y_data = Tensor::from_vec(vec![2.0, -2.0, 2.0, -2.0], vec![4, 1]);

    // Master weight: shape [in, out] = [2, 1].
    // Zeros init is fine - STE pushes off zero on the first step (W_eff=0 still
    // produces a non-zero gradient via the matmul-of-x part of the chain).'
    let mut master_w = Tensor::from_vec(vec![0.0, 0.0], vec![2, 1]);

    let mut initial_loss: Option<f32> = None;
    let mut final_loss = 0.0_f32;

    for step in 0..n_steps {
        let tape = Tape::new();
        let x_var = Var::leaf(&tape, x_data.clone());
        let y_var = Var::leaf(&tape, y_data.clone());
        let w_var = Var::leaf(&tape, master_w.clone());

        // BitLinear-shaped forward graph:
        //   x_eff = quantise_acts_ste(x)        per-row INT8 dequant, identity backward
        //   w_eff = quantise_weights_ste(w)     ternary dequant,      identity backward
        //   pred  = x_eff · w_eff               plain matmul on dequantised values
        //   loss  = mean((pred − y)²)
        //
        // Note: w_var is stored as [in, out] = [2, 1], matching matmul's expectations
        // directly - no transpose needed here. (Var::transpose_2d exists from M6 portion 1
        // for cases where you start from a [out, in] layout, e.g. if/when we wrap
        // the BitLinear *struct* itself onto the tape.)
        let x_eff = x_var.quantise_acts_ste();
        let w_eff = w_var.quantise_weights_ste();
        let pred = x_eff.matmul(w_eff);
        let diff = pred.sub(y_var);
        let sq = diff.mul(diff);
        let loss = sq.mean();

        let loss_value = loss.value().data[0];
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        tape.backward(loss.id);

        // SGD on master_w. Gradient comes through STE - the quantiser's
        // identity-backward is what makes this nonzero.
        let g = w_var.grad();
        for i in 0..master_w.data.len() {
            master_w.data[i] -= lr * g.data[i];
        }

        if step % 50 == 0 || step == n_steps - 1 {
            println!(
                "step {:>4}   loss = {:.6e}   w = [{:>+8.4}, {:>+8.4}]",
                step, loss_value, master_w.data[0], master_w.data[1]
            );
        }
    }

    (initial_loss.unwrap(), final_loss, master_w.data.clone())
}

/// Hyperparameters for one full BitNet LM training run. Sensible defaults
/// for the embedded TINY_CORPUS demo; override fields for real corpora.
pub struct TrainConfig {
    pub n_steps: usize,
    pub peak_lr: f32,
    pub floor_lr: f32,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub weight_decay: f32,
    pub adamw_beta1: f32,
    pub adamw_beta2: f32,
    pub seed: u64,
    pub model: crate::model::ModelConfig,
    /// If `Some(path)`, train on the file's contents instead of `TINY_CORPUS`.
    pub corpus_path: Option<std::path::PathBuf>,
    /// Print a status line every `log_every` steps (and on the final step).
    pub log_every: usize,
    /// If `Some(model)`, start training from these weights instead of fresh
    /// random init. The model's `ModelConfig` overrides `self.model` so the
    /// optimiser sizes and forward shapes match the loaded weights exactly.
    /// Optimiser state (AdamW m, v) and LR schedule warmup both restart fresh;
    /// expect a brief instability in the first ~30 steps after resuming.
    pub start_from: Option<crate::model::Model>,
}

impl TrainConfig {
    /// Defaults targeting the embedded TINY_CORPUS demo: small model, 300
    /// steps, AdamW with cosine LR + grad clipping.
    pub fn tiny_demo() -> Self {
        Self {
            n_steps: 300,
            peak_lr: 5e-3,
            floor_lr: 5e-4,
            warmup_steps: 30,
            grad_clip: 1.0,
            weight_decay: 0.05,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            seed: 1337,
            model: crate::model::ModelConfig {
                vocab_size: 0, // overridden after vocab is built
                hidden_dim: 16,
                head_dim: 16,
                ffn_dim: 32,
                max_seq_len: 16,
                n_blocks: 2,
            },
            corpus_path: None,
            log_every: 50,
            start_from: None,
        }
    }

    /// Defaults targeting the full TinyShakespeare corpus at `data/tinyshakespeare.txt`.
    /// Bigger model, longer sequences, more steps. Expect ~10-30 minutes on CPU.
    pub fn shakespeare() -> Self {
        Self {
            n_steps: 5_000,
            peak_lr: 3e-3,
            floor_lr: 3e-4,
            warmup_steps: 200,
            grad_clip: 1.0,
            weight_decay: 0.1,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            seed: 1337,
            model: crate::model::ModelConfig {
                vocab_size: 0, // filled after vocab is built
                hidden_dim: 64,
                head_dim: 64,
                ffn_dim: 128,
                max_seq_len: 64,
                n_blocks: 4,
            },
            corpus_path: Some(std::path::PathBuf::from("data/tinyshakespeare.txt")),
            log_every: 100,
            start_from: None,
        }
    }
}

/// Train the full BitNet LM on either the embedded TINY_CORPUS or a corpus
/// loaded from disk. AdamW + global-L2 grad clip + cosine-with-warmup LR.
/// Windows are Fisher-Yates shuffled at the start of every epoch.
///
/// Returns `(initial_loss, min_loss_seen, trained_model, vocab)`.
fn train_bitnet_lm(cfg: TrainConfig) -> (f32, f32, crate::model::Model, crate::data::Vocab) {
    use crate::data::{Lcg, TINY_CORPUS, Vocab, make_windows, read_corpus};
    use crate::model::Model;
    use crate::optim::{AdamW, clip_grad_norm, cosine_lr};

    let corpus_owned: String = match &cfg.corpus_path {
        Some(p) => match read_corpus(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "could not read corpus {:?}: {}; falling back to TINY_CORPUS",
                    p, e
                );
                TINY_CORPUS.to_string()
            }
        },
        None => TINY_CORPUS.to_string(),
    };

    let vocab = Vocab::from_text(&corpus_owned);
    let ids = vocab.encode(&corpus_owned);

    // If we're resuming, the loaded model's config wins (vocab + dimensions
    // must match the loaded weights). Otherwise build a fresh model from the
    // requested config + corpus vocab.
    let (mut model, model_cfg) = if let Some(loaded) = cfg.start_from {
        let lc = loaded.config;
        if lc.vocab_size != vocab.size() {
            eprintln!(
                "warning: loaded model vocab_size {} differs from corpus vocab {}.\n  \
                 Token ids will not match between training-time and load-time vocabs;\n  \
                 generation will produce nonsense unless the same corpus was used originally.",
                lc.vocab_size,
                vocab.size()
            );
        }
        println!(
            "resuming from checkpoint: hidden {}, ffn {}, head_dim {}, blocks {}, seq_len {}",
            lc.hidden_dim, lc.ffn_dim, lc.head_dim, lc.n_blocks, lc.max_seq_len
        );
        (loaded, lc)
    } else {
        let mut mc = cfg.model;
        mc.vocab_size = vocab.size();
        (Model::new(&mc, cfg.seed), mc)
    };
    let windows = make_windows(&ids, model_cfg.max_seq_len);

    let mut rng = Lcg::new(cfg.seed ^ 0xDEADBEEF);

    let mut opt = AdamW::new_for(&model, cfg.peak_lr);
    opt.beta1 = cfg.adamw_beta1;
    opt.beta2 = cfg.adamw_beta2;
    opt.weight_decay = cfg.weight_decay;

    println!(
        "── M9: BitNet LM training ──\n\
         corpus     = {} chars\n\
         vocab      = {}\n\
         model      = hidden {}, ffn {}, head_dim {}, blocks {}, seq_len {}\n\
         windows    = {}\n\
         optimiser  = AdamW(b1={}, b2={}, wd={}), peak_lr={:.1e}, floor_lr={:.1e},\n\
                      warmup={} steps, grad_clip={}, total_steps={}",
        corpus_owned.chars().count(),
        vocab.size(),
        model_cfg.hidden_dim,
        model_cfg.ffn_dim,
        model_cfg.head_dim,
        model_cfg.n_blocks,
        model_cfg.max_seq_len,
        windows.len(),
        cfg.adamw_beta1,
        cfg.adamw_beta2,
        cfg.weight_decay,
        cfg.peak_lr,
        cfg.floor_lr,
        cfg.warmup_steps,
        cfg.grad_clip,
        cfg.n_steps,
    );

    let eval_loss = |m: &Model, input: &[usize], target: &[usize]| -> f32 {
        let tape = Tape::new();
        let leaves = m.register_leaves(&tape);
        let logits = m.forward(&leaves, input);
        logits.cross_entropy(target).value().data[0]
    };

    let (input0, target0) = (windows[0].0.clone(), windows[0].1.clone());
    let initial_loss = eval_loss(&model, &input0, &target0);
    let mut min_loss = initial_loss;

    for step in 0..cfg.n_steps {
        // Sample a fresh random window every step.  The previous "shuffle once,
        // iterate" approach silently restricted training to the first n_steps
        // windows of the shuffle, which was tiny relative to the corpus and
        // caused catastrophic memorisation.  Uniform random sampling makes
        // every step a draw from the full 1.1M-window pool.
        let (input, target) = &windows[rng.gen_range(windows.len())];

        let lr = cosine_lr(
            step,
            cfg.warmup_steps,
            cfg.n_steps,
            cfg.peak_lr,
            cfg.floor_lr,
        );
        opt.lr = lr;

        let tape = Tape::new();
        let leaves = model.register_leaves(&tape);
        let logits = model.forward(&leaves, input);
        let loss = logits.cross_entropy(target);
        let loss_val = loss.value().data[0];
        if loss_val < min_loss {
            min_loss = loss_val;
        }
        tape.backward(loss.id);
        let pre_clip = clip_grad_norm(&leaves, cfg.grad_clip);
        opt.step(&mut model, &leaves);

        if step % cfg.log_every == 0 || step == cfg.n_steps - 1 {
            let anchor_loss = eval_loss(&model, &input0, &target0);
            println!(
                "step {:>5}   train_loss = {:.4}   anchor_loss = {:.4}   min_seen = {:.4}   lr = {:.4e}   |g| = {:.3}",
                step, loss_val, anchor_loss, min_loss, lr, pre_clip
            );
        }
    }

    (initial_loss, min_loss, model, vocab)
}

/// Directory where every trained model gets written. Created on demand by
/// `models_path` so the user never has to make it manually.
const MODELS_DIR: &str = "models";

/// Build a path under `models/` and ensure the directory exists. Failures to
/// create the directory are logged and the path is returned anyway, letting
/// the eventual `fs::write` produce its own clearer error.
fn models_path(filename: &str) -> std::path::PathBuf {
    if let Err(e) = std::fs::create_dir_all(MODELS_DIR) {
        eprintln!("could not create {}/: {}", MODELS_DIR, e);
    }
    std::path::PathBuf::from(MODELS_DIR).join(filename)
}

/// Entry point for the standalone Shakespeare training run. Called by
/// `cargo run --release -- shakespeare [resume_path]`. Trains (or continues
/// training) a larger BitNet on the full TinyShakespeare corpus, exports all
/// three formats, and generates samples in greedy + two temperature modes.
fn run_shakespeare_training(resume_path: Option<std::path::PathBuf>) {
    let mut cfg = TrainConfig::shakespeare();
    let path_present = cfg
        .corpus_path
        .as_ref()
        .map_or(false, |p| p.exists());
    if !path_present {
        eprintln!(
            "Could not find data/tinyshakespeare.txt.\n\
             Download it with:\n  \
             curl -sSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \\\n    \
             -o data/tinyshakespeare.txt"
        );
        std::process::exit(1);
    }

    // If a resume path is provided, load the model and seed the config with it.
    if let Some(p) = resume_path {
        let mut f = match std::fs::File::open(&p) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open resume file {}: {}", p.display(), e);
                std::process::exit(1);
            }
        };
        match export::import(&mut f) {
            Ok((m, fmt)) => {
                println!("loaded {:?}-format checkpoint from {}", fmt, p.display());
                cfg.start_from = Some(m);
            }
            Err(e) => {
                eprintln!("could not parse resume file {}: {}", p.display(), e);
                std::process::exit(1);
            }
        }
    }

    let (initial, min_seen, model, vocab) = train_bitnet_lm(cfg);
    println!(
        "\nShakespeare training done.  initial = {:.4}   min seen = {:.4}   ratio = {:.2}",
        initial,
        min_seen,
        min_seen / initial
    );

    println!("\n-- greedy generation --");
    for prompt in ["ROMEO:", "To be ", "King "] {
        let g = inference::generate(&model, &vocab, prompt, 200);
        println!("\nprompt: {:?}\n{}", prompt, g);
    }

    // Temperature sampling: T=0.8 (cooler, more conservative) and T=1.0 (raw model
    // distribution). Each call uses its own RNG seed so re-runs can be compared.
    let mut rng = data::Lcg::new(0xCAFEF00D);
    println!("\n-- temperature sampling (T=0.8) --");
    for prompt in ["ROMEO:", "To be ", "King "] {
        let g = inference::generate_with_temperature(&model, &vocab, prompt, 200, 0.8, &mut rng);
        println!("\nprompt: {:?}\n{}", prompt, g);
    }
    println!("\n-- temperature sampling (T=1.0) --");
    for prompt in ["ROMEO:", "To be ", "King "] {
        let g = inference::generate_with_temperature(&model, &vocab, prompt, 200, 1.0, &mut rng);
        println!("\nprompt: {:?}\n{}", prompt, g);
    }

    let mut packed = Vec::new();
    export::export_ternary_packed(&model, &mut packed)
        .expect("packed export to Vec cannot fail");
    let path = models_path("shakespeare.ternary_packed.bin");
    let _ = std::fs::write(&path, &packed);
    println!(
        "\nwrote {} ({:.2} KB)",
        path.display(),
        packed.len() as f32 / 1024.0
    );
}

fn main() {
    // CLI dispatch:
    //   cargo run --release                                -- runs M4-M10 demo
    //   cargo run --release -- shakespeare                 -- fresh training
    //   cargo run --release -- shakespeare <resume_path>   -- resumes from file
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "shakespeare" {
        let resume_path = args
            .get(2)
            .map(|s| std::path::PathBuf::from(s))
            .filter(|p| p.exists());
        run_shakespeare_training(resume_path);
        return;
    }

    println!("── M4: 1D linear regression by SGD on  y = 2x ──");
    let (m4_initial, m4_final, m4_w) = train_linear_regression(200, 0.05);
    println!(
        "\nM4 result:  initial loss = {:.6}   final loss = {:.6e}   w = {:.6} (target 2.0)\n",
        m4_initial, m4_final, m4_w
    );

    println!("── M6: 2D regression through BitLinear (STE) on  y = 2x₁ − 2x₂ ──");
    let (m6_initial, m6_final, m6_w) = train_bitlinear_regression(500, 0.05);
    println!(
        "\nM6 result:  initial loss = {:.6}   final loss = {:.6e}   w = [{:.4}, {:.4}] (target [2, -2])\n",
        m6_initial, m6_final, m6_w[0], m6_w[1]
    );

    let (m9_initial, m9_min, m9_model, m9_vocab) = train_bitnet_lm(TrainConfig::tiny_demo());
    println!(
        "\nM9 result:  initial loss = {:.4}   min loss seen = {:.4}   ratio = {:.2}",
        m9_initial,
        m9_min,
        m9_min / m9_initial
    );

    // ── M10: greedy inference on the trained model. ──
    println!("\n── M10: greedy generation from the trained BitNet LM ──");
    let prompts = ["to be ", "the ", "or "];
    for prompt in &prompts {
        let generated = inference::generate(&m9_model, &m9_vocab, prompt, 40);
        // {:?} so we see literal newlines / whitespace clearly.
        println!("prompt {:>10?}  →  {:?}", prompt, generated);
    }

    // ── M10: ternary export - the on-disk payoff ──
    println!("\n── M10: ternary export size comparison ──");

    let mut f32_buf = Vec::new();
    let mut ternary_buf = Vec::new();
    let mut packed_buf = Vec::new();
    let f32_size =
        export::export_f32(&m9_model, &mut f32_buf).expect("f32 export to Vec cannot fail");
    let ternary_size = export::export_ternary(&m9_model, &mut ternary_buf)
        .expect("ternary export to Vec cannot fail");
    let packed_size = export::export_ternary_packed(&m9_model, &mut packed_buf)
        .expect("packed export to Vec cannot fail");

    let kb = |n: usize| n as f32 / 1024.0;
    println!(
        "f32 export:           {:>6} bytes ({:.2} KB)",
        f32_size,
        kb(f32_size)
    );
    println!(
        "ternary (i8/value):   {:>6} bytes ({:.2} KB)   {:.2}x vs f32",
        ternary_size,
        kb(ternary_size),
        f32_size as f32 / ternary_size as f32
    );
    println!(
        "ternary (5-per-byte): {:>6} bytes ({:.2} KB)   {:.2}x vs f32",
        packed_size,
        kb(packed_size),
        f32_size as f32 / packed_size as f32
    );

    // Write all three so the user can inspect with `ls -la models/`.
    let demo_f32 = models_path("demo.f32.bin");
    let demo_ternary = models_path("demo.ternary.bin");
    let demo_packed = models_path("demo.ternary_packed.bin");
    let _ = std::fs::write(&demo_f32, &f32_buf);
    let _ = std::fs::write(&demo_ternary, &ternary_buf);
    let _ = std::fs::write(&demo_packed, &packed_buf);
    println!(
        "wrote {}, {}, {}",
        demo_f32.display(),
        demo_ternary.display(),
        demo_packed.display()
    );

    // Round-trip sanity check: load the packed file back and run greedy
    // generation on the same prompts. Output won't match exactly (the master
    // weights lost f32 precision), but it should still produce text.
    println!(
        "\nround-trip check: loading {} and generating...",
        demo_packed.display()
    );
    let mut cursor = std::io::Cursor::new(packed_buf);
    match export::import(&mut cursor) {
        Ok((loaded_model, fmt)) => {
            println!(
                "loaded {:?}-format model with vocab={}",
                fmt, loaded_model.config.vocab_size
            );
            let g = inference::generate(&loaded_model, &m9_vocab, "to be ", 30);
            println!("loaded -> {:?}", g);
        }
        Err(e) => println!("import failed: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE M4 GATE.
    /// If autograd + SGD work, loss must collapse, and w must converge to ≈ 2.0.
    /// If this test fails, every previous "passing" autograd test is a lie -
    /// some chain-rule wiring is wrong in a way that doesn't show up in
    /// closed-form single-op tests but does show up under iterative gradient descent.
    #[test]
    fn training_drives_loss_down_and_w_to_target() {
        // 200 steps at lr=0.05 is overkill (theoretical decay rate per step ≈ 0.56,
        // so error is ~10⁻⁵⁰ at the end), which makes the test extremely robust.
        let (initial, finalv, w) = train_linear_regression(200, 0.05);

        // 1. Loss must collapse. We assert at least 100x reduction; convergence is
        //    actually many orders of magnitude better, so the slack here is intentional.
        assert!(
            finalv < initial * 0.01,
            "loss did not drop enough: initial = {}, final = {}",
            initial,
            finalv
        );

        // 2. w must land near the true value 2.0.
        //    1e-3 tolerance is generous; expect to be at f32 precision in practice.
        assert!(
            (w - 2.0).abs() < 1e-3,
            "w did not converge: got {}, expected ≈ 2.0",
            w
        );
    }

    /// THE M6 GATE.
    /// Train through STE-quantised weights and STE-quantised activations.
    /// If STE works, master_w converges to ≈ (2, −2) - exactly representable
    /// on the ternary grid (γ=2, W_q=[+1, −1]). If any quantiser's backward
    /// returned 0 instead of identity, master_w would stay at (0, 0) forever
    /// and this test would catch it the same way the M4 test would catch a
    /// broken plain matmul backward.
    #[test]
    fn bitlinear_training_drives_loss_down_through_ste() {
        let (initial, final_, w) = train_bitlinear_regression(500, 0.05);

        // Loss must collapse. Theoretical floor ≈ 6e-5 (irreducible due to
        // INT8 grid bias on rows 2/3); 100× reduction from initial=4.0 is plenty.
        assert!(
            final_ < initial * 0.01,
            "BitLinear loss did not drop enough: initial = {}, final = {}",
            initial,
            final_
        );

        // Master weights must converge to the ternary-grid target (±2). Tolerance
        // is generous - actual convergence is to ≈ (2.008, −2.008) per the
        // back-of-envelope analysis. Anything within 0.5 proves STE is doing its job.
        assert!(
            (w[0] - 2.0).abs() < 0.5,
            "w[0] did not converge: got {}, expected ≈ +2.0",
            w[0]
        );
        assert!(
            (w[1] + 2.0).abs() < 0.5,
            "w[1] did not converge: got {}, expected ≈ −2.0",
            w[1]
        );
    }

    /// THE M9 GATE.
    /// Train the full BitNet LM on TINY_CORPUS for 300 steps.  At least one
    /// step must achieve loss < 0.7 × initial - that's "the architecture is
    /// trainable end-to-end on real text" demonstrated.
    /// We use min-loss-seen rather than last-step loss because STE training
    /// has noisy per-step loss (see model.rs `training_steps_reduce_loss_...`).
    #[test]
    fn bitnet_lm_training_drives_loss_down_substantially() {
        // Quieter config than the demo so the test runs faster.
        let mut cfg = TrainConfig::tiny_demo();
        cfg.n_steps = 200;
        cfg.log_every = usize::MAX; // suppress per-step prints inside the test
        let (initial, min_loss, _model, _vocab) = train_bitnet_lm(cfg);
        assert!(
            min_loss < initial * 0.7,
            "BitNet LM did not improve enough: initial = {}, min seen = {}, ratio = {:.3}",
            initial,
            min_loss,
            min_loss / initial
        );
    }
}
