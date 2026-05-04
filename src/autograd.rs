//! Tape-based reverse-node autograd
//!
//! Public types:
//! - `Tape`        - owns every recorded forward value and every gradient cell.
//! - `Var<'t>`     - `Copy` handle into a Tape: `{ tape: &Tape, id: usize }`,
//! - `NodeId`      - internal index into the tape's node list.
//!
//! Lifetime model:
//! - All `Var`'s share the lifetime of the `Tape` they reference.
//! - Build one `Tape` per training step (or per evaluation), then drop it.
//!   That keeps memory bounded - old tapes don't pile up.
//!
//! Why a tape (vs. embedding grad fields in `Tensor`:
//! - `Tensor` (M1) stays a pure value type with zero autograde overhead.
//! - Ownership of every recorded value lives in one place - no `Rc<RefCell<...>>`
//! on the public API.
//! - Mirrors how PyTorch internally records its autograd graph.

use crate::bitlinear::{absmax_int8, absmean_ternary};
use crate::tensor::Tensor;
use std::cell::RefCell;

/// Stable index into a `Tape`'s `nodes` Vec. Cheap, `Copy`, no lifetime.
pub type NodeId = usize;

/// One recorded operation result. Created by every forward call that goes through `Var`.
struct Node {
    /// The forward value this op produced. Cloned from caller; tape owns this copy.
    value: Tensor,

    /// Gradient accumulator. Same shape as `value`. Initialised to zeros at creation;
    /// `backward()` sums contributions into it. Inner `RefCell` so we can mutate one
    /// node's grad while only holding an *immutable* borrow on the outer node list.
    grad: RefCell<Tensor>,

    /// How to compute parent gradients given this node's incoming gradient.
    /// `None` for leaf nodes (created from raw tensors with no producing op).
    /// The closure is `Fn` (not `FnMut`/`FnOnce`) so a single backward pass can
    /// invoke it once; if we ever want higher-order gradients we'd revisit this.
    backward: Option<Box<dyn Fn(&Tensor) -> Vec<(NodeId, Tensor)>>>,

    /// Parent node IDs in op-input order. Currently only kept for debugging /
    /// future graph-walk strategies; the `backward closure already knows them.
    #[allow(dead_code)]
    parents: Vec<NodeId>,
}

/// Owns every node on the autograd graph for one forward+backward cycle.
/// Construct, run forward (which appends nodes), call `backward()`, read grads, drop.
pub struct Tape {
    /// Single `RefCell` around the whole node list. We never hold a `borrow_mut()`
    /// across a closure call - see `backward` for the access pattern.
    nodes: RefCell<Vec<Node>>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape {
            nodes: RefCell::new(Vec::new()),
        }
    }

    /// Append a node and return its fresh id. Used by every op (including leaf).
    fn push(&self, node: Node) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(node);
        id
    }

    /// Read-only clone of a node's forward value. Cloning is fine - Tensors here
    /// are tiny (Vec<f32> + Vec<usize>). If this becomes hot, swap to a borrowing
    /// accessor; for a toy training loop the clones don't matter.
    pub fn value(&self, id: NodeId) -> Tensor {
        self.nodes.borrow()[id].value.clone()
    }

    /// Read-only clone of a node's accumulated gradient. Mostly for inspection
    /// in tests and the optimiser step.
    pub fn grad(&self, id: NodeId) -> Tensor {
        self.nodes.borrow()[id].grad.borrow().clone()
    }

    /// Number of recorded nodes. Test-only sanity helper.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Replace a node's gradient cell. Used by `optim::clip_grad_norm` to
    /// rescale leaf gradients in place. Shape of `new_grad` must match.
    pub fn write_grad(&self, id: NodeId, new_grad: Tensor) {
        let nodes = self.nodes.borrow();
        let cell = &nodes[id].grad;
        assert_eq!(
            cell.borrow().shape,
            new_grad.shape,
            "write_grad: shape mismatch at node {}",
            id
        );
        *cell.borrow_mut() = new_grad;
    }

    /// Reverse-mode backward pass starting from `output_id`.
    ///
    /// Seeds the output's gradient with `ones` (same shape as the output value),
    /// then walks all recorded nodes in reverse-creation order - which is a valid
    /// topological order on a tape - invoking each non-leaf node's backward
    /// closure and accumulating its returned contributions onto parent grad cells.
    ///
    /// After this call, each lead's `grad()` holds the full gradient `∂L/∂leaf`,
    /// where `L = sum(output_value)` if the output isn't already scalar.
    pub fn backward(&self, output_id: NodeId) {
        // ── 1. Seed ∂output/∂output = ones. ───────────────────────────────
        {
            let nodes = self.nodes.borrow();
            let out_shape = nodes[output_id].value.shape.clone();
            // Overwrite the existing zero accumulator. We assign through the inner
            // RefCell, which is the only mutable cell we touch in this scope.
            *nodes[output_id].grad.borrow_mut() = Tensor::ones(out_shape);
        }

        // ── 2. Walk in reverse-creation order. ────────────────────────────
        // We snapshot `len` once; the walk does not push new nodes (closures are
        // puse w.r.t. the tape), so the length is stable.
        let n = self.nodes.borrow().len();
        for id in (0..n).rev() {
            // Skip leaf nodes (no producing op -> nothing to propagate through).
            // We check via a short, scoped immutable borrow so the borrow doesn't
            // outlive this branch decision.
            let has_backward = self.nodes.borrow()[id].backward.is_some();
            if !has_backward {
                continue;
            }

            // Snapshot this node's accumulated incoming gradient. Cloning lets us
            // drop the borrow before invoking the closure - keeps the access
            // pattern simple and protects us if a future closure ever wants to
            // read tape state (current closures don't, but future-proofing is affordable).
            let grad_snapshot = self.nodes.borrow()[id].grad.borrow().clone();

            // Invoke the backward closure to get per-parent contributions.
            // The Box<dyn Fn ...> is owned by the node; we just borrow it.
            let contributions = {
                let nodes = self.nodes.borrow();
                let bw = nodes[id]
                    .backward
                    .as_ref()
                    .expect("checked has_backward above");
                bw(&grad_snapshot)
            };

            // Accumulate contributions into parents' grad cells.
            // Outer immutable borrow on `nodes` is fine because the only mutation
            // is on a *different* RefCell (the parent's per-node grad cell), which
            // doesn't conflict with the outer-Vec immutable borrow.
            let nodes = self.nodes.borrow();
            for (parent_id, contrib) in contributions {
                let mut g = nodes[parent_id].grad.borrow_mut();
                // Sum-in: g_new = g_old + contrib. The temporary owned Tensor
                // returned by `add` is moved into the cell via `*g = ...`.
                *g = g.add(&contrib);
            }
        }
    }
}

/// Cheap `Copy` handle into a Tape. Operations on `Var`'s record new nodes.
#[derive(Copy, Clone)]
pub struct Var<'t> {
    pub tape: &'t Tape,
    pub id: NodeId,
}

impl<'t> Var<'t> {
    /// Register an existing `Tensor` as a leaf on the tape (no producing op,
    /// no parents). This is how weights, biases, and inputs enter the graph.
    pub fn leaf(tape: &'t Tape, value: Tensor) -> Self {
        let grad = Tensor::zeros(value.shape.clone()); // Same-shape zero accumulator
        let id = tape.push(Node {
            value,
            grad: RefCell::new(grad),
            backward: None,
            parents: Vec::new(),
        });
        Self { tape, id }
    }

    /// Read the current accumulated gradient. Zeros until `Tape::backward` runs.
    pub fn grad(&self) -> Tensor {
        self.tape.grad(self.id)
    }

    /// Read the forward value at this node.
    pub fn value(&self) -> Tensor {
        self.tape.value(self.id)
    }
}

// ── Operations ──────────────────────────────────────────────────────────────
//
// Each op:
//   1. Computes the forward value (delegating to Tensor's pure ops).
//   2. Captures whatever the backward closure will need (parent ids, sometimes
//      cloned input values for ops wwhose gradient depends on them).
//   3. Builds a backward closure of type `Fn(&Tensor) -> Vec<(NodeId, Tensor)>`:
//      Given this node's incoming gradient, return per-parent contributions.
//   4. Pushes a fresh Node onto the tape and returns a Var handle.
impl<'t> Var<'t> {
    /// Elementwise add: `c = a + b`. Both operands must live on the same Tape.
    ///
    /// Backward (identity passthrough):
    ///   `c = a + b`   ->  ∂c/∂a = 1,  ∂c/∂b = 1
    /// so the incoming gradient `∂L/∂c` flows unchanged into both inputs.
    /// We clone it once per parent because each parent gets its own owned tensor -
    /// ownership-wise we can't hand the same Tensor to two recipients.
    pub fn add(self, other: Var<'t>) -> Var<'t> {
        // `ptr::eq`  compares the &Tape addresses - exactly the identity check we need.
        // Two Vars on different tapes would build disconnected graphs and silently
        // produce wrong gradients; asset loudly instead.
        assert!(
            std::ptr::eq(self.tape, other.tape),
            "Var::add: operands belong to different tapes"
        );

        // Forward - Tensor::add already enforces shape equality.
        let v = self.value().add(&other.value());

        // Capture parent ids by value (NodeId is Copy). The closure becomes 'static,
        // which is what `Box<dyn Fn ...>` requires.
        let p0 = self.id;
        let p1 = other.id;
        let backward = Box::new(move |grad_out: &Tensor| -> Vec<(NodeId, Tensor)> {
            vec![(p0, grad_out.clone()), (p1, grad_out.clone())]
        });

        let grad_zero = Tensor::zeros(v.shape.clone());
        let id = self.tape.push(Node {
            value: v,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0, p1],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// 2D matrix multiply: `Y = X · W`.  X is `[m, k]`, W is `[k, n]`, Y is `[m, n]`.
    /// Both operands must live on the same tape.
    ///
    /// Backward - the foundational matmul gradient identity:
    ///     ∂L/∂X = ∂L/∂Y · Wᵀ          shape [m, n] · [n, k] = [m, k]
    ///     ∂L/∂W = Xᵀ · ∂L/∂Y          shape [k, m] · [m, n] = [k, n]
    ///
    /// We capture **clones** of X's and W's forward values into the closure; the
    /// backward computation needs both. This is the autograd "saved tensors"
    /// memory cost - one extra Tensor's worth per matmul kept on the tape.
    /// For a toy model on tiny inputs this is negligible.
    pub fn matmul(self, other: Var<'t>) -> Var<'t> {
        assert!(
            std::ptr::eq(self.tape, other.tape),
            "Var::matmul: operands belong to different tapes"
        );

        // Forward - Tensor::matmul checks rank-2 and inner-dim agreement.
        let x_val = self.value();
        let w_val = other.value();
        let y_val = x_val.matmul(&w_val);

        // Capture by value (move). Closure becomes 'static, satisfying Box<dyn Fn ...>.
        let p0 = self.id;
        let p1 = other.id;
        let x_saved = x_val; // saved for backward
        let w_saved = w_val;
        let backward = Box::new(move |grad_y: &Tensor| -> Vec<(NodeId, Tensor)> {
            // Each gradient is itself a matmul; transposes are computed on the fly.
            // For very large W this is wasted work (the transpose is O(N) per step) -
            // a real framework caches Wᵀ or uses strided views. Toy model: don't care.
            let grad_x = grad_y.matmul(&w_saved.transpose_2d());
            let grad_w = x_saved.transpose_2d().matmul(grad_y);
            vec![(p0, grad_x), (p1, grad_w)]
        });

        let grad_zero = Tensor::zeros(y_val.shape.clone());
        let id = self.tape.push(Node {
            value: y_val,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0, p1],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Elementwise subtract: `c = a - b`.
    /// Backward: ∂c/∂a = +1 (passthrough), ∂c/∂b = −1 (negate).
    pub fn sub(self, other: Var<'t>) -> Var<'t> {
        assert!(
            std::ptr::eq(self.tape, other.tape),
            "Var::sub: operands belong to different tapes"
        );
        let v = self.value().sub(&other.value());

        let p0 = self.id;
        let p1 = other.id;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            // Negate via mul_scalar so we don't have to define an in-place negate.
            let neg = g.mul_scalar(-1.0);
            vec![(p0, g.clone()), (p1, neg)]
        });

        let grad_zero = Tensor::zeros(v.shape.clone());
        let id = self.tape.push(Node {
            value: v,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0, p1],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Elementwise (Hadamard) multiply: `c = a * b`.
    /// Backward needs the OTHER operand to compute each input's gradient:
    ///     ∂c/∂a = b   →   grad_a = grad_out ⊙ b
    ///     ∂c/∂b = a   →   grad_b = grad_out ⊙ a
    /// We therefore save clones of both forward values into the closure.
    /// Note: `a.mul(a)` (square via reused leaf) works automatically - the
    /// graph correctly registers two contributions into the same parent and
    /// the existing accumulation logic sums them, giving the expected ∂(a²)/∂a = 2a.
    pub fn mul(self, other: Var<'t>) -> Var<'t> {
        assert!(
            std::ptr::eq(self.tape, other.tape),
            "Var::mul: operands belong to different tapes"
        );
        let a = self.value();
        let b = other.value();
        let v = a.mul(&b);

        let p0 = self.id;
        let p1 = other.id;
        let a_saved = a;
        let b_saved = b;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            vec![(p0, g.mul(&b_saved)), (p1, g.mul(&a_saved))]
        });

        let grad_zero = Tensor::zeros(v.shape.clone());
        let id = self.tape.push(Node {
            value: v,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0, p1],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Mean over all elements. Output is a scalar tensor of shape `[1]`.
    /// Backward: ∂mean(x)/∂xᵢ = 1/n for every element. So given grad-out (one
    /// scalar), we broadcast `g · (1/n)` across the input's shape.
    /// This is also the natural "loss reducer" for training: a scalar loss is
    /// what `tape.backward` is designed to seed cleanly.
    pub fn mean(self) -> Var<'t> {
        let v_in = self.value();
        let n = v_in.data.len() as f32;
        // Forward: sum-then-divide. f32 accumulation is fine for our toy sizes;
        // a real impl would use Kahan or pairwise summation for numeric stability.
        let total: f32 = v_in.data.iter().sum();
        let v = Tensor::from_vec(vec![total / n], vec![1]);

        let p0 = self.id;
        let in_shape = v_in.shape.clone(); // captured for the backward broadcast
        let inv_n = 1.0_f32 / n;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            // g has shape [1]; spread (g · 1/n) across every input element.
            let scalar = g.data[0] * inv_n;
            let n_elem: usize = in_shape.iter().product();
            let grad_in = Tensor::from_vec(vec![scalar; n_elem], in_shape.clone());
            vec![(p0, grad_in)]
        });

        let grad_zero = Tensor::zeros(v.shape.clone());
        let id = self.tape.push(Node {
            value: v,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Apply absmean-ternary weight quantisation in forward, identity in backward.
    /// Output is the **dequantised** weight `y · W_q`, so it can be matmul'd directly.
    /// Gradient flows through unchanged - that's the STE lie. The math truth (∂/∂W = 0
    /// almost everywhere because of round + clamp) would freeze training, so we
    /// ignore it and treat the whole quant-then-rescale composition as identity.
    pub fn quantise_weights_ste(self) -> Var<'t> {
        // ── Forward: quant + dequant. ──
        let w_val = self.value();
        let (w_q, gamma) = absmean_ternary(&w_val);
        // Build γ · W_q without touching tensor.rs's API surface.
        let w_eff_data: Vec<f32> = w_q.data.iter().map(|v| v * gamma).collect();
        let w_eff = Tensor {
            data: w_eff_data,
            shape: w_q.shape.clone(),
        };

        // ── Backward: identity passthrough (the STE). ──
        let p0 = self.id;
        let backward =
            Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> { vec![(p0, g.clone())] });

        let grad_zero = Tensor::zeros(w_eff.shape.clone());
        let id = self.tape.push(Node {
            value: w_eff,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![self.id],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Apply absmax_INT8 per-row activation quantisation in forward, identity in backward.
    /// Output is the dequantised `(α[i] / 127) · x_q[i, j]`. Same STE rationale as above.
    pub fn quantise_acts_ste(self) -> Var<'t> {
        // ── Forward: quant + per-row dequant. ──
        let x_val = self.value();
        let (x_q, alpha) = absmax_int8(&x_val);
        let (m, n) = (x_q.shape[0], x_q.shape[1]);
        let mut x_eff_data = vec![0.0_f32; m * n];
        let inv_127 = 1.0_f32 / 127.0;
        for i in 0..m {
            let row_scale = alpha.data[i] * inv_127; // = α[i] / 127
            for j in 0..n {
                x_eff_data[i * n + j] = x_q.data[i * n + j] * row_scale;
            }
        }
        let x_eff = Tensor {
            data: x_eff_data,
            shape: vec![m, n],
        };

        // ── Backward: identity. ──
        let p0 = self.id;
        let backward =
            Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> { vec![(p0, g.clone())] });

        let grad_zero = Tensor::zeros(x_eff.shape.clone());
        let id = self.tape.push(Node {
            value: x_eff,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// 2D transpose. NOT an STE - this is real math: ∂(Mᵀ)/∂M is itself a transpose,
    /// so the backward closure transposes the incoming gradient and routes it back.
    /// Needed by BitLinear's autograd path /we need Wᵀ on-tape).
    pub fn transpose_2d(self) -> Var<'t> {
        let v_in = self.value();
        let v_out = v_in.transpose_2d();

        let p0 = self.id;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            // ∂L/∂M = (∂L/∂Mᵀ)ᵀ.
            vec![(p0, g.transpose_2d())]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// ReLU activation: forward `max(0, x)`, backward passes gradient through
    /// the positive-x mask. Captures the input value to know which positions
    /// were active. The convention `x ≤ 0 → 0` (strict positivity for grad)
    /// means the boundary `x = 0` gets gradient 0 - same as PyTorch's default.
    pub fn relu(self) -> Var<'t> {
        let v_in = self.value();
        let v_out_data: Vec<f32> = v_in.data.iter().map(|&x| x.max(0.0)).collect();
        let v_out = Tensor {
            data: v_out_data,
            shape: v_in.shape.clone(),
        };

        let p0 = self.id;
        let x_saved = v_in; // need the input mask for backward
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let grad_in_data: Vec<f32> = g
                .data
                .iter()
                .zip(&x_saved.data)
                .map(|(&grad, &x)| if x > 0.0 { grad } else { 0.0 })
                .collect();
            vec![(
                p0,
                Tensor {
                    data: grad_in_data,
                    shape: x_saved.shape.clone(),
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// SiLU (Sigmoid Linear Unit), also known as Swish:
    ///     silu(x) = x · σ(x)         where σ(x) = 1 / (1 + exp(-x))
    ///
    /// Smooth alternative to ReLU with a small "leaky" gradient on the
    /// negative side - non-zero everywhere, so the dead-neuron problem
    /// of ReLU does not apply. SiLU is the activation in the "Swi" of
    /// SwiGLU, the gated FFN form used by LLaMA / BitNet b1.58.
    ///
    /// Backward via chain rule:
    ///     d/dx[x · σ(x)] = σ(x) + x · σ'(x)
    ///                    = σ(x) + x · σ(x) · (1 − σ(x))
    ///                    = σ(x) · (1 + x · (1 − σ(x)))
    ///
    /// We save the input `x` (not σ(x) or the output) so the backward closure
    /// can recompute σ once and derive everything else from it. Saving σ would
    /// cost the same memory and only avoid one exp per element; not worth a
    /// second tensor allocation.
    pub fn silu(self) -> Var<'t> {
        let v_in = self.value();
        let v_out_data: Vec<f32> = v_in
            .data
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();
        let v_out = Tensor {
            data: v_out_data,
            shape: v_in.shape.clone(),
        };

        let p0 = self.id;
        let x_saved = v_in;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let grad_in_data: Vec<f32> = g
                .data
                .iter()
                .zip(&x_saved.data)
                .map(|(&grad, &x)| {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    // d/dx[silu] = σ · (1 + x · (1 − σ))
                    let dsilu = sig * (1.0 + x * (1.0 - sig));
                    grad * dsilu
                })
                .collect();
            vec![(
                p0,
                Tensor {
                    data: grad_in_data,
                    shape: x_saved.shape.clone(),
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Per-row softmax over the last axis. Input must be 2D `[m, n]`.
    /// Forward uses the subtract-max trick: `exp(x − maxₖ x)` keeps every
    /// exponent ≤ 0 so `exp` never overflows. The post-normalisation result is
    /// invariant to that subtraction.
    ///
    /// Backward (Jacobian-vector product, derived from softmax's full Jacobian):
    ///     ∂L/∂x[i, j] = s[i, j] · (g[i, j] − Σₖ g[i, k] · s[i, k])
    /// We save the OUTPUT `s` (not the input) - the formula is purely in terms
    /// of `s`, and saving output costs the same memory as saving input.
    pub fn softmax(self) -> Var<'t> {
        let x = self.value();
        assert_eq!(
            x.ndim(),
            2,
            "softmax: expected rank-2, got rank {}",
            x.ndim()
        );
        let (m, n) = (x.shape[0], x.shape[1]);

        let mut s = vec![0.0_f32; m * n];
        for i in 0..m {
            // Pass 1: row max for numerical stability.
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..n {
                let v = x.data[i * n + j];
                if v > row_max {
                    row_max = v;
                }
            }
            // Pass 2: exp(x − max), accumulate denominator.
            let mut denom = 0.0_f32;
            for j in 0..n {
                let e = (x.data[i * n + j] - row_max).exp();
                s[i * n + j] = e;
                denom += e;
            }
            // Pass 3: normalise. (1/denom factored out so the inner loop is just
            // a multiply; division is the most expensive scalar op on most CPUs.)
            let inv = 1.0_f32 / denom;
            for j in 0..n {
                s[i * n + j] *= inv;
            }
        }
        let v_out = Tensor {
            data: s,
            shape: vec![m, n],
        };

        let p0 = self.id;
        let s_saved = v_out.clone();
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let mut grad_in = vec![0.0_f32; m * n];
            for i in 0..m {
                // dotᵢ = Σₖ g[i, k] · s[i, k] - one scalar per row.
                let mut dot = 0.0_f32;
                for k in 0..n {
                    dot += g.data[i * n + k] * s_saved.data[i * n + k];
                }
                // grad_in[i, j] = s[i, j] · (g[i, j] − dotᵢ)
                for j in 0..n {
                    grad_in[i * n + j] = s_saved.data[i * n + j] * (g.data[i * n + j] - dot);
                }
            }
            vec![(
                p0,
                Tensor {
                    data: grad_in,
                    shape: vec![m, n],
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Per-row RMS normalisation (no learnable gain).  Input must be 2D `[m, n]`.
    /// Each row gets divided by its RMS magnitude:
    ///     rmsᵢ = √(meanⱼ(x[i, j]²) + ε),   y[i, j] = x[i, j] / rmsᵢ
    ///
    /// Backward (per-row, derived from quotient rule + chain rule on rms):
    ///     ∂L/∂x[i, k] = g[i, k] / rmsᵢ − x[i, k] · dotᵢ / (n · rmsᵢ³)
    ///   where  dotᵢ = Σⱼ x[i, j] · g[i, j].
    pub fn rmsnorm(self) -> Var<'t> {
        let x = self.value();
        assert_eq!(
            x.ndim(),
            2,
            "rmsnorm: expected rank-2, got rank {}",
            x.ndim()
        );
        let (m, n) = (x.shape[0], x.shape[1]);
        let n_f = n as f32;
        const EPS: f32 = 1e-5;

        let mut y = vec![0.0_f32; m * n];
        let mut rms_per_row = vec![0.0_f32; m]; // saved for backward
        for i in 0..m {
            let mean_sq: f32 = (0..n).map(|j| x.data[i * n + j].powi(2)).sum::<f32>() / n_f;
            let rms = (mean_sq + EPS).sqrt();
            rms_per_row[i] = rms;
            let inv = 1.0_f32 / rms;
            for j in 0..n {
                y[i * n + j] = x.data[i * n + j] * inv;
            }
        }
        let v_out = Tensor {
            data: y,
            shape: vec![m, n],
        };

        let p0 = self.id;
        let x_saved = x; // need x for the dot term
        let rms_saved = rms_per_row; // owned Vec<f32>, captured by move
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let mut grad_in = vec![0.0_f32; m * n];
            for i in 0..m {
                let inv_rms = 1.0_f32 / rms_saved[i];
                // dotᵢ = Σⱼ x[i, j] · g[i, j]
                let mut dot = 0.0_f32;
                for j in 0..n {
                    dot += x_saved.data[i * n + j] * g.data[i * n + j];
                }
                // factor = dot · (1/rms)³ / n   - pre-computed once per row.
                let factor = dot * inv_rms.powi(3) / n_f;
                for j in 0..n {
                    grad_in[i * n + j] =
                        g.data[i * n + j] * inv_rms - x_saved.data[i * n + j] * factor;
                }
            }
            vec![(
                p0,
                Tensor {
                    data: grad_in,
                    shape: vec![m, n],
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Causal mask for attention scores. Input must be 2D [seq, seq].
    /// Forward: y[i, j] = x[i, j] if j <= i, else -inf. After softmax,
    /// positions with -inf score get zero probability, so attention at row i
    /// only attends to columns <= i (past + current).
    /// Backward: gradient passes through the lower triangle (where the value
    /// was preserved) and is zero on the upper triangle (where we overwrote
    /// the original value with -inf).
    pub fn causal_mask(self) -> Var<'t> {
        let v_in = self.value();
        assert_eq!(
            v_in.ndim(),
            2,
            "causal_mask: expected rank-2 [seq, seq], got rank {}",
            v_in.ndim()
        );
        let (m, n) = (v_in.shape[0], v_in.shape[1]);

        let mut v_out_data = v_in.data.clone();
        for i in 0..m {
            for j in 0..n {
                if j > i {
                    v_out_data[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }
        let v_out = Tensor {
            data: v_out_data,
            shape: v_in.shape.clone(),
        };

        let p0 = self.id;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let mut grad_in = vec![0.0_f32; m * n];
            for i in 0..m {
                for j in 0..(i + 1).min(n) {
                    grad_in[i * n + j] = g.data[i * n + j];
                }
            }
            vec![(
                p0,
                Tensor {
                    data: grad_in,
                    shape: vec![m, n],
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Rotary Position Embedding (RoPE). Input is rank-2 `[seq, head_dim]`;
    /// output has the same shape. For each row `pos in 0..seq` and each pair
    /// index `i in 0..head_dim/2`, rotates the 2D vector
    /// `(x[pos, 2i], x[pos, 2i+1])` by angle
    ///
    ///     theta_{pos, i} = pos * base ^ (-2i / head_dim)        # base = 10000
    ///
    /// concretely
    ///
    ///     y[pos, 2i]   = x[pos, 2i] * cos(t) - x[pos, 2i+1] * sin(t)
    ///     y[pos, 2i+1] = x[pos, 2i] * sin(t) + x[pos, 2i+1] * cos(t)
    ///
    /// Standard in LLaMA / BitNet b1.58. Replaces learned absolute position
    /// embeddings: position information is injected at attention time via
    /// rotated Q and K, so attention scores depend on the *difference*
    /// between query and key positions naturally.
    ///
    /// `head_dim` must be even. RoPE has no learned parameters; the rotation
    /// angles are fixed by `(pos, pair_index)`.
    ///
    /// Backward: each pair's 2D rotation is orthogonal, so its inverse is its
    /// transpose, which is rotation by the negated angle. We capture the
    /// per-position cos / sin tables in the closure to avoid recomputing the
    /// transcendentals during the backward pass.
    pub fn rope(self) -> Var<'t> {
        let x_in = self.value();
        assert_eq!(
            x_in.ndim(),
            2,
            "rope: expected rank-2 [seq, head_dim], got rank {}",
            x_in.ndim()
        );
        let (seq, head_dim) = (x_in.shape[0], x_in.shape[1]);
        assert!(
            head_dim % 2 == 0,
            "rope: head_dim ({}) must be even for pair-wise rotation",
            head_dim
        );
        let half = head_dim / 2;

        // Precompute cos / sin per (pos, pair). The same tables get used by
        // both the forward result here and the backward closure below.
        let mut cos_tab = vec![0.0_f32; seq * half];
        let mut sin_tab = vec![0.0_f32; seq * half];
        for pos in 0..seq {
            for i in 0..half {
                let theta_i = 10000_f32.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * theta_i;
                cos_tab[pos * half + i] = angle.cos();
                sin_tab[pos * half + i] = angle.sin();
            }
        }

        let mut y_data = vec![0.0_f32; seq * head_dim];
        for pos in 0..seq {
            for i in 0..half {
                let a = x_in.data[pos * head_dim + 2 * i];
                let b = x_in.data[pos * head_dim + 2 * i + 1];
                let c = cos_tab[pos * half + i];
                let s = sin_tab[pos * half + i];
                y_data[pos * head_dim + 2 * i] = a * c - b * s;
                y_data[pos * head_dim + 2 * i + 1] = a * s + b * c;
            }
        }
        let y_val = Tensor {
            data: y_data,
            shape: vec![seq, head_dim],
        };

        let p0 = self.id;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            // Inverse rotation: same trig table, sign of sin flipped.
            //     dL/dx[pos, 2i]   =  dL/dy[pos, 2i]   * cos + dL/dy[pos, 2i+1] * sin
            //     dL/dx[pos, 2i+1] = -dL/dy[pos, 2i]   * sin + dL/dy[pos, 2i+1] * cos
            let mut grad_in = vec![0.0_f32; seq * head_dim];
            for pos in 0..seq {
                for i in 0..half {
                    let ga = g.data[pos * head_dim + 2 * i];
                    let gb = g.data[pos * head_dim + 2 * i + 1];
                    let c = cos_tab[pos * half + i];
                    let s = sin_tab[pos * half + i];
                    grad_in[pos * head_dim + 2 * i] = ga * c + gb * s;
                    grad_in[pos * head_dim + 2 * i + 1] = -ga * s + gb * c;
                }
            }
            vec![(
                p0,
                Tensor {
                    data: grad_in,
                    shape: vec![seq, head_dim],
                },
            )]
        });

        let grad_zero = Tensor::zeros(y_val.shape.clone());
        let id = self.tape.push(Node {
            value: y_val,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Multiply every element by a constant `s`. The constant is captured by
    /// value (Copy), so the closure stays `Fn` and 'static.
    /// Backward:  d(s·x)/dx = s, so grad_in = grad_out · s.
    pub fn mul_scalar(self, s: f32) -> Var<'t> {
        let v = self.value().mul_scalar(s);
        let p0 = self.id;
        let backward =
            Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> { vec![(p0, g.mul_scalar(s))] });
        let grad_zero = Tensor::zeros(v.shape.clone());
        let id = self.tape.push(Node {
            value: v,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Embedding lookup. `self` is the embedding weight matrix `[vocab, hidden]`,
    /// `ids` are the token positions to look up. Output is `[ids.len(), hidden]`,
    /// where row `i` is `weights[ids[i], :]`.
    ///
    /// Backward: gradient on `weights[id, :]` accumulates `grad_out[i, :]` for
    /// EVERY position `i` where `ids[i] == id`. Duplicate ids stack - the same
    /// accumulation pattern as a reused leaf, just localised inside this op.
    pub fn embed(self, ids: &[usize]) -> Var<'t> {
        let w = self.value();
        assert_eq!(w.ndim(), 2, "embed: weights must be rank-2 [vocab, hidden]");
        let (vocab, hidden) = (w.shape[0], w.shape[1]);
        let seq = ids.len();

        // Forward: row-pick. Bound-check happens on the indexing path; an early
        // explicit assertion gives a friendlier failure message than an indexing panic.
        for &id in ids {
            assert!(id < vocab, "embed: id {} >= vocab {}", id, vocab);
        }
        let mut out_data = Vec::with_capacity(seq * hidden);
        for &id in ids {
            for j in 0..hidden {
                out_data.push(w.data[id * hidden + j]);
            }
        }
        let v_out = Tensor {
            data: out_data,
            shape: vec![seq, hidden],
        };

        // Backward: scatter-add grad_out[i, :] into grad_w[ids[i], :].
        let p0 = self.id;
        let ids_saved: Vec<usize> = ids.to_vec(); // owned capture for 'static closure
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let mut grad_w = vec![0.0_f32; vocab * hidden];
            for (i, &id) in ids_saved.iter().enumerate() {
                for j in 0..hidden {
                    // += because the same `id` may appear multiple times in `ids_saved`,
                    // and each occurrence contributes a row of grad_out.
                    grad_w[id * hidden + j] += g.data[i * hidden + j];
                }
            }
            vec![(
                p0,
                Tensor {
                    data: grad_w,
                    shape: vec![vocab, hidden],
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }

    /// Combined softmax + cross-entropy loss. `self` is the logits tensor
    /// `[seq, vocab]`; `targets` is a `[seq]` slice of class indices.
    /// Output is a scalar tensor of shape `[1]`.
    ///
    /// Forward uses log-sum-exp for numerical stability:
    ///     log(Σⱼ exp(xⱼ)) = max(x) + log(Σⱼ exp(xⱼ − max(x)))
    /// keeps every exponent ≤ 0 so `exp` never overflows.
    ///
    /// Backward - the closed-form gradient that classification training in
    /// every framework on Earth uses:
    ///     grad_logits[i, j] = (softmax(logits)[i, j] − onehot(target[i])[j]) / seq
    /// We save `softmax(logits)` from the forward (computed once) and apply
    /// the −1 at the target index per row in the backward closure.
    pub fn cross_entropy(self, targets: &[usize]) -> Var<'t> {
        let logits = self.value();
        assert_eq!(
            logits.ndim(),
            2,
            "cross_entropy: logits must be rank-2 [seq, vocab]"
        );
        let (seq, vocab) = (logits.shape[0], logits.shape[1]);
        assert_eq!(
            targets.len(),
            seq,
            "cross_entropy: targets len {} must match seq {}",
            targets.len(),
            seq
        );
        for &t in targets {
            assert!(t < vocab, "cross_entropy: target {} >= vocab {}", t, vocab);
        }

        // Forward + saved softmax in one pass.
        let mut softmax = vec![0.0_f32; seq * vocab];
        let mut total_loss = 0.0_f32;
        for i in 0..seq {
            // Subtract-max for stability.
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..vocab {
                let v = logits.data[i * vocab + j];
                if v > row_max {
                    row_max = v;
                }
            }
            // exp(x − max), accumulate denom.
            let mut denom = 0.0_f32;
            for j in 0..vocab {
                let e = (logits.data[i * vocab + j] - row_max).exp();
                softmax[i * vocab + j] = e;
                denom += e;
            }
            // log_denom = max + log(Σ exp(x − max)) - the log-sum-exp trick.
            let log_denom = row_max + denom.ln();
            // Normalise softmax row in-place for use in backward.
            let inv = 1.0_f32 / denom;
            for j in 0..vocab {
                softmax[i * vocab + j] *= inv;
            }
            // loss_i = −log(softmax[target]) = −(logits[target] − log_denom)
            let t = targets[i];
            total_loss += -(logits.data[i * vocab + t] - log_denom);
        }
        let loss = total_loss / seq as f32;
        let v_out = Tensor {
            data: vec![loss],
            shape: vec![1],
        };

        // Backward: grad_logits[i, j] = (softmax[i, j] − δ_{j, target[i]}) · g/seq
        let p0 = self.id;
        let targets_saved: Vec<usize> = targets.to_vec();
        let inv_seq = 1.0_f32 / seq as f32;
        let backward = Box::new(move |g: &Tensor| -> Vec<(NodeId, Tensor)> {
            let g_scalar = g.data[0] * inv_seq;
            // Start from the saved softmax; subtract 1 at each target index.
            let mut grad = softmax.clone();
            for i in 0..seq {
                grad[i * vocab + targets_saved[i]] -= 1.0;
            }
            for v in grad.iter_mut() {
                *v *= g_scalar;
            }
            vec![(
                p0,
                Tensor {
                    data: grad,
                    shape: vec![seq, vocab],
                },
            )]
        });

        let grad_zero = Tensor::zeros(v_out.shape.clone());
        let id = self.tape.push(Node {
            value: v_out,
            grad: RefCell::new(grad_zero),
            backward: Some(backward),
            parents: vec![p0],
        });
        Var {
            tape: self.tape,
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_records_value_and_zero_grad() {
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]));

        assert_eq!(tape.len(), 1);
        assert_eq!(x.value().data, vec![1.0, 2.0, 3.0]);
        assert_eq!(x.grad().data, vec![0.0, 0.0, 0.0]);
        assert_eq!(x.grad().shape, vec![3]);
    }

    #[test]
    fn multiple_leaves_get_distinct_ids() {
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::zeros(vec![2]));
        let b = Var::leaf(&tape, Tensor::ones(vec![2]));
        assert_ne!(a.id, b.id);
        assert_eq!(tape.len(), 2);
        assert_eq!(b.value().data, vec![1.0, 1.0]);
    }

    #[test]
    fn add_records_correct_forward_value() {
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let b = Var::leaf(&tape, Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]));
        let c = a.add(b);

        assert_eq!(c.value().data, vec![11.0, 22.0, 33.0]);
        assert_eq!(c.value().shape, vec![3]);
    }

    #[test]
    fn add_appends_exactly_one_node() {
        // Sanity: each forward op should add precisely one node - no hidden nodes,
        // no double-recording. A bug here would silently corrupt graph traversal later.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::zeros(vec![2]));
        let b = Var::leaf(&tape, Tensor::zeros(vec![2]));
        assert_eq!(tape.len(), 2); // 2 leaves
        let _c = a.add(b);
        assert_eq!(tape.len(), 3); // + 1 add result
    }

    #[test]
    #[should_panic] // cross-tape operands must be rejected loudly, never silently desync
    fn add_rejects_cross_tape_operands() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let a = Var::leaf(&t1, Tensor::zeros(vec![2]));
        let b = Var::leaf(&t2, Tensor::zeros(vec![2]));
        let _ = a.add(b);
    }

    #[test]
    fn backward_through_add_routes_ones_to_each_input() {
        // c = a + b, then backward.  Seed is ones-like(c) (shape [3]).
        // Add's backward is identity passthrough, so a.grad == b.grad == ones.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let b = Var::leaf(&tape, Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]));
        let c = a.add(b);

        tape.backward(c.id);

        assert_eq!(a.grad().data, vec![1.0, 1.0, 1.0]);
        assert_eq!(b.grad().data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn backward_accumulates_when_a_leaf_is_reused() {
        // c = a + a.  The same leaf appears in two paths; each path contributes
        // a passthrough-ones, so a.grad must end up at twos. This is THE test
        // that proves grad accumulation works (initialising leaf grad to zero
        // and += each contribution).
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0], vec![2]));
        let c = a.add(a);

        tape.backward(c.id);

        assert_eq!(a.grad().data, vec![2.0, 2.0]);
    }

    #[test]
    fn backward_chains_through_multiple_ops() {
        // d = (a + b) + c.  All three leaves must end up with grad = ones -
        // the chain rule reduces to identity-times-identity = identity here,
        // but it exercises a 2-deep traversal and verifies that intermediate
        // node grads are correctly computed and forwarded to parents.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::zeros(vec![2]));
        let b = Var::leaf(&tape, Tensor::zeros(vec![2]));
        let c = Var::leaf(&tape, Tensor::zeros(vec![2]));
        let d = a.add(b).add(c);

        tape.backward(d.id);

        assert_eq!(a.grad().data, vec![1.0, 1.0]);
        assert_eq!(b.grad().data, vec![1.0, 1.0]);
        assert_eq!(c.grad().data, vec![1.0, 1.0]);
    }

    #[test]
    fn matmul_records_correct_forward_value() {
        // x = [[1, 2]] , w = [[3], [4]] , y = [[1·3 + 2·4]] = [[11]]
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]));
        let w = Var::leaf(&tape, Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]));
        let y = x.matmul(w);

        assert_eq!(y.value().shape, vec![1, 1]);
        assert_eq!(y.value().data, vec![11.0]);
    }

    #[test]
    fn backward_through_matmul_matches_closed_form() {
        // Same forward as above. Seed ∂L/∂y = ones([1, 1]).
        // Hand-derived gradients:
        //   ∂L/∂x = ones · wᵀ  = [[1]] · [[3, 4]]   = [[3, 4]]
        //   ∂L/∂w = xᵀ · ones  = [[1],[2]] · [[1]] = [[1], [2]]
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]));
        let w = Var::leaf(&tape, Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]));
        let y = x.matmul(w);

        tape.backward(y.id);

        assert_eq!(x.grad().shape, vec![1, 2]);
        assert_eq!(x.grad().data, vec![3.0, 4.0]);
        assert_eq!(w.grad().shape, vec![2, 1]);
        assert_eq!(w.grad().data, vec![1.0, 2.0]);
    }

    #[test]
    fn backward_through_matmul_then_add_chains_correctly() {
        // y = (x @ w) + b - exactly what a Linear's forward does.
        // With x=[[1,2]] , w=[[3],[4]] , b=[[5]]:
        //   y_value = [[16]]
        //   ∂L/∂x   = [[3, 4]]
        //   ∂L/∂w   = [[1], [2]]
        //   ∂L/∂b   = [[1]]   (add is identity passthrough)
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]));
        let w = Var::leaf(&tape, Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]));
        let b = Var::leaf(&tape, Tensor::from_vec(vec![5.0], vec![1, 1]));
        let y = x.matmul(w).add(b);

        assert_eq!(y.value().data, vec![16.0]);

        tape.backward(y.id);

        assert_eq!(x.grad().data, vec![3.0, 4.0]);
        assert_eq!(w.grad().data, vec![1.0, 2.0]);
        assert_eq!(b.grad().data, vec![1.0]);
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        // Position 0 -> angle 0 for every pair index -> rotation matrix is I.
        // First row of any RoPE-rotated tensor must equal the input row to
        // bit-exact f32 precision.
        let tape = Tape::new();
        let x_data: Vec<f32> = (0..1 * 4).map(|i| i as f32 + 1.0).collect();
        let x = Var::leaf(&tape, Tensor::from_vec(x_data.clone(), vec![1, 4]));
        let y = x.rope();
        assert_eq!(y.value().data, x_data);
    }

    #[test]
    fn rope_preserves_per_pair_norm() {
        // Each (x[pos, 2i], x[pos, 2i+1]) pair is rotated by an angle - the
        // 2-norm of every pair must be preserved up to f32 round-off.
        let tape = Tape::new();
        let seq = 5;
        let head_dim = 8;
        let x_data: Vec<f32> = (0..seq * head_dim)
            .map(|i| ((i as f32) * 0.371).sin())
            .collect();
        let x = Var::leaf(&tape, Tensor::from_vec(x_data.clone(), vec![seq, head_dim]));
        let y = x.rope();
        for pos in 0..seq {
            for i in 0..head_dim / 2 {
                let a = x_data[pos * head_dim + 2 * i];
                let b = x_data[pos * head_dim + 2 * i + 1];
                let ya = y.value().data[pos * head_dim + 2 * i];
                let yb = y.value().data[pos * head_dim + 2 * i + 1];
                let n_in = (a * a + b * b).sqrt();
                let n_out = (ya * ya + yb * yb).sqrt();
                assert!(
                    (n_in - n_out).abs() < 1e-5,
                    "rope changed pair norm at pos {} pair {}: in {} out {}",
                    pos,
                    i,
                    n_in,
                    n_out
                );
            }
        }
    }

    #[test]
    fn rope_backward_is_inverse_rotation() {
        // The 2D rotation per pair is orthogonal, so its backward is the
        // transpose, which is rotation by the negated angle. Concretely:
        // if we seed grad_y = ones, the backward must equal what a forward
        // rope() pass on `ones` would produce *with the sin sign flipped*.
        let tape = Tape::new();
        let seq = 3;
        let head_dim = 4;
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                (0..seq * head_dim).map(|i| (i as f32) * 0.1 + 0.5).collect(),
                vec![seq, head_dim],
            ),
        );
        let y = x.rope();
        tape.backward(y.id);

        // Hand-compute the expected gradient. With grad_y = ones:
        //   grad_x[pos, 2i]   =  cos + sin
        //   grad_x[pos, 2i+1] = -sin + cos
        let half = head_dim / 2;
        for pos in 0..seq {
            for i in 0..half {
                let theta_i = 10000_f32.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * theta_i;
                let c = angle.cos();
                let s = angle.sin();
                let want_a = c + s;
                let want_b = -s + c;
                let got_a = x.grad().data[pos * head_dim + 2 * i];
                let got_b = x.grad().data[pos * head_dim + 2 * i + 1];
                assert!(
                    (got_a - want_a).abs() < 1e-5 && (got_b - want_b).abs() < 1e-5,
                    "rope backward drift at pos {} pair {}: got ({}, {}) want ({}, {})",
                    pos,
                    i,
                    got_a,
                    got_b,
                    want_a,
                    want_b
                );
            }
        }
    }

    #[test]
    #[should_panic] // cross-tape rejection - same contract as add
    fn matmul_rejects_cross_tape_operands() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let x = Var::leaf(&t1, Tensor::zeros(vec![1, 2]));
        let w = Var::leaf(&t2, Tensor::zeros(vec![2, 1]));
        let _ = x.matmul(w);
    }

    #[test]
    fn sub_forward_and_backward() {
        // c = a - b.  Seed = ones-like(c).  ∂c/∂a = +1, ∂c/∂b = −1.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![5.0, 7.0], vec![2]));
        let b = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0], vec![2]));
        let c = a.sub(b);

        assert_eq!(c.value().data, vec![4.0, 5.0]);

        tape.backward(c.id);

        assert_eq!(a.grad().data, vec![1.0, 1.0]);
        assert_eq!(b.grad().data, vec![-1.0, -1.0]);
    }

    #[test]
    fn mul_forward_and_backward() {
        // c = a ⊙ b.  ∂c/∂a = b, ∂c/∂b = a.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![2.0, 3.0], vec![2]));
        let b = Var::leaf(&tape, Tensor::from_vec(vec![5.0, 7.0], vec![2]));
        let c = a.mul(b);

        assert_eq!(c.value().data, vec![10.0, 21.0]);

        tape.backward(c.id);

        assert_eq!(a.grad().data, vec![5.0, 7.0]); // = b
        assert_eq!(b.grad().data, vec![2.0, 3.0]); // = a
    }

    #[test]
    fn mul_self_gives_two_a_via_grad_accumulation() {
        // c = a * a   (square via reused leaf).  ∂c/∂a = 2a, exercised through accumulation.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![3.0, -4.0], vec![2]));
        let c = a.mul(a);

        assert_eq!(c.value().data, vec![9.0, 16.0]);

        tape.backward(c.id);

        assert_eq!(a.grad().data, vec![6.0, -8.0]); // = 2a
    }

    #[test]
    fn mean_forward_and_backward() {
        // mean of 4 elements: output [1] = sum/4.  Backward distributes 1/n.
        let tape = Tape::new();
        let a = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let m = a.mean();

        assert_eq!(m.value().shape, vec![1]);
        assert_eq!(m.value().data, vec![2.5]);

        tape.backward(m.id);

        assert_eq!(a.grad().data, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn mse_loss_full_chain_gradients() {
        // Mini regression-step graph:  pred = x · w,   loss = mean((pred − y)²).
        // Hand-derived for x=[[1.0]], w=[[3.0]], y=[[5.0]] (single-element n=1):
        //     pred = [[3.0]]                       → diff = [[-2.0]]
        //     sq   = [[4.0]]                       → loss = 4.0
        //   ∂loss/∂sq    = 1/n = 1                 (mean reducer)
        //   ∂loss/∂diff  = 2 · diff = -4           (square via reused leaf, accumulation)
        //   ∂loss/∂pred  = ∂loss/∂diff · 1 = -4    (sub, passthrough on lhs)
        //   ∂loss/∂y     = ∂loss/∂diff · -1 = +4   (sub, negation on rhs)
        //   ∂loss/∂w     = xᵀ · ∂loss/∂pred = -4   (matmul backward)
        //   ∂loss/∂x     = ∂loss/∂pred · wᵀ = -12
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0], vec![1, 1]));
        let w = Var::leaf(&tape, Tensor::from_vec(vec![3.0], vec![1, 1]));
        let y = Var::leaf(&tape, Tensor::from_vec(vec![5.0], vec![1, 1]));

        let pred = x.matmul(w);
        let diff = pred.sub(y);
        let sq = diff.mul(diff);
        let loss = sq.mean();

        assert_eq!(loss.value().data, vec![4.0]);

        tape.backward(loss.id);

        // Approximate-equal because float chains accumulate sub-ε round-off.
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(
            approx(w.grad().data[0], -4.0),
            "w.grad = {} (expected −4)",
            w.grad().data[0]
        );
        assert!(
            approx(x.grad().data[0], -12.0),
            "x.grad = {} (expected −12)",
            x.grad().data[0]
        );
        assert!(
            approx(y.grad().data[0], 4.0),
            "y.grad = {} (expected +4)",
            y.grad().data[0]
        );
    }

    // ─── M6: STE + transpose Var ops ──────────────────────────────────────

    #[test]
    fn quantise_weights_ste_forward_matches_dequant() {
        // Forward must equal γ · W_q from absmean_ternary directly.
        // W = [[2.0, -2.0, 0.0, 0.0]] → γ = 1.0, W_q = [1, -1, 0, 0] → output = [1, -1, 0, 0].
        let tape = Tape::new();
        let w_input = Tensor::from_vec(vec![2.0, -2.0, 0.0, 0.0], vec![1, 4]);
        let w = Var::leaf(&tape, w_input);
        let w_eff = w.quantise_weights_ste();

        assert_eq!(w_eff.value().shape, vec![1, 4]);
        assert_eq!(w_eff.value().data, vec![1.0, -1.0, 0.0, 0.0]); // γ=1, scaled identity
    }

    #[test]
    fn quantise_weights_ste_backward_is_identity() {
        // Seed grad-out = ones; STE means input grad must equal that seed exactly.
        let tape = Tape::new();
        let w = Var::leaf(
            &tape,
            Tensor::from_vec(vec![2.0, -2.0, 0.0, 0.0], vec![1, 4]),
        );
        let w_eff = w.quantise_weights_ste();

        tape.backward(w_eff.id);

        // Identity backward: w.grad should be ones-like(w), unchanged from the seed.
        assert_eq!(w.grad().data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn quantise_acts_ste_forward_matches_dequant() {
        // x = [[1.0, 2.0, -1.0, 0.5]], α[0] = 2.0.
        // x_q = round(x · 127/2) = [64, 127, -64, 32]
        // x_eff = (α/127) · x_q = (2/127) · [64, 127, -64, 32]
        //                       = [128/127, 2.0, -128/127, 64/127]
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5], vec![1, 4]),
        );
        let x_eff = x.quantise_acts_ste();

        let s = 2.0_f32 / 127.0;
        let expected = vec![64.0 * s, 127.0 * s, -64.0 * s, 32.0 * s];
        for i in 0..4 {
            assert!(
                (x_eff.value().data[i] - expected[i]).abs() < 1e-5,
                "x_eff[{}] = {}, expected {}",
                i,
                x_eff.value().data[i],
                expected[i]
            );
        }
    }

    #[test]
    fn quantise_acts_ste_backward_is_identity() {
        // Same identity-passthrough check as for weights.
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5], vec![1, 4]),
        );
        let x_eff = x.quantise_acts_ste();

        tape.backward(x_eff.id);

        assert_eq!(x.grad().data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn transpose_2d_var_forward_matches_tensor_transpose() {
        // Trivial check: forward agrees with Tensor::transpose_2d.
        let tape = Tape::new();
        let m = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mv = Var::leaf(&tape, m.clone());
        let mt = mv.transpose_2d();

        assert_eq!(mt.value().shape, vec![3, 2]);
        assert_eq!(mt.value().data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_var_backward_routes_grad_correctly() {
        // Use transpose inside a chain so the backward does real work.
        //   x : [3, 1] = [[1], [2], [3]]
        //   xᵀ : [1, 3] = [[1, 2, 3]]
        //   y = xᵀ · w   where w : [3, 1] = [[10], [20], [30]]
        //   y : [1, 1] = [[140]]
        // Seed: ∂L/∂y = ones [1, 1]
        // ∂L/∂(xᵀ) = ∂L/∂y · wᵀ = [[1]] · [[10, 20, 30]] = [[10, 20, 30]]   shape [1, 3]
        // ∂L/∂x    = transpose([[10, 20, 30]]) = [[10], [20], [30]]          shape [3, 1]
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]));
        let w = Var::leaf(&tape, Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3, 1]));
        let xt = x.transpose_2d();
        let y = xt.matmul(w);

        tape.backward(y.id);

        assert_eq!(x.grad().shape, vec![3, 1]);
        assert_eq!(x.grad().data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn causal_mask_zeroes_upper_triangle_and_preserves_lower() {
        // Input: 3x3 of ones. After mask, positions with j > i become -inf;
        // others remain 1.0. Checked exhaustively.
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0; 9], vec![3, 3]),
        );
        let y = x.causal_mask();
        for i in 0..3 {
            for j in 0..3 {
                let v = y.value().data[i * 3 + j];
                if j > i {
                    assert!(
                        v.is_infinite() && v.is_sign_negative(),
                        "expected -inf at ({},{}), got {}",
                        i,
                        j,
                        v
                    );
                } else {
                    assert_eq!(v, 1.0, "expected 1.0 at ({},{}), got {}", i, j, v);
                }
            }
        }
    }

    #[test]
    fn causal_mask_backward_zeros_upper_passes_lower() {
        // Seed grad-out = ones. Backward must produce ones in lower triangle
        // and zeros in upper triangle.
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0; 9], vec![3, 3]));
        let y = x.causal_mask();
        tape.backward(y.id);
        let g = x.grad();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(
                    g.data[i * 3 + j],
                    expected,
                    "grad mismatch at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn causal_mask_then_softmax_zeroes_future_attention() {
        // After mask + softmax, every row should sum to 1 and every column
        // j > i in row i should be exactly 0 (because exp(-inf) = 0).
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3, 3],
            ),
        );
        let attn = x.causal_mask().softmax();
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                let v = attn.value().data[i * 3 + j];
                if j > i {
                    assert_eq!(v, 0.0, "attention leaked to future at ({},{})", i, j);
                }
                row_sum += v;
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {} sum = {}",
                i,
                row_sum
            );
        }
    }

    // ─── M7 portion 1: activation / normalisation ops ─────────────────────

    #[test]
    fn relu_forward_clamps_negatives() {
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
        );
        let y = x.relu();
        assert_eq!(y.value().data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn relu_backward_masks_through_positive_only() {
        // Seed = ones-like(y) [5]. Backward kills positions where x ≤ 0.
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
        );
        let y = x.relu();
        tape.backward(y.id);
        assert_eq!(x.grad().data, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn softmax_forward_rows_sum_to_one() {
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.5, -0.5, 1.5], vec![2, 3]),
        );
        let s = x.softmax();
        assert_eq!(s.value().shape, vec![2, 3]);
        for i in 0..2 {
            let row_sum: f32 = s.value().data[i * 3..(i + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn softmax_backward_via_mean_loss_is_zero() {
        // mean(softmax(x)) = 1/n always (each row sums to 1) - that loss is
        // constant in x, so its gradient must be ≈ 0 everywhere.  This is a
        // free correctness check on the softmax JVP backward: any non-trivial
        // bug in the formula breaks the constant-loss → zero-grad equivalence.
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]),
        );
        let m = x.softmax().mean();
        tape.backward(m.id);
        for &v in x.grad().data.iter() {
            assert!(
                v.abs() < 1e-5,
                "softmax mean-loss grad should be 0, got {}",
                v
            );
        }
    }

    #[test]
    fn rmsnorm_forward_normalises_to_unit_rms() {
        // Each row's output should have RMS magnitude ≈ 1 (within ε).
        let tape = Tape::new();
        let x = Var::leaf(
            &tape,
            Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0], vec![2, 3]),
        );
        let y = x.rmsnorm();

        for i in 0..2 {
            let row = &y.value().data[i * 3..(i + 1) * 3];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / 3.0;
            let rms = mean_sq.sqrt();
            assert!((rms - 1.0).abs() < 1e-3, "row {} rms = {}", i, rms);
        }
    }

    #[test]
    fn rmsnorm_backward_matches_numerical_gradient() {
        // Cross-check the closed-form backward against finite differences.
        // Loss = mean(rmsnorm(x)). For each input element, compare autograd's
        // gradient to (loss(x+h) − loss(x−h)) / (2h).
        let x_vec = vec![1.0_f32, 2.0, 3.0];
        let h = 1e-3_f32;

        fn loss(x: &[f32]) -> f32 {
            let n = x.len() as f32;
            let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
            let rms = (mean_sq + 1e-5).sqrt();
            let inv = 1.0 / rms;
            x.iter().map(|v| v * inv).sum::<f32>() / n
        }

        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(x_vec.clone(), vec![1, 3]));
        let m = x.rmsnorm().mean();
        tape.backward(m.id);
        let g = x.grad();

        for i in 0..x_vec.len() {
            let mut xp = x_vec.clone();
            xp[i] += h;
            let mut xm = x_vec.clone();
            xm[i] -= h;
            let num = (loss(&xp) - loss(&xm)) / (2.0 * h);
            assert!(
                (g.data[i] - num).abs() < 1e-3,
                "rmsnorm grad[{}] auto = {}, num = {}",
                i,
                g.data[i],
                num
            );
        }
    }

    #[test]
    fn mul_scalar_var_forward_and_backward() {
        // y = 3 · x.  ∂y/∂x = 3 (uniform).
        let tape = Tape::new();
        let x = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, -3.0], vec![3]));
        let y = x.mul_scalar(3.0);
        assert_eq!(y.value().data, vec![3.0, 6.0, -9.0]);

        tape.backward(y.id);
        assert_eq!(x.grad().data, vec![3.0, 3.0, 3.0]);
    }

    // ─── M9 portion 1: embed + cross_entropy ────────────────────────────

    #[test]
    fn embed_forward_picks_correct_rows() {
        // weights[0] = [1, 2], weights[1] = [3, 4], weights[2] = [5, 6]
        // ids = [1, 0, 2] → output = rows [3, 4], [1, 2], [5, 6]
        let tape = Tape::new();
        let w = Var::leaf(
            &tape,
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
        );
        let y = w.embed(&[1, 0, 2]);
        assert_eq!(y.value().shape, vec![3, 2]);
        assert_eq!(y.value().data, vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn embed_backward_accumulates_gradient_into_correct_rows() {
        // ids = [1, 1, 0] → row 1 used twice, row 0 used once, row 2 unused.
        // Seed grad-out = ones [3, 2].  Expected weight gradient:
        //   row 0 = [1, 1]   (one usage)
        //   row 1 = [2, 2]   (two usages, += accumulation)
        //   row 2 = [0, 0]   (unused)
        let tape = Tape::new();
        let w = Var::leaf(&tape, Tensor::zeros(vec![3, 2]));
        let y = w.embed(&[1, 1, 0]);
        tape.backward(y.id);

        assert_eq!(w.grad().data, vec![1.0, 1.0, 2.0, 2.0, 0.0, 0.0,]);
    }

    #[test]
    fn cross_entropy_forward_matches_hand_computation() {
        // logits = [1, 2, 3], target = 2.
        // softmax(logits − 3) = exp([−2, −1, 0])/Z, Z = e⁻² + e⁻¹ + 1 ≈ 1.5032
        //   ≈ [0.0900, 0.2447, 0.6652]
        // loss = −log(0.6652) ≈ 0.4076
        let tape = Tape::new();
        let logits = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]));
        let loss = logits.cross_entropy(&[2]);

        let expected = 0.40760595_f32;
        assert!(
            (loss.value().data[0] - expected).abs() < 1e-4,
            "loss = {}, expected {}",
            loss.value().data[0],
            expected
        );
    }

    #[test]
    fn cross_entropy_backward_matches_softmax_minus_onehot() {
        // grad = (softmax − onehot(target)) / seq  (seq = 1 here)
        //      ≈ [0.0900, 0.2447, 0.6652 − 1] = [0.0900, 0.2447, −0.3348]
        let tape = Tape::new();
        let logits = Var::leaf(&tape, Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]));
        let loss = logits.cross_entropy(&[2]);
        tape.backward(loss.id);

        let g = logits.grad();
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(approx(g.data[0], 0.09003), "g[0] = {}", g.data[0]);
        assert!(approx(g.data[1], 0.24473), "g[1] = {}", g.data[1]);
        assert!(approx(g.data[2], -0.33476), "g[2] = {}", g.data[2]);
    }

    #[test]
    fn cross_entropy_low_when_confident_and_correct() {
        // A strongly favoured correct class should give near-zero loss.
        // This isn't just a sanity check - it's the property that makes
        // gradient descent on this loss DRIVE the correct logit upward.
        let tape = Tape::new();
        let logits = Var::leaf(&tape, Tensor::from_vec(vec![0.0, 0.0, 10.0], vec![1, 3]));
        let loss = logits.cross_entropy(&[2]);
        assert!(
            loss.value().data[0] < 0.01,
            "confident correct prediction should have near-zero loss, got {}",
            loss.value().data[0]
        );
    }
}
