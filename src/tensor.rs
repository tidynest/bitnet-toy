//! Minimal hand-rolled tensor: Just enough to build a transformer on top of it.
//!
//! Design choices for M1:
//! - Row-major, contiguous storage (`Vec<f32>` + `Vec<usize>` shape). No strides yet.
//!   so transpose physically copies data. Strides are a perf optimisation to be added at a later
//!   point.
//! - f32 throughout. BitNet's master weights are nominally BF16, but f32 is what `std`
//! gives us for free, and master precision is thrown away at export anyway.
//! - Owned data only (no views, no `Rc`). Sharing arrives in M3 where autograd needs it.

/// N-dimensional tensor with row-major contiguous f32 storage.
///
/// Invariant: `data.len() == shape.iter().product()`. All constructors enforce it;
/// every op below preserves it.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Total number of elements. Test-only sanity helper.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions (rank). 0 = scalar, 1 = vector, 2 = matrix, etc.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Build from a flat row-major vector plus a shape.
    /// Panics on mismatch - failing loudly here is far cheaper than debugging
    /// silent truncation or out-of-bounds reads later.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data len {} does not match shape product {}",
            data.len(),
            n
        );
        Tensor { data, shape }
    }

    /// All-zeros tensor of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape,
        }
    }

    /// All-ones tensor of the given shape.
    pub fn ones(shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; n],
            shape,
        }
    }

    /// 2D transpose: `[r, c]` -> `[c, r]`. Physically copies because we have no strides;
    /// real frameworks make this O(1) by editing metadata, but that costs a strides field
    /// on every op. We pay the copy and keep the type minimal.
    /// Panics on non-2D input - higher-rank `permute` arrives with attention (M7).
    pub fn transpose_2d(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "transpose_2d: expected rank-2, got rank {}",
            self.ndim()
        );
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                // (i, j) in self -> (j, i) in out. Row-major addressing: row * row_stride + col.
                out[j * r + i] = self.data[i * c + j];
            }
        }
        Tensor {
            data: out,
            shape: vec![c, r],
        }
    }

    /// 2D matrix multiply: `[m, k] @ [k, n] -> [m, n]`.
    /// Naive triple loop - cache blocking and SIMD would dramatically speed this up.
    /// but a toy model on a tiny corpus runs fine without them. We revisit only if a
    /// profiler tells us to.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "matmul: lhs must be rank-2, got rank {}",
            self.ndim()
        );
        assert_eq!(
            other.ndim(),
            2,
            "matmul: rhs must be rank-2, got rank {}",
            other.ndim()
        );
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(
            k, k2,
            "matmul shape mismatch: [{} {}] * [{} {}]",
            m, k, k2, n
        );

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                // Accumulate the (i, j) inner product in a local f32; the compiler will
                //keep `acc` in a register, avoiding repeated reads/writes to `out`.
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += self.data[i * k + kk] * other.data[kk * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        Tensor {
            data: out,
            shape: vec![m, n],
        }
    }

    /// Elementwise add. Shapes must match exactly. Broadcasting is deferred to the layer
    /// tha actually needs it (M2's bias add).
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "add: shape mismatch {:?} vs {:?}",
            self.shape, other.shape
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Elementwise subtract. Same shape-equality contract as `add`.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "sub: shape mismatch {:?} vs {:?}",
            self.shape, other.shape
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Elementwise (Hadamard) multiply. Same shape-equality contract as `add`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "mul: shape mismatch {:?} vs {:?}",
            self.shape, other.shape
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Multiply every element by a scalar. `f32` is `Copy`, so we pass by value.
    pub fn mul_scalar(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|x| x * s).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
// Entire module compiled out of release / `cargo run` build; only `cargo test` sees it
mod tests {
    use super::*; // pulls `Tensor` into the test module's scope

    #[test]
    fn from_vec_enforces_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    #[should_panic]
    // Assertion in `from_vec` MUST fire on mismatch - a silent shape lie would be a nightmare to
    // debug later
    fn from_vec_panics_on_mismatch() {
        let _ = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn zeros_and_ones_have_right_values() {
        let z = Tensor::zeros(vec![2, 3]);
        assert!(z.data.iter().all(|&x| x == 0.0));
        assert_eq!(z.shape, vec![2, 3]);

        let o = Tensor::ones(vec![2, 3]);
        assert!(o.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn transpose_2d_swaps_axes() {
        // [[1, 2, 3],          [[1, 4],
        //  [4, 5, 6]]           [2, 5],
        //                       [3, 6]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = a.transpose_2d();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn matmul_known_case() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_identity_is_noop() {
        // I · A = A. Catches transposed-index bugs that the symmetric 2×2 case would miss.
        let i = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let a = Tensor::from_vec(vec![3.0, 7.0, -1.0, 4.0], vec![2, 2]);
        assert_eq!(i.matmul(&a).data, a.data);
    }

    #[test]
    fn elementwise_add_and_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        assert_eq!(a.add(&b).data, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(a.mul(&b).data, vec![10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn mul_scalar_scales_everything() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
        assert_eq!(a.mul_scalar(2.5).data, vec![2.5, -5.0, 7.5]);
    }

    #[test]
    fn elementwise_sub() {
        let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(a.sub(&b).data, vec![4.0, 5.0, 6.0]);
    }
}
