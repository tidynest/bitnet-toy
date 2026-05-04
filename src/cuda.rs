//! CUDA back-end (Phases 1 + 2.0: matmul-only, cuBLAS sgemm).
//!
//! Production matmul path is **cuBLAS sgemm** (Chunk 2.0). The v0.18
//! hand-rolled NVRTC tile-based GEMM is retained `#[cfg(test)]` as an
//! independent reference implementation that the cuBLAS path is
//! cross-checked against. **NOT bit-identical** to CPU (parallel
//! reduction across thread blocks / SMs reorders the per-cell sum,
//! which is non-associative in f32). Agreement is within roughly
//! `1e-4 + 1e-4 * |val|` for the tensor magnitudes the model produces;
//! empirically the actual drift is ~1-5e-6 absolute on Phase 1 / 2.0
//! test shapes.
//!
//! Phase 1 surface (this commit):
//!   - `CudaTensor` owns one `CudaSlice<f32>` plus row-major shape
//!   - `CudaTensor::from_cpu(&Tensor)` / `to_cpu()` for explicit H<->D
//!   - `CudaTensor::matmul(&Self) -> Self` runs entirely device-side
//!   - `cuda_matmul(&Tensor, &Tensor)` convenience that copies in,
//!     multiplies, copies back (slow per call; useful for tests + demos)
//!
//! Phase 2+ (future sessions):
//!   - `CudaTensor`-resident attention head, then full forward, then
//!     backward, then optimiser - so weights / activations / gradients
//!     live on device across the entire training step.
//!   - Quant-aware kernels: skip the f32 multiply when the weight is
//!     +1 / -1 / 0; this is where BitNet's tensor-core advantage lives.
//!
//! The whole module is gated behind `#[cfg(feature = "cuda")]` so the
//! default `cargo build` stays dep-free and CI on machines without the
//! CUDA toolkit still passes. Run the GPU tests with
//! `cargo test --release --features cuda`.

use std::sync::{Arc, OnceLock};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use crate::tensor::Tensor;

/// Concatenated CUDA C source for the production op kernels (add,
/// mul_scalar, transpose_2d, causal_mask, softmax, rope). Compiled once
/// per process via NVRTC and cached in `CudaContextHolder`. Each kernel
/// is small (5-25 lines) and uses only `__global__` entry points (no
/// device-only helpers, no template kernels), so the compile is fast
/// and the resulting PTX is human-inspectable if anything misbehaves.
///
/// `expf`, `cosf`, `sinf`, `powf` come from the CUDA math library and
/// are available without an explicit include in NVRTC.
const KERNELS_SRC: &str = r#"
extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_scalar_f32(
    const float* __restrict__ x,
    float s,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] * s;
}

extern "C" __global__ void transpose_2d_f32(
    const float* __restrict__ in_,
    float* __restrict__ out,
    int r, int c)
{
    // Thread (b, a) writes out[a, b] = in_[b, a]. Output shape [c, r].
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a < c && b < r) {
        out[a * r + b] = in_[b * c + a];
    }
}

extern "C" __global__ void causal_mask_f32(
    const float* __restrict__ in_,
    float* __restrict__ out,
    int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        float v = in_[i * n + j];
        out[i * n + j] = (j > i) ? -INFINITY : v;
    }
}

extern "C" __global__ void softmax_row_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int m, int n)
{
    // One thread per row. Three passes (max / sum-of-exps / normalise),
    // matching the CPU code exactly. Trades intra-row parallelism for
    // simplicity; at our model scale (n = seq_len <= 128) the per-row
    // cost is small enough that this is not the bottleneck.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const float* xr = x + row * n;
    float* yr = out + row * n;

    float row_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
        float v = xr[j];
        if (v > row_max) row_max = v;
    }
    float denom = 0.0f;
    for (int j = 0; j < n; ++j) {
        float e = expf(xr[j] - row_max);
        yr[j] = e;
        denom += e;
    }
    float inv = 1.0f / denom;
    for (int j = 0; j < n; ++j) yr[j] *= inv;
}

extern "C" __global__ void rope_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int seq, int head_dim)
{
    // One thread per (pos, pair). pair indexes 0..head_dim/2.
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pos  = blockIdx.y * blockDim.y + threadIdx.y;
    int half = head_dim / 2;
    if (pair >= half || pos >= seq) return;

    float theta_i = powf(10000.0f, -(2.0f * (float)pair) / (float)head_dim);
    float angle   = (float)pos * theta_i;
    float c = cosf(angle);
    float s = sinf(angle);
    float a = x[pos * head_dim + 2 * pair];
    float b = x[pos * head_dim + 2 * pair + 1];
    out[pos * head_dim + 2 * pair]     = a * c - b * s;
    out[pos * head_dim + 2 * pair + 1] = a * s + b * c;
}

extern "C" __global__ void silu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float sig = 1.0f / (1.0f + expf(-v));
    out[idx] = v * sig;
}

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}
"#;

#[cfg(test)]
const MATMUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_f32(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int m, int k, int n)
{
    constexpr int TILE = 16;
    __shared__ float ls[TILE][TILE];
    __shared__ float rs[TILE][TILE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int kk_l = t * TILE + tx;
        const int kk_r = t * TILE + ty;
        ls[ty][tx] = (row < m && kk_l < k) ? lhs[row * k + kk_l] : 0.0f;
        rs[ty][tx] = (kk_r < k && col < n) ? rhs[kk_r * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += ls[ty][i] * rs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}
"#;

/// Per-process CUDA state: device 0's context + default stream + the
/// cuBLAS handle bound to that stream + handles for every NVRTC-
/// compiled production kernel (Phase 2.2 added six op kernels). In
/// test builds we also compile + cache the hand-rolled tile-GEMM
/// kernel so cuBLAS sgemm can be cross-checked against an independent
/// matmul implementation.
pub struct CudaContextHolder {
    /// Held for destructor lifetime: the `Arc<CudaStream>`, `CudaBlas`
    /// handle, and `CudaFunction` handles all reference this context
    /// but are not strong owners. If `ctx` were dropped first they
    /// would dangle. Not read directly outside this module.
    #[allow(dead_code)]
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,

    /// Phase 2.2 / 2.3 op kernels (compiled together from `KERNELS_SRC`):
    pub add_fn: CudaFunction,
    pub mul_scalar_fn: CudaFunction,
    pub transpose_2d_fn: CudaFunction,
    pub causal_mask_fn: CudaFunction,
    pub softmax_row_fn: CudaFunction,
    pub rope_fn: CudaFunction,
    pub silu_fn: CudaFunction,
    pub mul_fn: CudaFunction,

    /// Hand-rolled tile-based GEMM kernel from `MATMUL_KERNEL_SRC`.
    /// Test-only - the production matmul path uses cuBLAS sgemm (much
    /// better tuned across shapes than a single fixed 16x16 tile
    /// kernel). Useful for asserting cuBLAS correctness against an
    /// independent code path.
    #[cfg(test)]
    pub matmul_fn: CudaFunction,
}

/// Lazy global handle. First call attaches to GPU 0, instantiates a
/// cuBLAS handle on the default stream, and NVRTC-compiles the
/// production kernel module (Phase 2.2 op kernels). In test builds it
/// also compiles the hand-rolled GEMM kernel into a separate module
/// for the cuBLAS-vs-NVRTC cross-check. Later calls return the cached
/// holder. The `Result` is itself cached, so a CUDA-less environment
/// fails fast on every call without re-trying the device-open each time.
pub fn cuda_state() -> Result<&'static CudaContextHolder, String> {
    static STATE: OnceLock<Result<CudaContextHolder, String>> = OnceLock::new();
    let r = STATE.get_or_init(|| {
        let ctx = CudaContext::new(0)
            .map_err(|e| format!("CudaContext::new(0) failed: {e:?}"))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("CudaBlas::new failed: {e:?}"))?;

        // Phase 2.2 production kernel module: one NVRTC compile, six
        // function handles loaded out of the resulting module.
        let ops_ptx = compile_ptx(KERNELS_SRC)
            .map_err(|e| format!("NVRTC compile (KERNELS_SRC) failed: {e:?}"))?;
        let ops_module = ctx
            .load_module(ops_ptx)
            .map_err(|e| format!("load_module (ops) failed: {e:?}"))?;
        let load = |name: &str| -> Result<CudaFunction, String> {
            ops_module
                .load_function(name)
                .map_err(|e| format!("load_function({name:?}) failed: {e:?}"))
        };
        let add_fn = load("add_f32")?;
        let mul_scalar_fn = load("mul_scalar_f32")?;
        let transpose_2d_fn = load("transpose_2d_f32")?;
        let causal_mask_fn = load("causal_mask_f32")?;
        let softmax_row_fn = load("softmax_row_f32")?;
        let rope_fn = load("rope_f32")?;
        let silu_fn = load("silu_f32")?;
        let mul_fn = load("mul_f32")?;

        #[cfg(test)]
        let matmul_fn = {
            let ptx = compile_ptx(MATMUL_KERNEL_SRC)
                .map_err(|e| format!("NVRTC compile (MATMUL) failed: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module (matmul) failed: {e:?}"))?;
            module
                .load_function("matmul_f32")
                .map_err(|e| format!("load_function(\"matmul_f32\") failed: {e:?}"))?
        };
        Ok(CudaContextHolder {
            ctx,
            stream,
            blas,
            add_fn,
            mul_scalar_fn,
            transpose_2d_fn,
            causal_mask_fn,
            softmax_row_fn,
            rope_fn,
            silu_fn,
            mul_fn,
            #[cfg(test)]
            matmul_fn,
        })
    });
    r.as_ref().map_err(|s| s.clone())
}

/// Device-resident f32 tensor. Owns its `CudaSlice<f32>`; the slice is
/// reference-counted by cudarc so cloning is cheap on the Rust side and
/// safe across stream boundaries.
pub struct CudaTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
}

impl CudaTensor {
    /// Copy a CPU `Tensor` into device memory. Synchronous on the default
    /// stream; the returned `CudaTensor` is ready for kernel launches.
    pub fn from_cpu(t: &Tensor) -> Result<Self, String> {
        let s = cuda_state()?;
        let data = s
            .stream
            .clone_htod(&t.data)
            .map_err(|e| format!("clone_htod failed: {e:?}"))?;
        Ok(Self {
            data,
            shape: t.shape.clone(),
        })
    }

    /// Copy device memory back into a CPU `Tensor`. Synchronous.
    pub fn to_cpu(&self) -> Result<Tensor, String> {
        let s = cuda_state()?;
        let v = s
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| format!("clone_dtoh failed: {e:?}"))?;
        Ok(Tensor::from_vec(v, self.shape.clone()))
    }

    /// Hand-rolled tile-based GEMM via the v0.18 NVRTC kernel. Kept as a
    /// reference implementation that we can A/B against the cuBLAS path
    /// in tests and benchmarks. Production callers should use the
    /// `MatMul::matmul` trait method (which dispatches to cuBLAS); this
    /// method is `#[cfg(test)]` to avoid surface-area drift while still
    /// letting the test suite exercise the bespoke kernel.
    #[cfg(test)]
    pub fn matmul_nvrtc(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(rhs.shape.len(), 2);
        assert_eq!(self.shape[1], rhs.shape[0]);
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let k_i = i32::try_from(k).expect("k exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out: CudaSlice<f32> = s
            .stream
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_i as u32).div_ceil(TILE),
                (m_i as u32).div_ceil(TILE),
                1,
            ),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = s.stream.launch_builder(&s.matmul_fn);
        launcher.arg(&self.data);
        launcher.arg(&rhs.data);
        launcher.arg(&mut out);
        launcher.arg(&m_i);
        launcher.arg(&k_i);
        launcher.arg(&n_i);
        // Safety: same as the v0.18 kernel - signature matches the six
        // args pushed above; output buffer sized m*n; lhs / rhs sized
        // m*k and k*n.
        unsafe { launcher.launch(cfg) }.expect("NVRTC kernel launch failed");
        s.stream.synchronize().expect("stream synchronize failed");
        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

/// Production GEMM via cuBLAS sgemm. Output shape `[m, n]` from inputs
/// `[m, k]` and `[k, n]`. Synchronises the stream before returning so
/// `to_cpu()` immediately afterwards is safe; in Phase 2+ we will
/// compose multiple cuBLAS / kernel calls without intermediate sync.
///
/// **Row-major-via-column-major trick.** cuBLAS uses Fortran (column-
/// major) storage: a (rows, cols) col-major matrix has element
/// `[r, c]` at index `c * rows + r`. Our tensors are row-major. The
/// trick: a row-major matrix viewed as col-major has its dimensions
/// transposed (the same bytes describe `(M, N)` row-major *and* `(N,
/// M)` col-major). So to compute `C_row = A_row @ B_row` of shape
/// `(M, N)` we ask cuBLAS for `C_col = B_col @ A_col` of shape `(N,
/// M)` - the bytes that come out match the C_row layout exactly.
/// Passing B before A and swapping (m, n) is the entire adapter; no
/// transpose flags are needed.
///
/// Errors from cuBLAS / cudarc (bad shape, lost device, allocation
/// failure) are panics here; callers have no recovery path and the
/// rest of the project is panic-on-invariant style.
impl crate::device::MatMul for CudaTensor {
    fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape.len(), 2, "lhs must be 2-D, got {:?}", self.shape);
        assert_eq!(rhs.shape.len(), 2, "rhs must be 2-D, got {:?}", rhs.shape);
        assert_eq!(
            self.shape[1], rhs.shape[0],
            "shape mismatch: {:?} @ {:?}", self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let k_i = i32::try_from(k).expect("k exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");

        let mut out: CudaSlice<f32> = s
            .stream
            .alloc_zeros::<f32>(m * n)
            .expect("alloc_zeros failed");

        // Compute C_col = B_col @ A_col of shape (N, M).
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            // cuBLAS-side dimensions: m_cublas=N, n_cublas=M, k_cublas=K.
            m: n_i,
            n: m_i,
            k: k_i,
            alpha: 1.0,
            // Leading dim of B in col-major is N (matches row-major's
            // stride - both views read N f32 values per "row" of B).
            lda: n_i,
            // Leading dim of A in col-major is K.
            ldb: k_i,
            beta: 0.0,
            // Leading dim of C in col-major is N (= row-major's stride).
            ldc: n_i,
        };

        // Safety: shapes / strides above match the row-major <-> col-major
        // adapter described in the doc comment; alloc_zeros above gave us
        // exactly m*n f32 of writable device memory; self.data and
        // rhs.data are owned device slices of size m*k and k*n.
        unsafe { s.blas.gemm(cfg, &rhs.data, &self.data, &mut out) }
            .expect("cuBLAS sgemm failed");
        s.stream.synchronize().expect("stream synchronize failed");

        Self {
            data: out,
            shape: vec![m, n],
        }
    }
}

// ---- Phase 2.2 op trait impls for CudaTensor.
//
// Pattern: derive launch dimensions from `self.shape`, alloc the
// output, push args via `launch_builder`, launch unsafe, synchronise
// the stream (so callers can `to_cpu()` immediately or pass to the
// next op without explicit barrier). cudarc errors `.expect()` at the
// boundary - same panic semantics as the matmul impl above.
//
// Block dim choices:
//   add / mul_scalar           : 1-D grid, 256 threads/block
//   transpose_2d / causal_mask : 2-D grid, 16x16 threads/block
//   softmax_row                : 1-D grid over rows, 256 threads/block
//   rope                       : 2-D grid over (pair, pos), 16x16 block

impl crate::device::Add for CudaTensor {
    fn add(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape, rhs.shape,
            "add: shape mismatch: {:?} vs {:?}", self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(n).expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.add_fn);
        l.arg(&self.data);
        l.arg(&rhs.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: kernel signature (const float*, const float*, float*,
        // int) matches the four args; output is sized n; lhs / rhs are
        // both sized n (asserted above).
        unsafe { l.launch(cfg) }.expect("add_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: self.shape.clone() }
    }
}

impl crate::device::MulScalar for CudaTensor {
    fn mul_scalar(&self, s: f32) -> Self {
        let st = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = st.stream.alloc_zeros::<f32>(n).expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = st.stream.launch_builder(&st.mul_scalar_fn);
        l.arg(&self.data);
        l.arg(&s);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, float, float*, int) matches
        // the four args; output sized n; input sized n.
        unsafe { l.launch(cfg) }.expect("mul_scalar_f32 launch failed");
        st.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: self.shape.clone() }
    }
}

impl crate::device::Transpose2D for CudaTensor {
    fn transpose_2d(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "transpose_2d: rank-2 only, got {:?}", self.shape);
        let s = cuda_state().expect("cuda_state failed");
        let r = self.shape[0];
        let c = self.shape[1];
        let r_i = i32::try_from(r).expect("r exceeds i32");
        let c_i = i32::try_from(c).expect("c exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(r * c).expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((r_i as u32).div_ceil(TILE), (c_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.transpose_2d_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&r_i);
        l.arg(&c_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; input sized r*c; output sized r*c (= c*r).
        unsafe { l.launch(cfg) }.expect("transpose_2d_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: vec![c, r] }
    }
}

impl crate::device::CausalMask for CudaTensor {
    fn causal_mask(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "causal_mask: rank-2 only, got {:?}", self.shape);
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(m * n).expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(TILE), (m_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.causal_mask_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("causal_mask_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: vec![m, n] }
    }
}

impl crate::device::Softmax for CudaTensor {
    fn softmax(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "softmax: rank-2 only, got {:?}", self.shape);
        let s = cuda_state().expect("cuda_state failed");
        let m = self.shape[0];
        let n = self.shape[1];
        let m_i = i32::try_from(m).expect("m exceeds i32");
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(m * n).expect("alloc_zeros failed");
        // One thread per row. 256 rows per block; covers shakespeare
        // configs without needing a multi-thread-per-row reduction.
        let cfg = LaunchConfig {
            grid_dim: ((m_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.softmax_row_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&m_i);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized m*n.
        unsafe { l.launch(cfg) }.expect("softmax_row_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: vec![m, n] }
    }
}

impl crate::device::Silu for CudaTensor {
    fn silu(&self) -> Self {
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(n).expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.silu_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, float*, int) matches the
        // three args; both buffers sized n.
        unsafe { l.launch(cfg) }.expect("silu_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: self.shape.clone() }
    }
}

impl crate::device::Mul for CudaTensor {
    fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape, rhs.shape,
            "mul: shape mismatch: {:?} vs {:?}", self.shape, rhs.shape
        );
        let s = cuda_state().expect("cuda_state failed");
        let n = self.data.len();
        let n_i = i32::try_from(n).expect("n exceeds i32");
        let mut out = s.stream.alloc_zeros::<f32>(n).expect("alloc_zeros failed");
        let cfg = LaunchConfig {
            grid_dim: ((n_i as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.mul_fn);
        l.arg(&self.data);
        l.arg(&rhs.data);
        l.arg(&mut out);
        l.arg(&n_i);
        // Safety: signature (const float*, const float*, float*, int)
        // matches the four args; output sized n; lhs / rhs both sized n
        // (asserted above).
        unsafe { l.launch(cfg) }.expect("mul_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: self.shape.clone() }
    }
}

impl crate::device::Rope for CudaTensor {
    fn rope(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "rope: rank-2 only, got {:?}", self.shape);
        let s = cuda_state().expect("cuda_state failed");
        let seq = self.shape[0];
        let head_dim = self.shape[1];
        assert!(head_dim % 2 == 0, "rope: head_dim ({head_dim}) must be even");
        let seq_i = i32::try_from(seq).expect("seq exceeds i32");
        let hd_i = i32::try_from(head_dim).expect("head_dim exceeds i32");
        let half_i = hd_i / 2;
        let mut out = s.stream.alloc_zeros::<f32>(seq * head_dim).expect("alloc_zeros failed");
        const TILE: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: ((half_i as u32).div_ceil(TILE), (seq_i as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.stream.launch_builder(&s.rope_fn);
        l.arg(&self.data);
        l.arg(&mut out);
        l.arg(&seq_i);
        l.arg(&hd_i);
        // Safety: signature (const float*, float*, int, int) matches
        // the four args; both buffers sized seq*head_dim.
        unsafe { l.launch(cfg) }.expect("rope_f32 launch failed");
        s.stream.synchronize().expect("synchronize failed");
        Self { data: out, shape: vec![seq, head_dim] }
    }
}

/// Convenience: copy two CPU tensors to device, multiply, copy back.
/// Per-call H<->D overhead dominates for the matmul shapes the v0.13
/// model produces (~150 us for the copies vs ~10 us for the kernel), so
/// this is intentionally a demo / test entry point and not used by
/// training. Phase 2 will keep tensors device-resident across whole
/// blocks.
pub fn cuda_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
    use crate::device::MatMul;
    let l = CudaTensor::from_cpu(lhs)?;
    let r = CudaTensor::from_cpu(rhs)?;
    l.matmul(&r).to_cpu()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{
        Add, CausalMask, MatMul, Mul, MulScalar, Rope, Silu, Softmax, Transpose2D,
    };

    /// Maximum acceptable absolute error per cell from a CPU-vs-CUDA
    /// matmul. Parallel reduction across thread blocks reorders the
    /// per-cell sum, and f32 add is non-associative, so bit-equality is
    /// impossible for any non-trivial K. The bound below is loose enough
    /// to absorb that drift across the prime-shaped test matrices.
    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 1e-4;

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape, b.shape, "shape mismatch");
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (&x, &y)) in a.data.iter().zip(&b.data).enumerate() {
            let abs = (x - y).abs();
            let rel = if y.abs() > 1e-6 { abs / y.abs() } else { 0.0 };
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            assert!(
                abs <= ABS_TOL + REL_TOL * y.abs(),
                "drift at idx {i}: cpu = {y}, cuda = {x}, |diff| = {abs}"
            );
        }
        eprintln!("cpu vs cuda: max |diff| = {max_abs:.3e}, max rel = {max_rel:.3e}");
    }

    /// Round-trip through device memory must preserve the bytes exactly:
    /// no math, just memcpy, so the tolerance here is byte-equality.
    #[test]
    fn cuda_round_trip_preserves_bytes() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let host = Tensor::from_vec((0..30).map(|i| i as f32 * 0.123).collect(), vec![5, 6]);
        let dev = CudaTensor::from_cpu(&host).expect("H->D copy failed");
        let back = dev.to_cpu().expect("D->H copy failed");
        assert_eq!(host.shape, back.shape);
        assert_eq!(host.data, back.data, "round-trip lost bytes");
    }

    /// Tile-aligned 32x32 @ 32x32 - all four edges land on TILE
    /// boundaries, so no boundary-mask thread does anything different
    /// from an interior thread. Should agree with CPU within the
    /// floating-point tolerance defined above.
    #[test]
    fn cuda_matmul_tile_aligned_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 32usize;
        let k = 32usize;
        let n = 32usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let cpu = lhs.matmul(&rhs);
        let gpu = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    /// Prime dimensions push every output edge through the boundary
    /// guard inside the kernel (`row < m && col < n`) and also exercise
    /// the partial-tile K-loop. If the bounds check is wrong - off by
    /// one, transposed, etc. - this test will diverge wildly, not just
    /// in the last bit.
    #[test]
    fn cuda_matmul_non_tile_aligned_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 17usize;
        let k = 23usize;
        let n = 31usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let cpu = lhs.matmul(&rhs);
        let gpu = cuda_matmul(&lhs, &rhs).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    /// cuBLAS sgemm and the hand-rolled NVRTC tile kernel both compute
    /// the same row-major GEMM, but their per-cell summation orders
    /// differ (cuBLAS uses internal blocking optimised for tensor cores
    /// / SM scheduling; the NVRTC path uses a fixed 16x16 tile order).
    /// f32 is non-associative, so they will diverge by a few ULPs. The
    /// tolerance below is the same as cpu-vs-cuda, which is loose enough
    /// to absorb both backends' differences from the textbook ordering.
    #[test]
    fn cublas_and_nvrtc_kernels_agree_within_tolerance() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let m = 64usize;
        let k = 192usize;
        let n = 384usize;
        let lhs_data: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let rhs_data: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 * 0.0259).cos() * 0.1)
            .collect();
        let lhs = CudaTensor::from_cpu(&Tensor::from_vec(lhs_data, vec![m, k]))
            .expect("H->D failed");
        let rhs = CudaTensor::from_cpu(&Tensor::from_vec(rhs_data, vec![k, n]))
            .expect("H->D failed");

        let cublas_out = lhs.matmul(&rhs).to_cpu().unwrap();
        let nvrtc_out = lhs.matmul_nvrtc(&rhs).to_cpu().unwrap();
        assert_close(&cublas_out, &nvrtc_out);
    }

    /// Realistic v0.13 attention shape: hidden_dim 192 broken into
    /// 12 heads x 16 head_dim. Tests Q computation: x [seq=64, hidden=192]
    /// times W_q [hidden=192, head_dim=16] -> [seq=64, head_dim=16]. The
    /// rectangular and skewed shapes catch grid-dim arithmetic bugs that
    /// the square test would not.
    #[test]
    fn cuda_matmul_attention_q_shape_matches_cpu() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let seq = 64usize;
        let hidden = 192usize;
        let head_dim = 16usize;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let w_data: Vec<f32> = (0..hidden * head_dim)
            .map(|i| (i as f32 * 0.0259).cos() * 0.1)
            .collect();
        let x = Tensor::from_vec(x_data, vec![seq, hidden]);
        let w = Tensor::from_vec(w_data, vec![hidden, head_dim]);

        let cpu = x.matmul(&w);
        let gpu = cuda_matmul(&x, &w).expect("CUDA matmul failed");
        assert_close(&gpu, &cpu);
    }

    // ---- Phase 2.2 per-op CPU vs CUDA agreement tests. Each one
    // computes the op on a CPU `Tensor`, copies the same input to a
    // `CudaTensor`, runs the trait-method GPU path, and asserts the
    // results agree within tolerance. Uses prime-ish dimensions where
    // possible so boundary handling is exercised in addition to the
    // happy interior path.

    fn random_tensor(rows: usize, cols: usize, seed: f32) -> Tensor {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| (i as f32 * 0.0173 + seed).sin() * 0.5)
                .collect(),
            vec![rows, cols],
        )
    }

    #[test]
    fn add_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(7, 19, 0.1);
        let b = random_tensor(7, 19, 0.7);
        let cpu = <Tensor as Add>::add(&a, &b);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .add(&CudaTensor::from_cpu(&b).unwrap())
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn mul_scalar_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(11, 23, 0.2);
        let cpu = <Tensor as MulScalar>::mul_scalar(&a, 0.31415);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .mul_scalar(0.31415)
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn transpose_2d_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(13, 17, 0.3);
        let cpu = <Tensor as Transpose2D>::transpose_2d(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .transpose_2d()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn causal_mask_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // Square seq x seq is the only shape causal_mask is meaningful
        // on; primes (17) push every edge through the bounds checks.
        let a = random_tensor(17, 17, 0.4);
        let cpu = <Tensor as CausalMask>::causal_mask(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .causal_mask()
            .to_cpu()
            .unwrap();
        // -inf cells must match exactly. Comparing -inf with assert_close's
        // f32 tolerance produces NaN diff; do a separate equality pass.
        for (i, (&c, &g)) in cpu.data.iter().zip(&gpu.data).enumerate() {
            if c.is_infinite() {
                assert_eq!(c, g, "infinity mismatch at idx {i}");
            } else {
                let abs = (c - g).abs();
                assert!(
                    abs <= ABS_TOL + REL_TOL * c.abs(),
                    "drift at idx {i}: cpu = {c}, cuda = {g}"
                );
            }
        }
    }

    #[test]
    fn softmax_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(7, 31, 0.5);
        let cpu = <Tensor as Softmax>::softmax(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .softmax()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
        // Each GPU row must sum to ~1 - the headline correctness
        // property of softmax. Tighter tolerance than the per-cell
        // diff because sum-of-row is an aggregate.
        for i in 0..gpu.shape[0] {
            let s: f32 = gpu.data[i * gpu.shape[1]..(i + 1) * gpu.shape[1]]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-5, "GPU row {i} sum = {s}");
        }
    }

    #[test]
    fn rope_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // head_dim must be even. Use realistic v0.13 attention shape.
        let a = random_tensor(64, 16, 0.6);
        let cpu = <Tensor as Rope>::rope(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .rope()
            .to_cpu()
            .unwrap();
        // RoPE uses transcendentals (cos/sin/pow); GPU + CPU may use
        // slightly different math implementations. Slightly looser
        // tolerance than the elementwise-arithmetic ops.
        for (i, (&c, &g)) in cpu.data.iter().zip(&gpu.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "rope drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    #[test]
    fn silu_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(13, 19, 0.7);
        let cpu = <Tensor as Silu>::silu(&a);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .silu()
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    #[test]
    fn mul_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let a = random_tensor(11, 17, 0.8);
        let b = random_tensor(11, 17, 0.9);
        let cpu = <Tensor as Mul>::mul(&a, &b);
        let gpu = CudaTensor::from_cpu(&a)
            .unwrap()
            .mul(&CudaTensor::from_cpu(&b).unwrap())
            .to_cpu()
            .unwrap();
        assert_close(&gpu, &cpu);
    }

    /// **The headline test of Chunk 2.3.** Matches the Chunk 2.2
    /// attention test in spirit: the same generic `ffn_inference<T>`
    /// runs on both backends and the outputs agree within tolerance.
    /// SwiGLU layout: gate_w, up_w map `[hidden, ffn]`; down_w maps
    /// `[ffn, hidden]`. Tolerance is the same `1e-3` chained-op
    /// budget as the attention test.
    #[test]
    fn ffn_inference_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::ffn_inference;
        let seq = 16usize;
        let hidden = 32usize;
        let ffn = 64usize;

        let x = random_tensor(seq, hidden, 2.0);
        let w_gate = random_tensor(hidden, ffn, 2.1);
        let w_up = random_tensor(hidden, ffn, 2.2);
        let w_down = random_tensor(ffn, hidden, 2.3);

        let cpu_out = ffn_inference::<Tensor>(&x, &w_gate, &w_up, &w_down);
        let x_g = CudaTensor::from_cpu(&x).unwrap();
        let g_g = CudaTensor::from_cpu(&w_gate).unwrap();
        let u_g = CudaTensor::from_cpu(&w_up).unwrap();
        let d_g = CudaTensor::from_cpu(&w_down).unwrap();
        let gpu_out = ffn_inference::<CudaTensor>(&x_g, &g_g, &u_g, &d_g)
            .to_cpu()
            .unwrap();

        assert_eq!(cpu_out.shape, gpu_out.shape, "ffn output shape mismatch");
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "ffn drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }

    /// **The headline test of Chunk 2.2.** `attention_head_inference`
    /// is the smallest realistic model layer expressed purely against
    /// the trait surface. Running it once with `Tensor` and once with
    /// `CudaTensor` over the same inputs must produce matching outputs
    /// within the cross-backend tolerance. If any of the six new ops
    /// regresses on either backend, this test catches it.
    #[test]
    fn attention_head_inference_cpu_vs_cuda_matches() {
        if cuda_state().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        use crate::device::attention_head_inference;
        let seq = 16usize;
        let hidden = 32usize;
        let head_dim = 8usize;

        let x = random_tensor(seq, hidden, 1.0);
        let w_q = random_tensor(hidden, head_dim, 1.1);
        let w_k = random_tensor(hidden, head_dim, 1.2);
        let w_v = random_tensor(hidden, head_dim, 1.3);
        let w_o = random_tensor(head_dim, hidden, 1.4);

        let cpu_out = attention_head_inference::<Tensor>(&x, &w_q, &w_k, &w_v, &w_o, head_dim);

        let x_g = CudaTensor::from_cpu(&x).unwrap();
        let q_g = CudaTensor::from_cpu(&w_q).unwrap();
        let k_g = CudaTensor::from_cpu(&w_k).unwrap();
        let v_g = CudaTensor::from_cpu(&w_v).unwrap();
        let o_g = CudaTensor::from_cpu(&w_o).unwrap();
        let gpu_out =
            attention_head_inference::<CudaTensor>(&x_g, &q_g, &k_g, &v_g, &o_g, head_dim)
                .to_cpu()
                .unwrap();

        assert_eq!(cpu_out.shape, gpu_out.shape, "attention output shape mismatch");
        // Tolerance widened slightly because the chain accumulates
        // drift across 5 matmuls + softmax + RoPE; per-op tolerance is
        // 1e-4 + 1e-4*|val| but the chained product needs ~1e-3.
        for (i, (&c, &g)) in cpu_out.data.iter().zip(&gpu_out.data).enumerate() {
            let abs = (c - g).abs();
            assert!(
                abs <= 1e-3 + 1e-3 * c.abs(),
                "attention drift at idx {i}: cpu = {c}, cuda = {g}"
            );
        }
    }
}
