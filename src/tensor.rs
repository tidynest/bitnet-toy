//! Minimal hand-rolled tensor: Just enough to build a transformer on top of it.
//!
//! Design choices for M1:
//! - Row-major, contiguous storage (`Vec<f32>` + `Vec<usize>` shape). No strides yet.
//!   so transpose physically copies data. Strides are a perf optimisation to be added at a later
//!   point.
//! - f32 throughout. BitNet's master weights are nominally BF16, but f32 is what `std`
//!   gives us for free, and master precision is thrown away at export anyway.
//! - Owned data only (no views, no `Rc`). Sharing arrives in M3 where autograd needs it.
//!
//! Parallelism (v0.10, pooled since issue #7): `Tensor::matmul` shards the
//! output rows across threads when the result is large enough to amortise
//! dispatch overhead. Each band is a contiguous, disjoint slice of the
//! output `Vec<f32>` via `chunks_mut`, so no atomics or locks guard the
//! data and the result is **bit-identical** to the serial version
//! regardless of thread count (per-row summation order is preserved, and
//! rows do not combine). Bands beyond the first are executed by a
//! long-lived channel-fed worker pool (`matmul_pool`) instead of freshly
//! spawned OS threads - a channel send is ~1-5 us vs ~10-30 us per spawn,
//! about x2 per call at training shapes. Thread (band) count is read once
//! via `OnceLock` from the env var `BITNET_MATMUL_THREADS`; default 1
//! (serial) because the outer batch level is the productive place to
//! spend parallelism during training.
//!
//! SIMD (v0.11): the inner kernel is rewritten as register-blocked AXPY
//! (loop order i, kk, j) so the innermost loop becomes
//! `out_row[j] += a * rhs_row[j]` over contiguous f32 slices. On x86_64
//! with AVX2 detected at runtime we issue 256-bit `_mm256_mul_ps` +
//! `_mm256_add_ps` pairs, processing 8 f32 per inner-loop iteration; tail
//! elements (n not a multiple of 8) fall back to scalar. AVX2 is
//! bit-identical to scalar because per-cell accumulation order
//! (kk = 0..k) is unchanged and we deliberately do not use FMA, which
//! would round once instead of twice.
//!
//! SIMD (v0.17): a third path is layered on top - AVX-512 foundation
//! (`avx512f`, 16 f32 per inner-loop step via `_mm512_mul_ps` +
//! `_mm512_add_ps`) for Sapphire Rapids and later. Detection is cached once
//! via `OnceLock`; the dispatcher picks the widest path that actually wins.
//! (v0.19): on Zen 4 (CPUID family `0x19`) AVX-512 is double-pumped and ~9%
//! slower on this bandwidth-bound matmul, so the dispatcher auto-selects AVX2
//! there. Export `BITNET_MATMUL_SIMD=avx512` to force AVX-512 back on (A/B
//! timing), `=avx2` to force AVX2 anywhere, or `=off` (also `0 | none |
//! scalar`) to force the scalar path. On aarch64 (issue #6) the dispatcher
//! picks the 4-lane NEON kernel unconditionally (NEON is baseline there);
//! only `=off` opts out. All paths are byte-identical per cell.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

/// Output-element count below which `matmul` stays serial. Handing a band to
/// a pool worker costs a channel send + wake (~1-5 us); for tiny matmuls
/// (e.g. per-position attention scoring on small `head_dim`) that overhead
/// dominates. 256 elements is the empirical break-even on the 7940HS for the
/// v0.9 model shapes (measured against thread-spawn cost; the pool only
/// lowers the constant, so the threshold stays conservative).
const MATMUL_PARALLEL_THRESHOLD: usize = 256;

/// Lifetime count of pool workers spawned. Observability hook for the
/// issue #7 invariant: workers are spawned once, then reused forever.
static POOL_WORKERS_SPAWNED: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
fn matmul_pool_spawned_workers() -> usize {
    POOL_WORKERS_SPAWNED.load(Ordering::Relaxed)
}

/// One output band of a sharded matmul, sent to a pool worker. Raw
/// pointers rather than slices because the job must cross an mpsc
/// channel into a long-lived thread (`'static` bound); the submitting
/// thread restores borrow discipline by blocking until every band it
/// dispatched has acked, so the pointed-to buffers outlive every
/// worker-side access (see `AckDrain`).
struct MatmulJob {
    lhs: *const f32,
    rhs: *const f32,
    out: *mut f32,
    rows: usize,
    k: usize,
    n: usize,
    ack: mpsc::Sender<()>,
}

// Safety: the pointers are dereferenced only between job receipt and
// ack send. The submitter keeps the buffers alive until all acks are
// in, and each job's `out` band is a disjoint chunk of the output, so
// no two threads ever alias the same writable memory.
unsafe impl Send for MatmulJob {}

impl MatmulJob {
    /// Reconstruct the slices and run the shared AXPY kernel.
    fn run(self) {
        // Safety: submitter guarantees liveness and disjointness for
        // the duration of this call (it is still blocked in AckDrain).
        let lhs = unsafe { std::slice::from_raw_parts(self.lhs, self.rows * self.k) };
        let rhs = unsafe { std::slice::from_raw_parts(self.rhs, self.k * self.n) };
        let out = unsafe { std::slice::from_raw_parts_mut(self.out, self.rows * self.n) };
        matmul_kernel(lhs, rhs, out, self.rows, self.k, self.n);
        // A closed receiver just means the submitter is already
        // draining after a panic; the buffers stayed valid throughout.
        let _ = self.ack.send(());
    }
}

/// Blocks until `remaining` acks arrive - **including on unwind**. This
/// is what makes the raw-pointer jobs sound: the submitter creates the
/// drain before dispatching and only counts successfully sent jobs, so
/// no code path (early return, panic, assert) can free the lhs/rhs/out
/// buffers while a worker still holds pointers into them. `recv` only
/// errors once every outstanding job's ack sender is gone, i.e. when
/// no worker can touch the buffers any more, so breaking on `Err` is
/// safe and avoids deadlock if a worker dies.
struct AckDrain<'a> {
    rx: &'a mpsc::Receiver<()>,
    remaining: usize,
}

impl Drop for AckDrain<'_> {
    fn drop(&mut self) {
        while self.remaining > 0 {
            if self.rx.recv().is_err() {
                break;
            }
            self.remaining -= 1;
        }
    }
}

/// The long-lived matmul worker pool (issue #7). One mpsc channel per
/// worker: band `i` of a call goes to worker `(i - 1) % workers` while
/// band 0 runs on the calling thread, so a `threads = t` call keeps the
/// same banding (and therefore bit-identical output) as the old
/// spawn-per-call path. Workers park in `recv` between matmuls; an
/// idle pool costs zero CPU.
struct MatmulPool {
    senders: Vec<mpsc::Sender<MatmulJob>>,
}

/// Pool accessor. Spawned once on first parallel matmul: one worker
/// per hardware thread up to 8 total, minus the calling thread (which
/// always computes band 0 itself rather than idling in a wait).
fn matmul_pool() -> &'static MatmulPool {
    static POOL: OnceLock<MatmulPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(8)
            .saturating_sub(1)
            .max(1);
        let senders = (0..workers)
            .map(|i| {
                let (tx, rx) = mpsc::channel::<MatmulJob>();
                std::thread::Builder::new()
                    .name(format!("bitnet-matmul-{i}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            job.run();
                        }
                    })
                    .expect("matmul pool: worker spawn failed");
                POOL_WORKERS_SPAWNED.fetch_add(1, Ordering::Relaxed);
                tx
            })
            .collect();
        MatmulPool { senders }
    })
}

/// Cached thread budget for `Tensor::matmul`. Resolved on first call and held
/// for the rest of the process. The env-var override (`BITNET_MATMUL_THREADS`)
/// lets callers opt in to per-matmul threading.
///
/// Default is **1** (serial matmul). At the v0.9 model scale every matmul
/// has m*n in the 1k-16k element range, which is small enough that the cost
/// of spawning 4-8 OS threads via `std::thread::scope` (~10-30 us each)
/// dominates the actual compute. Parallelism here is best expressed at the
/// outer batch-of-windows level (`TrainConfig.n_workers`) where each unit of
/// work is a full forward+backward pass and the spawn overhead amortises.
///
/// Set `BITNET_MATMUL_THREADS=4` (or similar) explicitly when:
///   - the model is large enough that a single matmul exceeds ~100k
///     elements (roughly hidden_dim 256+ at seq_len 64+, or larger ffn);
///   - the outer training loop is single-threaded for some other reason
///     (e.g. inference, where there are no batches to parallelise across).
fn matmul_thread_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Ok(s) = std::env::var("BITNET_MATMUL_THREADS")
            && let Ok(n) = s.parse::<usize>()
        {
            return n.max(1);
        }
        1
    })
}

/// SIMD strategy selected by `matmul_kernel`. Resolved once at process startup
/// from runtime CPU detection plus the `BITNET_MATMUL_SIMD` env-var override.
/// All three paths produce **bit-identical** output per cell because the
/// per-cell summation sequence (`kk = 0..k`) is fixed and none of them uses
/// fused multiply-add (FMA collapses two roundings into one).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MatmulSimd {
    /// 16-lane AVX-512 (`_mm512_mul_ps` + `_mm512_add_ps`). Available on Zen 4
    /// and Intel Sapphire Rapids onwards.
    Avx512,
    /// 8-lane AVX2 (`_mm256_mul_ps` + `_mm256_add_ps`). Available on every
    /// modern x86_64 CPU shipped since ~2013.
    Avx2,
    /// 4-lane ARM NEON (`vmulq_f32` + `vaddq_f32`). Baseline on every
    /// ARMv8/aarch64 CPU (Apple Silicon, Graviton, Raspberry Pi 4+), so
    /// no runtime detection is needed. Never constructed on x86_64,
    /// where the dispatcher only returns the AVX/scalar variants.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Neon,
    /// Portable scalar AXPY. LLVM may still auto-vectorise this on `--release`.
    Scalar,
}

/// Cached SIMD strategy. Selection rules (highest priority first):
///   - `BITNET_MATMUL_SIMD=off | 0 | none | scalar` -> scalar
///   - `BITNET_MATMUL_SIMD=avx2` -> AVX2 even if AVX-512 is present
///   - `BITNET_MATMUL_SIMD=avx512` -> AVX-512 if present, overriding the
///     Zen 4 auto-demotion below (lets us A/B the two widths without
///     recompiling)
///   - default -> widest that actually wins: AVX-512, except on Zen 4 where
///     its double-pumped 512-bit path is ~9% slower on this bandwidth-bound
///     matmul so AVX2 is picked instead; then AVX2 -> scalar
///
/// Reading the env var (and CPUID) once via `OnceLock` keeps the hot dispatch
/// branch-free.
#[allow(clippy::needless_return)]
fn matmul_simd_mode() -> MatmulSimd {
    static M: OnceLock<MatmulSimd> = OnceLock::new();
    *M.get_or_init(|| {
        let env = std::env::var("BITNET_MATMUL_SIMD")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase());
        if matches!(env.as_deref(), Some("off" | "0" | "none" | "scalar")) {
            return MatmulSimd::Scalar;
        }
        let force_avx2 = matches!(env.as_deref(), Some("avx2"));
        let force_avx512 = matches!(env.as_deref(), Some("avx512" | "avx-512"));
        #[cfg(target_arch = "x86_64")]
        {
            // AVX-512 is the widest path but not always the fastest: on Zen 4
            // it is double-pumped and this matmul is memory-bandwidth bound, so
            // it measured ~9% slower than AVX2. Auto-demote there unless the
            // caller explicitly forces AVX-512 for A/B timing.
            let avx512_preferred = force_avx512 || !is_zen4();
            if !force_avx2 && avx512_preferred && std::is_x86_feature_detected!("avx512f") {
                return MatmulSimd::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return MatmulSimd::Avx2;
            }
        }
        // The forcing flags are x86-only; consume them on other arches.
        let _ = (force_avx2, force_avx512);
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is architecturally mandatory on aarch64: no runtime
            // detection, no forcing flag needed - only `off`/`scalar`
            // (handled above) opts out.
            return MatmulSimd::Neon;
        }
        #[allow(unreachable_code)]
        MatmulSimd::Scalar
    })
}

/// True on AMD Zen 4 (CPU family `0x19`), whose 512-bit AVX ops are double-
/// pumped; on this bandwidth-bound matmul that makes the AVX-512 kernel ~9%
/// slower than AVX2 (measured v0.17 on a Ryzen 9 7940HS). Zen 5 (family `0x1A`)
/// has a native full-width datapath and is faster at AVX-512, so it is
/// deliberately excluded.
///
/// `std` exposes CPU *features* but not vendor/family, so we read CPUID leaf 0
/// for the vendor string and leaf 1 for the family field directly.
#[cfg(target_arch = "x86_64")]
fn is_zen4() -> bool {
    use std::arch::x86_64::__cpuid;
    // Leaves 0 and 1 are architectural on every x86_64 CPU, so `__cpuid` is safe.
    let vendor = __cpuid(0);
    // "AuthenticAMD" is packed into EBX, EDX, ECX in that order.
    let is_amd = vendor.ebx == 0x6874_7541 // "Auth"
        && vendor.edx == 0x6974_6e65 // "enti"
        && vendor.ecx == 0x444d_4163; // "cAMD"
    is_amd && decode_family(__cpuid(1).eax) == 0x19
}

/// Decode the effective CPU family from CPUID leaf 1 EAX. The 4-bit base family
/// is bits [11:8]; when it is `0xF` the 8-bit extended family in bits [27:20]
/// is added (AMD Zen reports base `0xF` + ext `0x0A` = `0x19`). Split out as a
/// pure function so the bit math is unit-testable without the target hardware.
#[cfg(target_arch = "x86_64")]
fn decode_family(eax: u32) -> u32 {
    let base = (eax >> 8) & 0xf;
    if base == 0xf {
        base + ((eax >> 20) & 0xff)
    } else {
        base
    }
}

/// Compute `out += lhs @ rhs` (m,k) @ (k,n) -> (m,n) via the AXPY-ordered
/// kernel. The output buffer must be zero-initialised on entry; the kernel
/// accumulates into it. Dispatches to the widest SIMD path the CPU supports
/// (AVX-512 -> AVX2 -> scalar), subject to the `BITNET_MATMUL_SIMD` override.
///
/// Every path visits `(i, kk, j)` in the same order, and per-output-cell the
/// summation sequence is `kk = 0, 1, ..., k - 1` - identical to the classic
/// `acc += lhs[i,kk] * rhs[kk,j]` triple loop. The bit pattern of every
/// output element therefore matches the textbook implementation regardless
/// of which SIMD width was used.
fn matmul_kernel(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    match matmul_simd_mode() {
        // Safety: each path is gated by runtime feature detection; the AVX2
        // and AVX-512 kernels only use unaligned load/store intrinsics whose
        // pointer offsets stay strictly within the per-row slices.
        MatmulSimd::Avx512 => unsafe {
            matmul_kernel_avx512(lhs, rhs, out, m, k, n);
        },
        MatmulSimd::Avx2 => unsafe {
            matmul_kernel_avx2(lhs, rhs, out, m, k, n);
        },
        // Neon is never selected on x86_64; the arm exists because the
        // enum is shared across arches.
        MatmulSimd::Scalar | MatmulSimd::Neon => matmul_kernel_scalar(lhs, rhs, out, m, k, n),
    }
    #[cfg(target_arch = "aarch64")]
    match matmul_simd_mode() {
        // Safety: NEON is baseline on aarch64; the kernel only uses
        // unaligned load/store intrinsics whose pointer offsets stay
        // strictly within the per-row slices.
        MatmulSimd::Neon => unsafe {
            matmul_kernel_neon(lhs, rhs, out, m, k, n);
        },
        _ => matmul_kernel_scalar(lhs, rhs, out, m, k, n),
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    matmul_kernel_scalar(lhs, rhs, out, m, k, n);
}

/// NEON kernel (issue #6): 4 f32 per inner-loop iteration. Bit-identical
/// to the scalar kernel because each output lane accumulates the same
/// `kk = 0..k` sequence of `mul` then `add` operations, and
/// `vmulq_f32` / `vaddq_f32` are IEEE-754 per-lane equivalents of scalar
/// `f32 * +`. We deliberately do *not* use `vfmaq_f32`: FMA collapses
/// the two roundings into one, breaking bit-equality with the scalar
/// and x86 paths.
///
/// # Safety
/// Caller must pass `lhs.len() >= m*k`, `rhs.len() >= k*n`,
/// `out.len() >= m*n`. Inside, every pointer offset is bounded by the
/// row's length and `j + 4 <= n` where applicable. (NEON itself needs
/// no detection: it is baseline on aarch64.)
#[cfg(target_arch = "aarch64")]
unsafe fn matmul_kernel_neon(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    let n_simd = (n / 4) * 4; // largest multiple of 4 <= n

    for i in 0..m {
        let row_lhs_off = i * k;
        let row_out_off = i * n;
        for kk in 0..k {
            let a = lhs[row_lhs_off + kk];
            // Safety: every intrinsic and pointer offset below is bounded
            // by the row's length; the function's preconditions guarantee
            // lhs / rhs / out are large enough.
            unsafe {
                let a_vec = vdupq_n_f32(a);
                let rhs_base = rhs.as_ptr().add(kk * n);
                let out_base = out.as_mut_ptr().add(row_out_off);

                let mut j = 0;
                while j < n_simd {
                    let out_vec = vld1q_f32(out_base.add(j));
                    let rhs_vec = vld1q_f32(rhs_base.add(j));
                    let prod = vmulq_f32(a_vec, rhs_vec);
                    let sum = vaddq_f32(out_vec, prod);
                    vst1q_f32(out_base.add(j), sum);
                    j += 4;
                }
                // Tail: n - n_simd remaining columns, scalar.
                while j < n {
                    *out_base.add(j) += a * *rhs_base.add(j);
                    j += 1;
                }
            }
        }
    }
}

/// Portable scalar AXPY-ordered kernel. The inner `j` loop is the per-row
/// AXPY `out_row[j] += a * rhs_row[j]`. LLVM may auto-vectorise this when
/// `-C target-cpu` enables SSE / AVX, but that is opportunistic; the
/// explicit AVX2 path above is the deterministic SIMD route.
fn matmul_kernel_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let lhs_row = &lhs[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for kk in 0..k {
            let a = lhs_row[kk];
            let rhs_row = &rhs[kk * n..(kk + 1) * n];
            for j in 0..n {
                out_row[j] += a * rhs_row[j];
            }
        }
    }
}

/// AVX2 kernel: 8 f32 per inner-loop iteration. Bit-identical to the scalar
/// kernel because each output lane accumulates the same `kk = 0..k` sequence
/// of `mul` then `add` operations, and `_mm256_{mul,add}_ps` are IEEE-754
/// per-lane equivalents of scalar `f32 * +`. We deliberately do *not* use
/// `_mm256_fmadd_ps` because FMA collapses the two roundings into one,
/// breaking bit-equality with scalar mul + add.
///
/// # Safety
/// Caller must have detected AVX2 (`is_x86_feature_detected!("avx2")`) and
/// pass `lhs.len() >= m*k`, `rhs.len() >= k*n`, `out.len() >= m*n`. Inside,
/// every pointer offset is bounded by the row's length and `j + 8 <= n`
/// where applicable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_kernel_avx2(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    let n_simd = (n / 8) * 8; // largest multiple of 8 <= n

    for i in 0..m {
        let row_lhs_off = i * k;
        let row_out_off = i * n;
        for kk in 0..k {
            let a = lhs[row_lhs_off + kk];
            // Safety: every intrinsic and pointer offset below is bounded
            // by the row's length; the function's preconditions guarantee
            // lhs / rhs / out are large enough. The 2024 edition requires
            // explicit `unsafe` blocks even inside an `unsafe fn`.
            unsafe {
                let a_vec = _mm256_set1_ps(a);
                let rhs_base = rhs.as_ptr().add(kk * n);
                let out_base = out.as_mut_ptr().add(row_out_off);

                let mut j = 0;
                while j < n_simd {
                    let out_vec = _mm256_loadu_ps(out_base.add(j));
                    let rhs_vec = _mm256_loadu_ps(rhs_base.add(j));
                    let prod = _mm256_mul_ps(a_vec, rhs_vec);
                    let sum = _mm256_add_ps(out_vec, prod);
                    _mm256_storeu_ps(out_base.add(j), sum);
                    j += 8;
                }
                // Tail: n - n_simd remaining columns, scalar.
                while j < n {
                    *out_base.add(j) += a * *rhs_base.add(j);
                    j += 1;
                }
            }
        }
    }
}

/// AVX-512 kernel: 16 f32 per inner-loop iteration. Bit-identical to the
/// AVX2 and scalar kernels because each output lane accumulates the same
/// `kk = 0..k` sequence of `mul` then `add` operations and `_mm512_{mul,add}_ps`
/// are IEEE-754 per-lane equivalents of scalar `f32 * +`. As with the AVX2
/// path we deliberately avoid `_mm512_fmadd_ps` because FMA collapses two
/// roundings into one and would diverge from the other paths.
///
/// On Zen 4 the 512-bit FP units are double-pumped 256-bit (so the
/// per-cycle throughput is the same as AVX2), but the doubled lane count
/// halves the inner-loop trip count and the per-iteration scalar overhead
/// (loop bookkeeping, `_mm512_set1_ps`, base pointer maths). On Intel
/// Sapphire Rapids and later the 512-bit units are native and the speedup
/// over AVX2 is closer to the theoretical 2x.
///
/// # Safety
/// Caller must have detected AVX-512 foundation (`is_x86_feature_detected!("avx512f")`)
/// and pass `lhs.len() >= m*k`, `rhs.len() >= k*n`, `out.len() >= m*n`. Inside,
/// every pointer offset is bounded by the row's length and `j + 16 <= n` for
/// the SIMD loop body; the tail loop runs scalar within `j < n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn matmul_kernel_avx512(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
    };

    let n_simd = (n / 16) * 16; // largest multiple of 16 <= n

    for i in 0..m {
        let row_lhs_off = i * k;
        let row_out_off = i * n;
        for kk in 0..k {
            let a = lhs[row_lhs_off + kk];
            // Safety: same reasoning as `matmul_kernel_avx2`. All offsets
            // are within the row slices guaranteed by the preconditions; the
            // unaligned 512-bit load/store intrinsics need only readable /
            // writable memory at the offsets, no alignment.
            unsafe {
                let a_vec = _mm512_set1_ps(a);
                let rhs_base = rhs.as_ptr().add(kk * n);
                let out_base = out.as_mut_ptr().add(row_out_off);

                let mut j = 0;
                while j < n_simd {
                    let out_vec = _mm512_loadu_ps(out_base.add(j));
                    let rhs_vec = _mm512_loadu_ps(rhs_base.add(j));
                    let prod = _mm512_mul_ps(a_vec, rhs_vec);
                    let sum = _mm512_add_ps(out_vec, prod);
                    _mm512_storeu_ps(out_base.add(j), sum);
                    j += 16;
                }
                // Tail: n - n_simd remaining columns (0..15), scalar so the
                // bit pattern matches the AVX2 / scalar paths exactly.
                while j < n {
                    *out_base.add(j) += a * *rhs_base.add(j);
                    j += 1;
                }
            }
        }
    }
}

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
    /// Triple loop with output-row sharding when the result is large enough to
    /// pay for thread-spawn overhead (see `MATMUL_PARALLEL_THRESHOLD`). Cache
    /// blocking and SIMD would speed this up further; deferred until after
    /// the parallel layer is well-understood.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k, n) = self.matmul_dims(other);
        let threads = matmul_thread_count();
        if threads <= 1 || m * n < MATMUL_PARALLEL_THRESHOLD {
            self.matmul_serial_inner(&other.data, m, k, n)
        } else {
            self.matmul_parallel_inner(&other.data, m, k, n, threads)
        }
    }

    /// Validate that two tensors are matmul-compatible and return `(m, k, n)`.
    /// Shared by every public matmul entry point so the error messages and
    /// dimension extraction stay consistent.
    fn matmul_dims(&self, other: &Tensor) -> (usize, usize, usize) {
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
        (m, k, n)
    }

    /// Single-threaded matmul reference. Exposed for tests so the parallel
    /// path's bit-identity guarantee can be asserted directly. `cargo test`
    /// for the binary crate doesn't see the regular `pub` use sites, so this
    /// is gated behind `#[cfg(test)]` to keep release builds free of dead-code
    /// noise.
    #[cfg(test)]
    pub fn matmul_serial(&self, other: &Tensor) -> Tensor {
        let (m, k, n) = self.matmul_dims(other);
        self.matmul_serial_inner(&other.data, m, k, n)
    }

    /// Matmul with an explicit thread count. Test/benchmark hook for pinning
    /// behaviour; production code goes through `matmul`, which reads the
    /// process-wide thread budget once and caches it.
    #[cfg(test)]
    pub fn matmul_with_threads(&self, other: &Tensor, threads: usize) -> Tensor {
        let (m, k, n) = self.matmul_dims(other);
        if threads <= 1 {
            self.matmul_serial_inner(&other.data, m, k, n)
        } else {
            self.matmul_parallel_inner(&other.data, m, k, n, threads)
        }
    }

    /// Single-threaded matmul. Allocates a zero-initialised output buffer and
    /// delegates to the shared AXPY-ordered kernel (scalar or AVX2 depending
    /// on runtime detection).
    fn matmul_serial_inner(&self, rhs: &[f32], m: usize, k: usize, n: usize) -> Tensor {
        let mut out = vec![0.0f32; m * n];
        matmul_kernel(&self.data, rhs, &mut out, m, k, n);
        Tensor {
            data: out,
            shape: vec![m, n],
        }
    }

    /// Row-sharded matmul on the long-lived worker pool (issue #7).
    /// Splits the output rows into `threads` contiguous bands - the
    /// same banding as the old spawn-per-call path, so output stays
    /// bit-identical to `matmul_serial_inner` (same kernel, same
    /// per-cell accumulation order, band boundaries do not change
    /// per-cell maths). Band 0 runs on the calling thread; bands
    /// 1..threads are round-robined onto the pool workers via one
    /// channel send each (~1-5 us vs ~10-30 us per OS thread spawn).
    ///
    /// Memory safety: `AckDrain` is constructed before any job is
    /// dispatched and only counts jobs whose `send` succeeded. Its
    /// `Drop` blocks until every dispatched band acks, so `out`, `lhs`
    /// and `rhs` (declared before it, dropped after it) can never be
    /// freed while a worker still holds pointers into them - even if
    /// this thread panics mid-call.
    fn matmul_parallel_inner(
        &self,
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        threads: usize,
    ) -> Tensor {
        let rows_per_thread = m.div_ceil(threads);
        let mut out = vec![0.0f32; m * n];
        let lhs: &[f32] = &self.data;
        let pool = matmul_pool();
        let (ack_tx, ack_rx) = mpsc::channel::<()>();
        let mut drain = AckDrain {
            rx: &ack_rx,
            remaining: 0,
        };

        let mut bands = out.chunks_mut(rows_per_thread * n).enumerate();
        // Detach band 0 for the calling thread; dispatch the rest first
        // so the workers overlap with the local compute.
        let local_band = bands.next();
        for (b_idx, chunk) in bands {
            let row_start = b_idx * rows_per_thread;
            let chunk_rows = chunk.len() / n;
            let job = MatmulJob {
                lhs: lhs[row_start * k..].as_ptr(),
                rhs: rhs.as_ptr(),
                out: chunk.as_mut_ptr(),
                rows: chunk_rows,
                k,
                n,
                ack: ack_tx.clone(),
            };
            pool.senders[(b_idx - 1) % pool.senders.len()]
                .send(job)
                .expect("matmul pool: worker channel closed");
            drain.remaining += 1;
        }
        // Drop the local ack sender so `recv` can observe worker death
        // (channel error) instead of hanging, should a worker ever die
        // mid-job. With it gone, only in-flight jobs hold senders.
        drop(ack_tx);
        if let Some((_, chunk)) = local_band {
            let chunk_rows = chunk.len() / n;
            matmul_kernel(&lhs[..chunk_rows * k], rhs, chunk, chunk_rows, k, n);
        }
        drop(drain); // blocks until every dispatched band has finished

        Tensor {
            data: out,
            shape: vec![m, n],
        }
    }

    /// The pre-pool implementation: spawn fresh scoped OS threads per
    /// call. Kept (test-only) as the A/B baseline for the issue #7
    /// microbenchmark and as a second bit-identity witness.
    #[cfg(test)]
    fn matmul_parallel_scoped_inner(
        &self,
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        threads: usize,
    ) -> Tensor {
        let rows_per_thread = m.div_ceil(threads);
        let mut out = vec![0.0f32; m * n];
        let lhs: &[f32] = &self.data;

        std::thread::scope(|s| {
            // chunks_mut yields disjoint &mut [f32] slices into `out`. Each
            // chunk is moved into a spawn closure with a `'scope` lifetime;
            // the borrow on `out` outlives the scope so no aliasing risk.
            for (t_idx, chunk) in out.chunks_mut(rows_per_thread * n).enumerate() {
                let row_start = t_idx * rows_per_thread;
                let chunk_rows = chunk.len() / n;
                let lhs_band = &lhs[row_start * k..(row_start + chunk_rows) * k];
                s.spawn(move || {
                    matmul_kernel(lhs_band, rhs, chunk, chunk_rows, k, n);
                });
            }
        });

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

    /// Per-row softmax over the last axis. Input shape `[m, n]`. Uses
    /// the max-subtraction trick: `exp(x - max(row))` keeps every
    /// exponent in `[exp(-inf), exp(0)]`, so finite logits cannot
    /// produce overflow even when their absolute magnitudes are large.
    /// Three passes per row (max, sum-of-exps, normalise); the final
    /// loop multiplies by `1/denom` rather than dividing because
    /// scalar division is the most expensive primitive on most CPUs.
    pub fn softmax(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "softmax: expected rank-2, got rank {}",
            self.ndim()
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut s = vec![0.0_f32; m * n];
        for i in 0..m {
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..n {
                let v = self.data[i * n + j];
                if v > row_max {
                    row_max = v;
                }
            }
            let mut denom = 0.0_f32;
            for j in 0..n {
                let e = (self.data[i * n + j] - row_max).exp();
                s[i * n + j] = e;
                denom += e;
            }
            let inv = 1.0_f32 / denom;
            for j in 0..n {
                s[i * n + j] *= inv;
            }
        }
        Tensor {
            data: s,
            shape: vec![m, n],
        }
    }

    /// Causal mask: set `out[i, j] = -inf` for `j > i`, leave the lower
    /// triangle (incl. diagonal) untouched. Input shape `[seq, seq]`.
    /// Applied to attention scores before softmax so a query at row `i`
    /// cannot attend to keys at columns `> i`.
    pub fn causal_mask(&self) -> Tensor {
        self.causal_mask_window(self.shape[0])
    }

    /// Block-diagonal-causal mask (issue #22): cell `(i, j)` survives
    /// iff `j <= i` AND `i / window == j / window`, so a batched
    /// `[B*seq, B*seq]` score matrix cannot attend across windows.
    /// `window == rows` reduces to the classic causal mask.
    pub fn causal_mask_window(&self, window: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "causal_mask: expected rank-2, got rank {}",
            self.ndim()
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = self.data.clone();
        for i in 0..m {
            for j in 0..n {
                if j > i || j / window != i / window {
                    data[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    /// Per-row softmax backward (Phase 4 chunk 4.3). `self` is the
    /// upstream gradient `grad_y`, `s_out` is the saved softmax
    /// forward output. Implements the JVP of `J = diag(s) - s s^T`:
    ///
    ///     dot_i             = sum_k grad_y[i, k] * s_out[i, k]
    ///     grad_in[i, j]     = s_out[i, j] * (grad_y[i, j] - dot_i)
    ///
    /// Matches the `Var::softmax` closure (autograd.rs:618-621)
    /// cell-for-cell.
    pub fn softmax_backward(&self, s_out: &Tensor) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "softmax_backward: rank-2 only, got rank {}",
            self.ndim()
        );
        assert_eq!(self.shape, s_out.shape, "softmax_backward: shape mismatch");
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut grad_in = vec![0.0_f32; m * n];
        for i in 0..m {
            let dot: f32 = (0..n)
                .map(|k| self.data[i * n + k] * s_out.data[i * n + k])
                .sum();
            for j in 0..n {
                grad_in[i * n + j] = s_out.data[i * n + j] * (self.data[i * n + j] - dot);
            }
        }
        Tensor {
            data: grad_in,
            shape: vec![m, n],
        }
    }

    /// Causal-mask backward (Phase 4 chunk 4.3). `self` is the upstream
    /// gradient. Lower triangle (`j <= i`) passes through unchanged;
    /// upper triangle is zeroed (the forward overwrote those cells with
    /// `-inf`, so they contribute no gradient to the input). No saved
    /// tensor is needed - the mask pattern is shape-determined.
    /// Matches `Var::causal_mask` (autograd.rs:805-811).
    pub fn causal_mask_backward(&self) -> Tensor {
        self.causal_mask_backward_window(self.shape[0])
    }

    /// Backward of `causal_mask_window` (issue #22): upstream gradient
    /// passes only where the forward kept the cell.
    pub fn causal_mask_backward_window(&self, window: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "causal_mask_backward: rank-2 only, got rank {}",
            self.ndim()
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut grad_in = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..(i + 1).min(n) {
                if j / window == i / window {
                    grad_in[i * n + j] = self.data[i * n + j];
                }
            }
        }
        Tensor {
            data: grad_in,
            shape: vec![m, n],
        }
    }

    /// Per-row RMS normalisation (no learnable gain). Input `[m, n]`,
    /// output same shape. Each row gets divided by its RMS magnitude:
    ///     rms_i = sqrt(mean_j(x[i, j]^2) + EPS),    EPS = 1e-5
    ///     y[i, j] = x[i, j] / rms_i
    /// Matches the autograd `Var::rmsnorm` math exactly so checkpoints
    /// trained with the existing CPU forward stay numerically valid
    /// when later evaluated through the trait-based path.
    pub fn rmsnorm(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "rmsnorm: rank-2 only, got rank {}",
            self.ndim()
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let n_f = n as f32;
        const EPS: f32 = 1e-5;
        let mut y = vec![0.0_f32; m * n];
        for i in 0..m {
            let mean_sq: f32 = (0..n).map(|j| self.data[i * n + j].powi(2)).sum::<f32>() / n_f;
            let inv = 1.0_f32 / (mean_sq + EPS).sqrt();
            for j in 0..n {
                y[i * n + j] = self.data[i * n + j] * inv;
            }
        }
        Tensor {
            data: y,
            shape: vec![m, n],
        }
    }

    /// Sigmoid Linear Unit activation: `silu(x) = x / (1 + exp(-x))`.
    /// Smooth, differentiable everywhere. Used as SwiGLU's gate branch.
    pub fn silu(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|&x| {
                let sig = 1.0_f32 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// SiLU backward (Phase 4 chunk 4.2). `self` is the upstream
    /// gradient `grad_y`; `x` is the saved forward input. Per-cell
    /// derivative `d(silu)/dx = sig * (1 + x * (1 - sig))` matches the
    /// CPU `Var::silu` closure (autograd.rs:579-590) byte-for-byte.
    pub fn silu_backward(&self, x: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, x.shape,
            "silu_backward: grad_y and x shape mismatch"
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&x.data)
            .map(|(&grad, &xv)| {
                let sig = 1.0_f32 / (1.0 + (-xv).exp());
                grad * sig * (1.0 + xv * (1.0 - sig))
            })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Rotary Position Embedding (RoPE). Input shape `[seq, head_dim]`;
    /// `head_dim` must be even. For each `(pos, i)` rotates the 2-D
    /// vector `(x[pos, 2i], x[pos, 2i+1])` by `pos * 10000^(-2i/head_dim)`.
    /// Parameter-free; trig table is recomputed per call (cheap, and
    /// avoids cache state for the inference path).
    pub fn rope(&self) -> Tensor {
        self.rope_period(self.shape[0])
    }

    /// RoPE with an explicit rotation period (issue #22): position is
    /// `row % period`, so a `[B*seq, head_dim]` slab rotates each
    /// window by 0..seq. `period == rows` is classic RoPE.
    pub fn rope_period(&self, period: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "rope: expected rank-2, got rank {}",
            self.ndim()
        );
        let (seq, head_dim) = (self.shape[0], self.shape[1]);
        assert!(
            head_dim % 2 == 0,
            "rope: head_dim ({}) must be even",
            head_dim
        );
        let half = head_dim / 2;
        let mut y = vec![0.0_f32; seq * head_dim];
        for pos in 0..seq {
            for i in 0..half {
                let theta_i = 10000_f32.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = (pos % period) as f32 * theta_i;
                let c = angle.cos();
                let s = angle.sin();
                let a = self.data[pos * head_dim + 2 * i];
                let b = self.data[pos * head_dim + 2 * i + 1];
                y[pos * head_dim + 2 * i] = a * c - b * s;
                y[pos * head_dim + 2 * i + 1] = a * s + b * c;
            }
        }
        Tensor {
            data: y,
            shape: vec![seq, head_dim],
        }
    }

    /// Per-row RMSNorm backward (Phase 4 chunk 4.4). `self` is the
    /// upstream gradient `grad_y`; `x_saved` is the saved forward
    /// input. Recomputes `rms_i` from `x_saved` rather than carrying
    /// a separate `[m]` tensor of saved row norms (cost: one extra
    /// row pass; benefit: keeps the API symmetric with the other
    /// "saved-tensor" backwards). Per-cell formula matches
    /// `Var::rmsnorm` (autograd.rs:702-758) byte-for-byte.
    pub fn rmsnorm_backward(&self, x_saved: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "rmsnorm_backward: rank-2 only");
        assert_eq!(
            self.shape, x_saved.shape,
            "rmsnorm_backward: shape mismatch"
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let n_f = n as f32;
        const EPS: f32 = 1e-5;
        let mut grad_in = vec![0.0_f32; m * n];
        for i in 0..m {
            let mean_sq: f32 = (0..n).map(|j| x_saved.data[i * n + j].powi(2)).sum::<f32>() / n_f;
            let inv_rms = 1.0_f32 / (mean_sq + EPS).sqrt();
            let dot: f32 = (0..n)
                .map(|j| x_saved.data[i * n + j] * self.data[i * n + j])
                .sum();
            let factor = dot * inv_rms.powi(3) / n_f;
            for j in 0..n {
                grad_in[i * n + j] =
                    self.data[i * n + j] * inv_rms - x_saved.data[i * n + j] * factor;
            }
        }
        Tensor {
            data: grad_in,
            shape: vec![m, n],
        }
    }

    /// BitNet absmean-ternary weight quantisation, STE forward
    /// (Phase 5.a). Output is the **dequantised** tensor `gamma * W_q`
    /// where `W_q` is in {-1, 0, +1} and `gamma = mean(|W|)`. Same f32
    /// shape as input. Matches `Var::quantise_weights_ste`
    /// (autograd.rs:404-436) byte-for-byte. Backward is identity (the
    /// STE), so callers route the upstream gradient straight through
    /// the original (pre-quant) weight without a separate backward
    /// op.
    pub fn quantise_weights_ste(&self) -> Tensor {
        let (w_q, gamma) = crate::bitlinear::absmean_ternary(self);
        let data: Vec<f32> = w_q.data.iter().map(|v| v * gamma).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// BitNet absmax-INT8 per-row activation quantisation, STE
    /// forward (Phase 5.a). Output is `(alpha[i] / 127) * x_q[i, j]`
    /// where `x_q` lives on the INT8 grid `[-128, 127]`. Matches
    /// `Var::quantise_acts_ste` (autograd.rs:438-474). Backward is
    /// identity (STE).
    pub fn quantise_acts_ste(&self) -> Tensor {
        let (x_q, alpha) = crate::bitlinear::absmax_int8(self);
        let (m, n) = (x_q.shape[0], x_q.shape[1]);
        let inv_127 = 1.0_f32 / 127.0;
        let mut data = vec![0.0_f32; m * n];
        for i in 0..m {
            let row_scale = alpha.data[i] * inv_127;
            for j in 0..n {
                data[i * n + j] = x_q.data[i * n + j] * row_scale;
            }
        }
        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    /// Fused softmax + cross-entropy forward (Phase 4 chunk 4.5.d).
    /// `self` is the logits `[seq, vocab]`; `targets` is the per-row
    /// class index. Returns `(loss_scalar, softmax_output)` in one
    /// pass over the data. Matches `Var::cross_entropy` (autograd.rs:
    /// 1025-1115) byte-for-byte (subtract-max log-sum-exp trick,
    /// mean-over-seq loss).
    pub fn cross_entropy_forward_save(&self, targets: &[usize]) -> (f32, Tensor) {
        assert_eq!(self.ndim(), 2, "cross_entropy: rank-2 logits required");
        let (seq, vocab) = (self.shape[0], self.shape[1]);
        assert_eq!(targets.len(), seq, "cross_entropy: target len mismatch");
        let mut softmax = vec![0.0_f32; seq * vocab];
        let mut total_loss = 0.0_f32;
        for i in 0..seq {
            let row = &self.data[i * vocab..(i + 1) * vocab];
            let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0_f32;
            for j in 0..vocab {
                let e = (row[j] - row_max).exp();
                softmax[i * vocab + j] = e;
                denom += e;
            }
            let log_denom = row_max + denom.ln();
            let inv = 1.0_f32 / denom;
            for v in &mut softmax[i * vocab..(i + 1) * vocab] {
                *v *= inv;
            }
            assert!(
                targets[i] < vocab,
                "cross_entropy: target {} >= vocab {}",
                targets[i],
                vocab
            );
            total_loss += -(self.data[i * vocab + targets[i]] - log_denom);
        }
        let loss = total_loss / seq as f32;
        (
            loss,
            Tensor {
                data: softmax,
                shape: vec![seq, vocab],
            },
        )
    }

    /// Cross-entropy backward (Phase 4 chunk 4.5.d). `self` is the
    /// saved softmax output from `cross_entropy_forward_save`.
    /// Per-cell formula `(softmax - onehot) / seq` matches the
    /// autograd closure (autograd.rs:1093-1106) with `g_scalar = 1`.
    pub fn cross_entropy_backward(&self, targets: &[usize], seq: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "cross_entropy_backward: rank-2 softmax required"
        );
        let (seq_chk, vocab) = (self.shape[0], self.shape[1]);
        assert_eq!(seq, seq_chk, "cross_entropy_backward: seq mismatch");
        assert_eq!(
            targets.len(),
            seq,
            "cross_entropy_backward: target len mismatch"
        );
        let inv_seq = 1.0_f32 / seq as f32;
        let mut grad = self.data.clone();
        for i in 0..seq {
            grad[i * vocab + targets[i]] -= 1.0;
        }
        for v in grad.iter_mut() {
            *v *= inv_seq;
        }
        Tensor {
            data: grad,
            shape: vec![seq, vocab],
        }
    }

    /// RoPE backward (Phase 4 chunk 4.4). `self` is the upstream
    /// gradient. Inverse rotation: same trig table with `sin` flipped
    /// because each per-pair rotation is orthogonal. No saved tensor:
    /// the angles depend only on shape. Matches `Var::rope`
    /// (autograd.rs:902-925).
    pub fn rope_backward(&self) -> Tensor {
        self.rope_backward_period(self.shape[0])
    }

    /// Backward of `rope_period` (issue #22): inverse rotation at
    /// `row % period`.
    pub fn rope_backward_period(&self, period: usize) -> Tensor {
        assert_eq!(self.ndim(), 2, "rope_backward: expected rank-2");
        let (seq, head_dim) = (self.shape[0], self.shape[1]);
        assert!(
            head_dim % 2 == 0,
            "rope_backward: head_dim ({}) must be even",
            head_dim
        );
        let half = head_dim / 2;
        let mut grad_in = vec![0.0_f32; seq * head_dim];
        for pos in 0..seq {
            for i in 0..half {
                let theta_i = 10000_f32.powf(-(2.0 * i as f32) / head_dim as f32);
                let angle = (pos % period) as f32 * theta_i;
                let c = angle.cos();
                let s = angle.sin();
                let ga = self.data[pos * head_dim + 2 * i];
                let gb = self.data[pos * head_dim + 2 * i + 1];
                grad_in[pos * head_dim + 2 * i] = ga * c + gb * s;
                grad_in[pos * head_dim + 2 * i + 1] = -ga * s + gb * c;
            }
        }
        Tensor {
            data: grad_in,
            shape: vec![seq, head_dim],
        }
    }
}

// ---- CPU `device::*` trait impls. Each delegates to the corresponding
// inherent method above (or to the existing pre-Phase-2 op like `add`).
// Lives at module scope per Rust's trait impl rules.

impl crate::device::MatMul for Tensor {
    fn matmul(&self, rhs: &Self) -> Self {
        Tensor::matmul(self, rhs)
    }
}

impl crate::device::Add for Tensor {
    fn add(&self, rhs: &Self) -> Self {
        Tensor::add(self, rhs)
    }
}

impl crate::device::MulScalar for Tensor {
    fn mul_scalar(&self, s: f32) -> Self {
        Tensor::mul_scalar(self, s)
    }
}

impl crate::device::Transpose2D for Tensor {
    fn transpose_2d(&self) -> Self {
        Tensor::transpose_2d(self)
    }
}

impl crate::device::Softmax for Tensor {
    fn softmax(&self) -> Self {
        Tensor::softmax(self)
    }
}

impl crate::device::CausalMask for Tensor {
    fn causal_mask(&self, window: usize) -> Self {
        Tensor::causal_mask_window(self, window)
    }
}

impl crate::device::Rope for Tensor {
    fn rope(&self, period: usize) -> Self {
        Tensor::rope_period(self, period)
    }
}

impl crate::device::Silu for Tensor {
    fn silu(&self) -> Self {
        Tensor::silu(self)
    }
}

impl crate::device::SiluBackward for Tensor {
    fn silu_backward(&self, x: &Self) -> Self {
        Tensor::silu_backward(self, x)
    }
}

impl crate::device::SoftmaxBackward for Tensor {
    fn softmax_backward(&self, s_out: &Self) -> Self {
        Tensor::softmax_backward(self, s_out)
    }
}

impl crate::device::CausalMaskBackward for Tensor {
    fn causal_mask_backward(&self, window: usize) -> Self {
        Tensor::causal_mask_backward_window(self, window)
    }
}

impl crate::device::RmsNormBackward for Tensor {
    fn rmsnorm_backward(&self, x_saved: &Self) -> Self {
        Tensor::rmsnorm_backward(self, x_saved)
    }
}

impl crate::device::RopeBackward for Tensor {
    fn rope_backward(&self, period: usize) -> Self {
        Tensor::rope_backward_period(self, period)
    }
}

impl crate::device::CrossEntropy for Tensor {
    fn cross_entropy_forward_save(&self, targets: &[usize]) -> (f32, Self) {
        Tensor::cross_entropy_forward_save(self, targets)
    }
}

impl crate::device::CrossEntropyBackward for Tensor {
    fn cross_entropy_backward(&self, targets: &[usize], seq: usize) -> Self {
        Tensor::cross_entropy_backward(self, targets, seq)
    }
}

impl crate::device::QuantiseWeightsSTE for Tensor {
    fn quantise_weights_ste(&self) -> Self {
        Tensor::quantise_weights_ste(self)
    }
}

impl crate::device::QuantiseActsSTE for Tensor {
    fn quantise_acts_ste(&self) -> Self {
        Tensor::quantise_acts_ste(self)
    }
}

impl crate::device::BitLinear for Tensor {
    fn bit_linear(&self, rhs: &Self) -> Self {
        // Forward via the existing matmul on quantised inputs.
        let x_eff = Tensor::quantise_acts_ste(self);
        let w_eff = Tensor::quantise_weights_ste(rhs);
        x_eff.matmul(&w_eff)
    }
}

impl crate::device::Mul for Tensor {
    fn mul(&self, rhs: &Self) -> Self {
        Tensor::mul(self, rhs)
    }
}

impl crate::device::RmsNorm for Tensor {
    fn rmsnorm(&self) -> Self {
        Tensor::rmsnorm(self)
    }
}

#[cfg(test)]
// Entire module compiled out of release / `cargo run` build; only `cargo test` sees it
mod tests {
    use super::*; // pulls `Tensor` into the test module's scope

    /// Decode the effective CPU family from representative CPUID leaf-1 EAX
    /// values. Machine-independent: exercises the base/extended-family math
    /// `is_zen4` relies on without needing the actual silicon.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode_family_matches_known_cpuid_values() {
        // AMD Zen 4: base 0xF + ext 0x0A -> 0x19 (this EAX also decodes to
        // model 0x74, i.e. a Phoenix 7940HS).
        assert_eq!(decode_family(0x00A7_0F40), 0x19);
        // AMD Zen 5: ext 0x0B -> 0x1A, must NOT read as Zen 4.
        assert_eq!(decode_family(0x00B0_0F00), 0x1A);
        // AMD Zen / Zen+: ext 0x08 -> 0x17.
        assert_eq!(decode_family(0x0080_0F11), 0x17);
        // Intel Skylake: base family 6, no extended-family addend.
        assert_eq!(decode_family(0x0009_06ED), 0x06);
    }

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
    fn parallel_matmul_matches_serial_bit_identical() {
        // Generate deterministic inputs with enough rows for at least 4 shards
        // and a hidden dim large enough that float-accumulation order matters
        // if the implementation got it wrong. The inner-loop order is identical
        // between serial and parallel paths, so equality is exact (not approx).
        let m = 64;
        let k = 128;
        let n = 256;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.041).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let serial = lhs.matmul_serial(&rhs);
        // Parallel path with several thread counts. All must equal serial bit-for-bit.
        for threads in [2usize, 3, 4, 7, 8] {
            let parallel = lhs.matmul_with_threads(&rhs, threads);
            assert_eq!(
                parallel.shape, serial.shape,
                "shape mismatch at threads={}",
                threads
            );
            assert_eq!(
                parallel.data, serial.data,
                "parallel matmul (threads={}) drifted from serial; shard maths is wrong",
                threads
            );
        }
    }

    #[test]
    fn parallel_matmul_reuses_pool_workers() {
        // Issue #7: the parallel path must run on a long-lived worker
        // pool, not fresh OS threads per call. The pool's lifetime
        // spawn counter must not move after warm-up, no matter how many
        // parallel matmuls run.
        let (m, k, n) = (64usize, 32usize, 64usize); // m*n over the threshold
        let lhs = Tensor::from_vec(
            (0..m * k).map(|i| (i as f32 * 0.3).sin()).collect(),
            vec![m, k],
        );
        let rhs = Tensor::from_vec(
            (0..k * n).map(|i| (i as f32 * 0.7).cos()).collect(),
            vec![k, n],
        );
        let serial = lhs.matmul_serial(&rhs);

        let _ = lhs.matmul_with_threads(&rhs, 4); // warm-up: pool spawns here
        let spawned_after_warmup = matmul_pool_spawned_workers();
        assert!(spawned_after_warmup >= 1, "pool spawned no workers");

        for _ in 0..100 {
            let pooled = lhs.matmul_with_threads(&rhs, 4);
            assert_eq!(
                pooled.data, serial.data,
                "pooled matmul drifted from serial"
            );
        }
        assert_eq!(
            matmul_pool_spawned_workers(),
            spawned_after_warmup,
            "workers were respawned: pool is not long-lived"
        );
    }

    /// Issue #7 microbench: pooled dispatch vs the old spawn-per-call
    /// scoped path, at a training-realistic shape. Not a correctness
    /// gate (timing-dependent), so ignored by default. Run with:
    /// `cargo test --release matmul_pool_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark, run explicitly with --ignored --nocapture"]
    fn matmul_pool_bench_vs_scoped() {
        let (m, k, n) = (64usize, 128usize, 256usize); // seq x hidden @ hidden x ffn
        let lhs = Tensor::from_vec(
            (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect(),
            vec![m, k],
        );
        let rhs = Tensor::from_vec(
            (0..k * n).map(|i| (i as f32 * 0.041).cos()).collect(),
            vec![k, n],
        );
        let threads = 8usize;
        let iters = 2000usize;

        // Same maths on both paths, or the timing is meaningless.
        let pooled_out = lhs.matmul_with_threads(&rhs, threads);
        let scoped_out = lhs.matmul_parallel_scoped_inner(&rhs.data, m, k, n, threads);
        assert_eq!(pooled_out.data, scoped_out.data, "paths diverged");

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = lhs.matmul_with_threads(&rhs, threads);
        }
        let pooled = t0.elapsed();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = lhs.matmul_parallel_scoped_inner(&rhs.data, m, k, n, threads);
        }
        let scoped = t0.elapsed();

        println!(
            "matmul [{m}x{k}]@[{k}x{n}], {threads} threads, {iters} iters:\n  pooled: {:>8.1} us/call\n  scoped: {:>8.1} us/call\n  speedup: x{:.2}",
            pooled.as_secs_f64() * 1e6 / iters as f64,
            scoped.as_secs_f64() * 1e6 / iters as f64,
            scoped.as_secs_f64() / pooled.as_secs_f64(),
        );
    }

    #[test]
    fn parallel_matmul_handles_m_smaller_than_thread_count() {
        // When `m` is smaller than the requested thread count we should still
        // produce the correct result; some threads simply do less work or no
        // work at all. The chunks_mut iterator naturally caps at `m` rows.
        let lhs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let serial = lhs.matmul_serial(&rhs);
        // Force the parallel branch by asking for 8 threads; 2 rows means 2
        // shards with 1 row each, the remaining 6 thread slots are unused.
        let parallel = lhs.matmul_with_threads(&rhs, 8);
        assert_eq!(parallel.data, serial.data);
        assert_eq!(parallel.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn axpy_kernel_matches_classic_triple_loop_bit_identical() {
        // Hand-roll the textbook (i, j, kk) inner-product matmul and compare
        // against `Tensor::matmul`, which now goes through the AXPY kernel
        // (scalar or AVX2). Both must agree to byte-equality because the
        // per-output-cell summation order is identical: kk = 0, 1, ..., k-1.
        // Primes for m, k, n exercise both the SIMD body (n >= 8) and the
        // tail path (n % 8 != 0).
        let m = 17usize;
        let k = 23usize;
        let n = 31usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();

        let mut classic = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += lhs_data[i * k + kk] * rhs_data[kk * n + j];
                }
                classic[i * n + j] = acc;
            }
        }

        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);
        let result = lhs.matmul(&rhs);

        assert_eq!(result.shape, vec![m, n]);
        for (idx, (&got, &want)) in result.data.iter().zip(&classic).enumerate() {
            assert_eq!(
                got,
                want,
                "AXPY/SIMD drift at idx {} (i = {}, j = {})",
                idx,
                idx / n,
                idx % n
            );
        }
    }

    #[test]
    fn parallel_axpy_with_simd_tail_matches_serial() {
        // Force every shard count down to 1 row and pick prime n to push
        // through the SIMD-tail path (n % 8 != 0). If chunks_mut indexing
        // or the lhs-band split gets the per-thread bounds wrong, this
        // test catches it before training does.
        let m = 13usize;
        let k = 17usize;
        let n = 19usize;
        let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.111).sin()).collect();
        let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.222).cos()).collect();
        let lhs = Tensor::from_vec(lhs_data, vec![m, k]);
        let rhs = Tensor::from_vec(rhs_data, vec![k, n]);

        let serial = lhs.matmul_serial(&rhs);
        for threads in [2usize, 3, 4, 7, 13] {
            let parallel = lhs.matmul_with_threads(&rhs, threads);
            assert_eq!(
                parallel.data, serial.data,
                "drift between serial and parallel at threads = {}",
                threads
            );
        }
    }

    /// Hand-rolled textbook (i, j, kk) inner-product matmul. Re-used by the
    /// per-kernel direct-call tests below. Returns the row-major output.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn classic_triple_loop_reference(
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += lhs[i * k + kk] * rhs[kk * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn axpy_kernel_avx2_direct_matches_classic_triple_loop_bit_identical() {
        // The high-level `axpy_kernel_matches_classic_triple_loop_bit_identical`
        // test goes through `Tensor::matmul`, which dispatches via the
        // `OnceLock`-cached `matmul_simd_mode()`. On Zen 4 / Sapphire Rapids
        // the OnceLock now selects AVX-512, so the AVX2 path stops being
        // exercised by that test on those machines. This test pins the AVX2
        // kernel by calling it directly. Primes (m=17, k=23, n=31) exercise
        // both the 8-wide SIMD body and the n % 8 = 7 tail.
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: AVX2 not available on this CPU");
            return;
        }
        let m = 17usize;
        let k = 23usize;
        let n = 31usize;
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let want = classic_triple_loop_reference(&lhs, &rhs, m, k, n);

        let mut got = vec![0.0f32; m * n];
        // Safety: AVX2 was detected above; slice lengths satisfy m*k, k*n,
        // m*n preconditions by construction.
        unsafe { matmul_kernel_avx2(&lhs, &rhs, &mut got, m, k, n) };
        assert_eq!(got, want, "AVX2 kernel diverged from textbook triple loop");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn axpy_kernel_neon_direct_matches_classic_triple_loop_bit_identical() {
        // Mirror of the AVX2/AVX-512 direct tests for the issue #6 NEON
        // kernel. Primes (m=17, k=23, n=31) exercise both the 4-wide
        // SIMD body and the n % 4 = 3 tail. NEON is baseline on
        // aarch64, so no runtime detection guard is needed.
        let m = 17usize;
        let k = 23usize;
        let n = 31usize;
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let want = classic_triple_loop_reference(&lhs, &rhs, m, k, n);

        let mut got = vec![0.0f32; m * n];
        // Safety: slice lengths satisfy the m*k, k*n, m*n preconditions
        // by construction.
        unsafe { matmul_kernel_neon(&lhs, &rhs, &mut got, m, k, n) };
        assert_eq!(got, want, "NEON kernel diverged from textbook triple loop");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn axpy_kernel_avx512_direct_matches_classic_triple_loop_bit_identical() {
        // Mirror of the AVX2 direct test for the v0.17 AVX-512 kernel.
        // Primes m=17, k=23, n=37 deliberately push past 32 columns so the
        // 16-wide SIMD body runs twice and leaves a 5-element scalar tail
        // (37 % 16 = 5). The bit-equality assertion catches both ordering
        // bugs (loop transposition) and SIMD numerical drift (any
        // accidental FMA, mismatched lane widths, etc.).
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: AVX-512 foundation not available on this CPU");
            return;
        }
        let m = 17usize;
        let k = 23usize;
        let n = 37usize;
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.219).cos()).collect();
        let want = classic_triple_loop_reference(&lhs, &rhs, m, k, n);

        let mut got = vec![0.0f32; m * n];
        // Safety: AVX-512 foundation was detected above; slice lengths
        // satisfy the kernel's m*k, k*n, m*n preconditions by construction.
        unsafe { matmul_kernel_avx512(&lhs, &rhs, &mut got, m, k, n) };
        assert_eq!(
            got, want,
            "AVX-512 kernel diverged from textbook triple loop"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_and_avx512_kernels_produce_byte_identical_output() {
        // Headline v0.17 guarantee: switching SIMD widths must not change a
        // single output bit. Also catches the case where AVX2 happens to be
        // bit-identical to scalar via lucky cancellation but AVX-512 picks
        // up a real divergence (or vice versa). Prime n=23 hits the n % 8 =
        // 7 tail of AVX2 *and* the n % 16 = 7 tail of AVX-512, so both tail
        // paths are exercised in the same call.
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: needs both AVX2 and AVX-512 foundation");
            return;
        }
        let m = 13usize;
        let k = 17usize;
        let n = 23usize;
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.4321).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.7654).cos()).collect();

        let mut from_avx2 = vec![0.0f32; m * n];
        let mut from_avx512 = vec![0.0f32; m * n];
        // Safety: both features were detected above; slice lengths satisfy
        // the kernels' m*k, k*n, m*n preconditions by construction.
        unsafe {
            matmul_kernel_avx2(&lhs, &rhs, &mut from_avx2, m, k, n);
            matmul_kernel_avx512(&lhs, &rhs, &mut from_avx512, m, k, n);
        }
        assert_eq!(
            from_avx2, from_avx512,
            "AVX2 and AVX-512 kernels diverged at the byte level"
        );
    }

    #[test]
    fn matmul_with_one_thread_is_serial() {
        // Sanity: matmul_with_threads(_, 1) must hit the serial path. Not
        // strictly observable from the result (output is bit-identical
        // either way) but documents the threshold dispatch contract.
        let lhs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        assert_eq!(
            lhs.matmul_with_threads(&rhs, 1).data,
            lhs.matmul_serial(&rhs).data
        );
    }

    #[test]
    fn elementwise_sub() {
        let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], vec![3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(a.sub(&b).data, vec![4.0, 5.0, 6.0]);
    }
}
