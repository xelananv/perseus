// SPDX-License-Identifier: MIT
// gpu_encode.cu — see include/gpu_encode.h.
#include "gpu_encode.h"

#include <CKKS/Context.cuh>
#include <CKKS/Limb.cuh>
#include <CKKS/LimbPartition.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/RNSPoly.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>

namespace gpu_encode {

namespace {

constexpr int kThreads = 256;
constexpr const char* kWho = "gpu_encode";
// llround is defined up to 2^63 - 1; the coefficients are bounded by sf * max|W| (the twist carries
// the 1/n of the transform, so |c_k| <= max|Y| = max|W|), checked per batch against this margin.
constexpr double kCoeffBound = 9.0e18;

void check(cudaError_t e, const char* what) { cuda_check(e, kWho, what); }
void check_fft(cufftResult r, const char* what) { cufft_check(r, kWho, what); }

// Kernel parameter blocks (by value, in the constant bank; both under the 4 KB parameter limit).
struct K1Params {
    const double* tiles[kMaxBatch];  // device tile of batch entry b
    int baby[kMaxBatch];             // baby index of batch entry b
};
struct K2Params {
    void* const* tables[kMaxBatch];  // each poly's device limb-pointer table (LimbPartition::limbptr)
    uint64_t prime[kMaxLimbs];       // q_l of limb l (chain order)
    uint64_t mu64[kMaxLimbs];        // floor(2^64 / q_l): the Barrett ratio FIDESlib keeps as C_.prime_mu64
};

// ---- K1: gather + expand + permute ------------------------------------------------------------
// grid (n / kThreads, B), block kThreads. Thread (v, b): j = invperm[v] is the slot whose value lands
// at FFT input position v; the slot holds channel i = ((j - g*s*t) mod n) >> log2(t) of diagonal
// (g, baby[b]), i.e. W_b[(i + g*s + baby[b]) mod d][i]. Writes are coalesced in v; the reads gather
// 8-byte values from one 8 MiB tile (read-only path).
__global__ void k_gather_expand_permute(K1Params p, int g, int s, int d, int log2t, int n,
                                        const uint16_t* __restrict__ invperm,  // [n]
                                        cufftDoubleComplex* __restrict__ in)   // [B][n]
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (v >= n) return;
    const int j = invperm[v];
    const int shift = (g * s) << log2t;  // g*s*t < d*t = n
    int m = j - shift;
    if (m < 0) m += n;
    const int i = m >> log2t;         // channel of slot j
    int row = i + g * s + p.baby[b];  // < 2d: one conditional subtract is the mod
    if (row >= d) row -= d;
    const double* __restrict__ W = p.tiles[b];
    in[static_cast<size_t>(b) * n + v] = make_cuDoubleComplex(__ldg(W + static_cast<size_t>(row) * d + i), 0.0);
}

// ---- K2: twist + scale + round + reduce, fused -------------------------------------------------
// grid (n / kThreads, B), block kThreads. Thread (k, b): c = in[b][k] * twist[k] * sf;
// re = llround(c.x), im = llround(c.y) stay in registers while the limb loop writes
// limb[b][l][k] = red(re, q_l) and limb[b][l][k + n] = red(im, q_l) in natural COEFF order.
// red(x) = x >= 0 ? x mod q : (m = (-x) mod q; m ? q - m : 0). The reduction is the exact 64-bit
// Barrett FIDESlib uses for its lazy base conversion (ModMult.cuh modreduce_lazy, mu = floor(2^64/q):
// q_hat = floor(a mu / 2^64) is floor(a/q) or one less, so one conditional subtract finishes) — the
// constants travel in the parameter block instead of the library's __constant__ bank. No native %.
__device__ __forceinline__ uint32_t barrett64(uint64_t a, uint64_t q, uint64_t mu) {
    const uint64_t r = a - __umul64hi(a, mu) * q;
    return static_cast<uint32_t>(r >= q ? r - q : r);
}
__device__ __forceinline__ uint32_t red_signed(long long x, uint64_t q, uint64_t mu) {
    if (x >= 0) return barrett64(static_cast<uint64_t>(x), q, mu);
    const uint32_t m = barrett64(static_cast<uint64_t>(-x), q, mu);  // |x| < 2^63: the negation is defined
    return m ? static_cast<uint32_t>(q) - m : 0u;
}

__global__ void k_twist_scale_round_reduce(K2Params p, const cufftDoubleComplex* __restrict__ in,  // [B][n]
                                           const cufftDoubleComplex* __restrict__ twist,          // [n]
                                           double sf, int n, int nlimbs)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (k >= n) return;
    const cufftDoubleComplex c = in[static_cast<size_t>(b) * n + k];
    const cufftDoubleComplex w = __ldg(twist + k);
    const double re = (c.x * w.x - c.y * w.y) * sf;
    const double im = (c.x * w.y + c.y * w.x) * sf;
    const long long mre = llround(re);
    const long long mim = llround(im);
    void* const* table = p.tables[b];
    for (int l = 0; l < nlimbs; ++l) {
        uint32_t* limb = static_cast<uint32_t*>(table[l]);
        const uint64_t q = p.prime[l], mu = p.mu64[l];
        limb[k] = red_signed(mre, q, mu);
        limb[k + n] = red_signed(mim, q, mu);
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

}  // namespace

// ---- construction: tables, buffers, streams, events --------------------------------------------
GpuEncoder::GpuEncoder(FIDESlib::CKKS::Context& ctx, int n, int d) : ctx_(ctx), cc_(*ctx), n_(n), d_(d) {
    if (n <= 0 || d <= 0 || n % d) throw std::invalid_argument("gpu_encode: n must be a multiple of d");
    if (n > 65536) throw std::invalid_argument("gpu_encode: n > 65536 (invperm is uint16)");
    if (2 * n != (1 << cc_.param.logN))
        throw std::invalid_argument("gpu_encode: n must be N/2 of the context (n=" + std::to_string(n) +
                                    ", context logN=" + std::to_string(cc_.param.logN) + ")");
    t_ = n / d;
    log2t_ = 0;
    while ((1 << log2t_) < t_) ++log2t_;
    if ((1 << log2t_) != t_) throw std::invalid_argument("gpu_encode: n/d must be a power of two");
    check(cudaGetDevice(&device_), "cudaGetDevice");
    timing_ = env_on("FHE_GPU_ENCODE_TIMING");
    debug_sync_ = env_on("FHE_GPU_ENCODE_SYNC");

    // invperm: v = (5^j mod 4n - 1)/4 is a bijection on [0, n); store j for each v (uint16: n <= 65536)
    std::vector<uint16_t> inv(n);
    {
        const uint64_t M = 4ull * n;
        uint64_t p = 1;
        for (int j = 0; j < n; ++j) {
            inv[(p - 1) / 4] = static_cast<uint16_t>(j);
            p = (p * 5) % M;
        }
    }
    // twist[k] = (1/n) * exp(-2*pi*i*k/M): computed in long double on the host, stored as double
    std::vector<cufftDoubleComplex> tw(n);
    for (int k = 0; k < n; ++k) {
        const long double ang = -2.0L * 3.141592653589793238462643383279502884L * k / (4.0L * n);
        tw[k].x = static_cast<double>(std::cos(ang) / n);
        tw[k].y = static_cast<double>(std::sin(ang) / n);
    }
    invperm_ = device_alloc<uint16_t>(n, kWho, "cudaMalloc invperm");
    check(cudaMemcpy(invperm_.get(), inv.data(), sizeof(uint16_t) * n, cudaMemcpyHostToDevice), "upload invperm");
    twist_ = device_alloc<cufftDoubleComplex>(n, kWho, "cudaMalloc twist");
    check(cudaMemcpy(twist_.get(), tw.data(), sizeof(cufftDoubleComplex) * n, cudaMemcpyHostToDevice), "upload twist");
    for (int i = 0; i < 2; ++i) {
        in_[i] = device_alloc<cufftDoubleComplex>(static_cast<size_t>(kMaxBatch) * n, kWho, "cudaMalloc in");
        s_enc_[i] = Stream(cudaStreamNonBlocking, kWho);
        alloc_done_[i] = Event(cudaEventDisableTiming, kWho);
        enc_done_[i] = Event(cudaEventDisableTiming, kWho);
        ready_[i] = Event(cudaEventDisableTiming, kWho);
        for (auto& e : tm_[i]) e = Event(cudaEventDefault, kWho);
    }
}

cufftHandle GpuEncoder::plan_for(int batch, int parity) {
    for (auto& r : plans_)
        if (r.batch == batch && r.parity == parity) return r.plan.get();
    FftPlan plan = FftPlan::create(kWho);
    int nn = n_;
    size_t ws = 0;
    check_fft(cufftSetAutoAllocation(plan.get(), 0), "cufftSetAutoAllocation");
    // 1-D, contiguous input and output, `batch` transforms of length n, double complex, in place
    check_fft(cufftMakePlanMany(plan.get(), 1, &nn, /*inembed*/ nullptr, /*istride*/ 1, /*idist*/ nn,
                                /*onembed*/ nullptr, /*ostride*/ 1, /*odist*/ nn, CUFFT_Z2Z, batch, &ws),
              "cufftMakePlanMany");
    if (ws > work_bytes_[parity]) {
        // the other batch of this parity (two batches ago) may still be in flight on its work area
        check(cudaStreamSynchronize(s_enc_[parity].get()), "sync before work-area growth");
        work_[parity].reset();
        work_[parity] = device_alloc<void>(ws, kWho, "cudaMalloc cufft work area");
        work_bytes_[parity] = ws;
        for (auto& r : plans_)
            if (r.parity == parity) check_fft(cufftSetWorkArea(r.plan.get(), work_[parity].get()), "cufftSetWorkArea");
    }
    if (work_bytes_[parity]) check_fft(cufftSetWorkArea(plan.get(), work_[parity].get()), "cufftSetWorkArea");
    check_fft(cufftSetStream(plan.get(), s_enc_[parity].get()), "cufftSetStream");
    plans_.push_back({batch, parity, std::move(plan)});
    return plans_.back().plan.get();
}

GpuEncoder::~GpuEncoder() {
    for (int i = 0; i < 2; ++i)
        if (s_enc_[i]) cudaStreamSynchronize(s_enc_[i].get());  // nothing in flight reads the buffers freed below
    cudaGetLastError();  // a teardown error is not this object's to report
}

TileId GpuEncoder::register_tile(const std::vector<double>& W, int d) {
    if (d != d_) throw std::invalid_argument("gpu_encode: tile dimension differs from the encoder's d");
    if (W.size() != static_cast<size_t>(d) * d) throw std::invalid_argument("gpu_encode: tile must be d*d");
    if (tiles_.size() >= 0xffffffffu) throw std::length_error("gpu_encode: too many tiles");
    DeviceGuard guard(device_, kWho);
    Tile t;
    t.W = device_alloc<double>(W.size(), kWho, "cudaMalloc tile");
    check(cudaMemcpy(t.W.get(), W.data(), sizeof(double) * W.size(), cudaMemcpyHostToDevice), "upload tile");
    t.d = d;
    double m = 0.0;
    for (double v : W) m = std::max(m, std::fabs(v));
    t.max_abs = m;
    tiles_.push_back(std::move(t));
    return (static_cast<TileId>(generation_) << 32) | static_cast<TileId>(tiles_.size() - 1);
}

void GpuEncoder::release_tiles() {
    for (int i = 0; i < 2; ++i)
        if (s_enc_[i]) check(cudaStreamSynchronize(s_enc_[i].get()), "sync before releasing tiles");
    tiles_.clear();
    ++generation_;
}

const GpuEncoder::Tile& GpuEncoder::tile(TileId id) const {
    const uint32_t gen = static_cast<uint32_t>(id >> 32);
    const size_t idx = static_cast<size_t>(id & 0xffffffffu);
    if (gen != generation_)
        throw std::invalid_argument("gpu_encode: tile id from before release_tiles(); register the tile again");
    if (idx >= tiles_.size()) throw std::out_of_range("gpu_encode: unknown tile id");
    return tiles_[idx];
}

std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> GpuEncoder::encode_group(const GroupJob& job) {
    using FIDESlib::CKKS::Limb;
    using FIDESlib::CKKS::Plaintext;
    const int B = static_cast<int>(job.tiles.size());
    if (B < 1 || B > kMaxBatch) throw std::invalid_argument("gpu_encode: batch size out of range");
    if (job.baby.size() != job.tiles.size()) throw std::invalid_argument("gpu_encode: tiles and baby differ in length");
    if (job.s < 1 || (job.g + 1) * job.s > d_ || job.g < 0)
        throw std::invalid_argument("gpu_encode: (g, s) outside the tile");
    const int nlimbs = cc_.param.L + 1 - job.level;
    if (nlimbs < 1 || nlimbs > kMaxLimbs)
        throw std::invalid_argument("gpu_encode: level " + std::to_string(job.level) + " has no limbs");
    if (!(job.sf > 0.0)) throw std::invalid_argument("gpu_encode: scaling factor must be positive");
    K1Params k1{};
    double bound = 0.0;
    for (int b = 0; b < B; ++b) {
        const Tile& t = tile(job.tiles[b]);
        if (job.baby[b] < 0 || job.baby[b] >= job.s) throw std::invalid_argument("gpu_encode: baby index out of range");
        k1.tiles[b] = t.W.get();
        k1.baby[b] = job.baby[b];
        bound = std::max(bound, t.max_abs);
    }
    if (bound * job.sf >= kCoeffBound)
        throw std::overflow_error("gpu_encode: sf * max|W| = " + std::to_string(bound * job.sf) +
                                  " exceeds the 63-bit coefficient range (OpenFHE's encoder refuses the same)");
    DeviceGuard guard(device_, kWho);

    const int i = static_cast<int>(cur_ & 1);
    cudaStream_t se = s_enc_[i].get();
    const dim3 grid(static_cast<unsigned>((n_ + kThreads - 1) / kThreads), static_cast<unsigned>(B));

    // 1. K1 on the encode stream: in_[i] was last read by K2 of the batch two back, on this same stream.
    if (timing_) cudaEventRecord(tm_[i][0].get(), se);
    k_gather_expand_permute<<<grid, kThreads, 0, se>>>(k1, job.g, job.s, d_, log2t_, n_, invperm_.get(), in_[i].get());
    check(cudaGetLastError(), "K1 launch");
    if (debug_sync_) check(cudaStreamSynchronize(se), "K1");

    // 2. batched cuFFT, in place
    if (timing_) cudaEventRecord(tm_[i][1].get(), se);
    check_fft(cufftExecZ2Z(plan_for(B, i), in_[i].get(), in_[i].get(), CUFFT_FORWARD), "cufftExecZ2Z");
    if (timing_) cudaEventRecord(tm_[i][2].get(), se);
    if (debug_sync_) check(cudaStreamSynchronize(se), "FFT");

    // 3. the polys: limbs with aux (grow with constant = false), pooled and pointer-tabled on the
    //    legacy stream; the event recorded there afterwards also orders after every pending
    //    ciphertext op.
    std::vector<std::shared_ptr<Plaintext>> pts;
    pts.reserve(B);
    K2Params k2{};
    cudaStream_t ps = nullptr;
    for (int b = 0; b < B; ++b) {
        auto pt = std::make_shared<Plaintext>(ctx_);
        pt->c0.grow(nlimbs - 1, false, false);
        if (pt->c0.GPU.size() != 1 || pt->c0.getLevel() != nlimbs - 1)
            throw std::runtime_error("gpu_encode: poly did not grow to the requested level");
        auto& part = pt->c0.GPU[0];
        if (static_cast<int>(part.limb.size()) < nlimbs) throw std::runtime_error("gpu_encode: poly has too few limbs");
        if (b == 0) {
            ps = part.getS().ptr();
            if (!stream_checked_) {
                if (!stream_is_blocking(ps, kWho))
                    throw std::runtime_error("gpu_encode: the library's poly stream is non-blocking; the legacy-stream "
                                             "ordering this encoder relies on does not hold");
                stream_checked_ = true;
            }
            for (int l = 0; l < nlimbs; ++l) {
                if (!std::holds_alternative<Limb<uint32_t>>(part.limb[l]))
                    throw std::runtime_error("gpu_encode: the chain has a 64-bit limb; the encoder writes 32-bit "
                                             "limbs only");
                const auto& limb = std::get<Limb<uint32_t>>(part.limb[l]);
                const int pid = limb.primeid;
                if (pid != l)
                    throw std::runtime_error("gpu_encode: limb " + std::to_string(l) + " carries prime id " +
                                             std::to_string(pid));
                if (limb.v.size != 2 * n_)
                    throw std::runtime_error("gpu_encode: limb size " + std::to_string(limb.v.size) + " != N");
                if (limb.v.device != device_) throw std::runtime_error("gpu_encode: limb on another device");
                const uint64_t q = cc_.param.primes.at(pid).p;
                if (q >= (1ull << 32)) throw std::runtime_error("gpu_encode: prime wider than 32 bits");
                k2.prime[l] = q;
                k2.mu64[l] = ~0ull / q;  // floor((2^64 - 1)/q) = floor(2^64/q) for odd q > 1
            }
        } else if (part.getS().ptr() != ps) {
            throw std::runtime_error("gpu_encode: plaintext polys on different streams (unsupported)");
        }
        k2.tables[b] = part.limbptr.data;
        pt->NoiseFactor = job.sf;
        pt->NoiseLevel = 1;
        pt->slots = n_;
        pts.push_back(std::move(pt));
    }
    check(cudaEventRecord(alloc_done_[i].get(), ps), "record alloc_done");
    check(cudaStreamWaitEvent(se, alloc_done_[i].get(), 0), "wait alloc_done");

    // 4. K2 on the encode stream, then its completion event
    if (timing_) cudaEventRecord(tm_[i][3].get(), se);
    k_twist_scale_round_reduce<<<grid, kThreads, 0, se>>>(k2, in_[i].get(), twist_.get(), job.sf, n_, nlimbs);
    check(cudaGetLastError(), "K2 launch");
    if (timing_) cudaEventRecord(tm_[i][4].get(), se);
    check(cudaEventRecord(enc_done_[i].get(), se), "record enc_done");
    if (debug_sync_) check(cudaStreamSynchronize(se), "K2");

    // 5. the library's NTT (ALGO_SHOUP, the default) on the polys' stream, after K2
    check(cudaStreamWaitEvent(ps, enc_done_[i].get(), 0), "wait enc_done");
    if (timing_) cudaEventRecord(tm_[i][5].get(), ps);
    if (!job.skip_ntt)
        for (auto& pt : pts) pt->c0.NTT(cc_.param.batch, true);
    if (timing_) cudaEventRecord(tm_[i][6].get(), ps);
    check(cudaEventRecord(ready_[i].get(), ps), "record ready");
    if (debug_sync_) check(cudaDeviceSynchronize(), "NTT");

    stats_.plaintexts += B;
    stats_.batches += 1;
    if (timing_) {
        check(cudaEventSynchronize(tm_[i][6].get()), "timing sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, tm_[i][0].get(), tm_[i][1].get());
        stats_.k1_ms += ms;
        cudaEventElapsedTime(&ms, tm_[i][1].get(), tm_[i][2].get());
        stats_.fft_ms += ms;
        cudaEventElapsedTime(&ms, tm_[i][3].get(), tm_[i][4].get());
        stats_.k2_ms += ms;
        cudaEventElapsedTime(&ms, tm_[i][5].get(), tm_[i][6].get());
        stats_.ntt_ms += ms;
    }
    ++cur_;
    return pts;
}

}  // namespace gpu_encode
