// SPDX-License-Identifier: MIT
// gpu_encode.cu — see include/gpu_encode.h. DRAFT FOR REVIEW: plan/buffer initialisation and kernel
// signatures; the bodies marked TODO follow the review of the memory layout.
#include "gpu_encode.h"

#include <CKKS/Plaintext.cuh>
#include <CKKS/RNSPoly.cuh>
#include <ConstantsGPU.cuh>   // C_ (primes, prime_mu64, prime_bits in __constant__)
#include <ModMult.cuh>        // modreduce_lazy(uint64_t, primeid): exact 64-bit Barrett

#include <cmath>
#include <stdexcept>

namespace gpu_encode {

namespace {

constexpr int kThreads = 256;

// ---- K1: gather + expand + permute ------------------------------------------------------------
// grid (n / kThreads, B), block kThreads. Thread (v, b): j = invperm[v] is the slot whose value lands
// at FFT input position v; the slot holds channel i = ((j - g*s*t) mod n) >> log2(t) of diagonal
// (g, baby[b]), i.e. W_b[(i + g*s + baby[b]) mod d][i]. Writes are coalesced in v; the reads hit a
// d-entry column of one 8 MiB tile (L1/L2-resident across the block).
__global__ void k_gather_expand_permute(const double* const* __restrict__ tiles,  // [B] device ptrs
                                        const int* __restrict__ baby,              // [B]
                                        int g, int s, int d, int log2t, int n,
                                        const uint16_t* __restrict__ invperm,      // [n]
                                        cufftDoubleComplex* __restrict__ in)       // [B][n]
{
    // TODO(review): body
}

// ---- K2: twist + scale + round + reduce, fused -------------------------------------------------
// grid (n / kThreads, B), block kThreads. Thread (k, b): c = in[b][k] * twist[k] * sf;
// re = llround(c.x), im = llround(c.y) stay in registers while the limb loop writes
// limb[b][l][k] = red(re, prime[l]) and limb[b][l][k + n] = red(im, prime[l]) — natural COEFF order
// (parity step 0 decides whether the store index is bit-reversed for FIDESlib's NTT).
// red(x) = x >= 0 ? modreduce_lazy(x) : (m = modreduce_lazy(-x), m ? q - m : 0).
__global__ void k_twist_scale_round_reduce(const cufftDoubleComplex* __restrict__ in,     // [B][n]
                                           const cufftDoubleComplex* __restrict__ twist,  // [n]
                                           double sf, int n, int nlimbs,
                                           const int* __restrict__ primeids,              // [nlimbs]
                                           uint32_t* const* __restrict__ limb_ptrs)       // [B][nlimbs]
{
    // TODO(review): body
}

}  // namespace

// ---- construction: tables, buffers, plans, streams --------------------------------------------
GpuEncoder::GpuEncoder(FIDESlib::CKKS::ContextData& cc, int n, int d) : cc_(cc), n_(n), d_(d) {
    if (n <= 0 || d <= 0 || n % d) throw std::invalid_argument("gpu_encode: n must be a multiple of d");
    t_ = n / d;
    log2t_ = 0;
    while ((1 << log2t_) < t_) ++log2t_;
    if ((1 << log2t_) != t_) throw std::invalid_argument("gpu_encode: n/d must be a power of two");

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
    // TODO(review): cudaMalloc + cudaMemcpy for invperm_, twist_; cudaMalloc in_[2] (kMaxBatch*n complex each)
    //               and limb_ptrs_[2]; cudaStreamCreateWithFlags(s_enc_[i], NonBlocking); events.
    (void)inv;
    (void)tw;
}

cufftHandle GpuEncoder::plan_for(int batch) {
    for (auto& [b, h] : plans_)
        if (b == batch) return h;
    cufftHandle h;
    int nn = n_;
    // 1-D, contiguous input and output, batch transforms of length n, double complex, in place
    if (cufftPlanMany(&h, 1, &nn, /*inembed*/ nullptr, /*istride*/ 1, /*idist*/ nn,
                      /*onembed*/ nullptr, /*ostride*/ 1, /*odist*/ nn, CUFFT_Z2Z, batch) != CUFFT_SUCCESS)
        throw std::runtime_error("gpu_encode: cufftPlanMany failed");
    plans_.emplace_back(batch, h);
    return h;
}

GpuEncoder::~GpuEncoder() {
    for (auto& [b, h] : plans_) cufftDestroy(h);
    // TODO(review): free buffers, streams, events
}

TileHandle GpuEncoder::register_tile(const std::vector<double>& W, int d) {
    if (static_cast<int>(W.size()) != d * d) throw std::invalid_argument("gpu_encode: tile must be d*d");
    TileHandle h;
    h.d = d;
    // TODO(review): cudaMalloc d*d doubles, cudaMemcpy H2D (once per model)
    return h;
}

void GpuEncoder::release_tile(TileHandle& h) {
    if (h.W) cudaFree(const_cast<double*>(h.W));
    h = {};
}

std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> GpuEncoder::encode_group(const GroupJob& job) {
    // Outline (bodies after review):
    //  1. parity i = cur_ & 1; wait enc_done_[i] from two batches ago (buffer reuse).
    //  2. upload tiles[]/baby[] (B pointers + B ints) to small device arrays; K1 on s_enc_[i].
    //  3. cufftSetStream(plan_for(B), s_enc_[i]); cufftExecZ2Z(plan, in_[i], in_[i], CUFFT_FORWARD).
    //  4. for b < B: pts[b] = make_shared<Plaintext>(ctx); pts[b]->c0.grow(nlimbs - 1); collect
    //     std::get<Limb<uint32_t>>(pts[b]->c0.GPU[0].limb[l]).v.data and .primeid into the pointer table.
    //  5. K2 on s_enc_[i]; cudaEventRecord(enc_done_[i], s_enc_[i]).
    //  6. partition stream waits enc_done_[i]; for b < B: pts[b]->c0.NTT<ALGO_SHOUP>(cc_.batch, false);
    //     NoiseFactor = sf, NoiseLevel = 1, slots = n; cudaEventRecord(ready_[i], partition stream).
    //  7. ++cur_; return pts.
    (void)job;
    throw std::logic_error("gpu_encode: encode_group is a draft (bodies after review)");
}

}  // namespace gpu_encode
