// SPDX-License-Identifier: MIT
// gpu_encode.h — CKKS plaintexts for the port's weight diagonals made on the device (R1, 2026-10-11).
//
// A diagonal plaintext of the filled layout is block-constant: channel i's value over t = n/d consecutive
// slots, rolled by g*s*t (examples/fhe_ssm_from_primitives/filling.py). Encoding it on the CPU costs
// 2-4 ms (FFT-bound) for every one of the ~345k products of a tick; here the slot vector never exists
// on the host: a kernel gathers it from the resident folded matrix straight into a batched cuFFT
// buffer, cuFFT does the length-n transform, and one fused kernel twists, scales, rounds and reduces
// the result into a FIDESlib RNSPoly's limbs, which the library then NTTs. Maths, layout and the
// validation ladder: results/perseus-20261006/fork_port/gpu_20261011/r1/DESIGN.md (the owner's tree).
//
//   c_k = (1/n) FFT_n(Y)_k zeta^(-k),   Y_v = z_j,  v = (5^j mod 4n - 1)/4,   m_k = llround(sf Re c_k),
//   m_{k+n} = llround(sf Im c_k),   residue_l = m mod q_l (sign in 64-bit, as openfhe-n32 patch 0005),
//   written in natural coefficient order (parity step 0: r1/layout_probe.out).
//
// Streams. K1 -> cuFFT -> K2 run on one of two non-blocking encode streams (batch parity), so the next
// group's encode overlaps the current group's products. The plaintext polys live on the legacy default
// stream (Plaintext's c0 is built with def_stream = true), where their limbs are pooled, their pointer
// tables uploaded and their NTTs launched; the legacy stream also orders after every blocking stream
// (the ciphertext ops), so one event recorded there after the grows is the barrier K2 waits for (a
// recycled limb is never written while a pending product still reads it), and K2's completion event is
// what the legacy stream waits for before the NTTs.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace FIDESlib::CKKS {
class Plaintext;
class ContextData;
using Context = std::shared_ptr<ContextData>;
}  // namespace FIDESlib::CKKS

namespace gpu_encode {

// One d x d tile of a folded weight matrix, resident on the device as float64 (y = x @ W convention:
// W[row][col], the diagonal (g, b) at channel i reads W[(i + g*s + b) mod d][i]).
struct TileHandle {
    const double* W = nullptr;  // device, d*d doubles
    int d = 0;
};

// Per-batch job: B diagonals of giant index g, each from one tile (several tiles share the babies in
// linear_multi / down). All at one (level, sf). B <= kMaxBatch.
struct GroupJob {
    std::vector<TileHandle> tiles;  // size B: tiles[b] is the tile of batch entry b
    std::vector<int> baby;          // size B: the baby index b' of entry b (0 <= b' < s)
    int g = 0;                      // giant index
    int s = 0;                      // baby steps (32 for d = 1024)
    int level = 0;                  // the plaintexts' level (OpenFHE's: primes dropped, as encode_like_params gives it)
    double sf = 0.0;                // ScalingFactorReal(level)
    bool skip_ntt = false;          // test hook: leave the limbs in COEFF form (parity against the CPU encode)
};

constexpr int kMaxBatch = 256;      // W_k + W_r: 8 output tiles x 32 babies share one set of babies
constexpr int kMaxLimbs = 64;       // FIDESlib's MAXP

// The tables and the resident tiles' owner. One per session (one device, one context).
class GpuEncoder {
  public:
    // n slots (N/2), d channels; builds invperm[n] (uint16: j for each v), twist[n] ((1/n) zeta^(-k)),
    // two in-place cuFFT buffers of kMaxBatch x n cufftDoubleComplex, two non-blocking encode streams
    // and their events. cuFFT plans (cufftPlanMany, 1-D, Z2Z, contiguous) are made per (batch size,
    // parity) on first use, with one explicit work area per parity.
    GpuEncoder(FIDESlib::CKKS::Context& ctx, int n, int d);
    ~GpuEncoder();
    GpuEncoder(const GpuEncoder&) = delete;
    GpuEncoder& operator=(const GpuEncoder&) = delete;

    // Upload a folded d x d tile once; the handle is what encode_group reads from.
    TileHandle register_tile(const std::vector<double>& W_row_major, int d);
    void release_tile(TileHandle& h);

    // Encode one batch: K1 -> cuFFT -> K2 on the encode stream of this batch's parity, then the
    // library's NTT on the polys' (legacy default) stream; returns B device plaintexts (EVAL form,
    // NoiseFactor = sf, NoiseLevel 1, slots n), all ready when last_ready_event() has fired.
    std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> encode_group(const GroupJob& job);

    // The event recorded after the batch's NTTs (for cudaStreamWaitEvent on any other stream).
    cudaEvent_t last_ready_event() const { return ready_[(cur_ + 1) & 1]; }

    // Counters for the timing report. Device-side stage times are accumulated only when
    // FHE_GPU_ENCODE_TIMING=1 (the host then waits for every batch, so the overlap is lost).
    struct Stats {
        uint64_t plaintexts = 0, batches = 0;
        double k1_ms = 0, fft_ms = 0, k2_ms = 0, ntt_ms = 0;
    };
    const Stats& stats() const { return stats_; }

  private:
    cufftHandle plan_for(int batch, int parity);

    FIDESlib::CKKS::Context& ctx_;
    FIDESlib::CKKS::ContextData& cc_;
    int n_, d_, t_, log2t_, device_;
    uint16_t* invperm_ = nullptr;          // device, n
    cufftDoubleComplex* twist_ = nullptr;  // device, n
    cufftDoubleComplex* in_[2] = {nullptr, nullptr};  // device, kMaxBatch * n each
    void* work_[2] = {nullptr, nullptr};              // cuFFT work areas, one per parity
    size_t work_bytes_[2] = {0, 0};
    cudaStream_t s_enc_[2] = {nullptr, nullptr};
    cudaEvent_t alloc_done_[2] = {nullptr, nullptr};  // legacy stream: limbs pooled + pointer tables uploaded
    cudaEvent_t enc_done_[2] = {nullptr, nullptr};    // K2 finished on s_enc[i]
    cudaEvent_t ready_[2] = {nullptr, nullptr};       // NTTs finished on the polys' stream
    cudaEvent_t tm_[2][7] = {{nullptr}};              // timing marks (FHE_GPU_ENCODE_TIMING=1)
    struct PlanRec { int batch; int parity; cufftHandle h; };
    std::vector<PlanRec> plans_;
    uint64_t cur_ = 0;
    bool timing_ = false, debug_sync_ = false;
    Stats stats_;
};

}  // namespace gpu_encode
