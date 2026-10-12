// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_encode.h — CKKS plaintexts for the port's weight diagonals made on the device.
//
// A diagonal plaintext of the filled layout is block-constant: channel i's value over t = n/d
// consecutive slots, rolled by g*s*t (examples/fhe_ssm_from_primitives/filling.py). Encoding it on
// the CPU costs 2-4 ms (FFT-bound) for every one of the ~345k products of a tick; here the slot
// vector never exists on the host: a kernel gathers it from the resident folded matrix straight
// into a batched cuFFT buffer, cuFFT does the length-n transform, and one fused kernel twists,
// scales, rounds and reduces the result into a FIDESlib RNSPoly's limbs, which the library then
// NTTs. The acceptance test is tests/dev/gpu_encode_probe.cu; the limb order and the plaintext
// wrap were measured on the library by tests/dev/gpu_encode_layout_probe.cu.
//
//   c_k = (1/n) FFT_n(Y)_k zeta^(-k),   Y_v = z_j,  v = (5^j mod 4n - 1)/4,   m_k = llround(sf Re c_k),
//   m_{k+n} = llround(sf Im c_k),   residue_l = m mod q_l (sign in 64-bit, as openfhe-n32 patch 0005),
//   written in natural coefficient order (the order the library's own import uses).
//
// Streams. K1 -> cuFFT -> K2 run on one of two non-blocking encode streams (batch parity), so the
// next group's encode overlaps the current group's products. The plaintext polys live on the legacy
// default stream (Plaintext's c0 is built with def_stream = true), where their limbs are pooled,
// their pointer tables uploaded and their NTTs launched; the legacy stream also orders after every
// blocking stream (the ciphertext ops), so one event recorded there after the grows is the barrier
// K2 waits for (a recycled limb is never written while a pending product still reads it), and K2's
// completion event is what the legacy stream waits for before the NTTs. That contract holds only
// while the library's ciphertext streams are blocking ones; the bridge checks the stream of the
// ciphertext the products will run on before the first batch and refuses otherwise.
//
// Context layout. FIDESlib is built with -DNCCL and its consumers are not, so ContextData's fields
// after `precom` sit at different offsets in this translation unit than in the library's
// (Context.cuh:164). This file reads `param` (declared before the shift) and the limbs' own plain
// fields only.
#pragma once

#include "gpu_cuda_raii.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace FIDESlib::CKKS {
class Plaintext;
class ContextData;
using Context = std::shared_ptr<ContextData>;
}  // namespace FIDESlib::CKKS

namespace gpu_encode {

// A resident tile's id: the encoder's generation in the high half, the slot index in the low half,
// so an id kept across release_tiles() is refused instead of selecting whatever was registered next.
using TileId = uint64_t;

// Per-batch job: B diagonals of giant index g, each from one tile (several tiles share the babies in
// linear_multi / down). All at one (level, sf). B <= kMaxBatch.
struct GroupJob {
    std::vector<TileId> tiles;  // size B: tiles[b] is the tile of batch entry b
    std::vector<int> baby;      // size B: the baby index b' of entry b (0 <= b' < s)
    int g = 0;                  // giant index
    int s = 0;                  // baby steps (32 for d = 1024)
    int level = 0;              // the plaintexts' level (OpenFHE's: primes dropped, as encode_like_params gives it)
    double sf = 0.0;            // ScalingFactorReal(level)
    bool skip_ntt = false;      // test hook: leave the limbs in COEFF form (parity against the CPU encode)
};

constexpr int kMaxBatch = 256;  // W_k + W_r: 8 output tiles x 32 babies share one set of babies
constexpr int kMaxLimbs = 64;   // FIDESlib's MAXP

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

    // Upload a folded d x d tile (row-major, y = x @ W: the diagonal (g, b) at channel i reads
    // W[(i + g*s + b) mod d][i]) once; the id is what encode_group reads from.
    TileId register_tile(const std::vector<double>& W_row_major, int d);
    // Free every resident tile; ids issued before this call are refused from here on. Never throws
    // (it runs from destructors): a context error at that point is reported on stderr, not thrown.
    void release_tiles() noexcept;
    std::size_t tile_count() const { return tiles_.size(); }
    // The resident tile back on the host (row-major d x d; `d` is set): the CPU route's source
    // when a group cannot take the device route.
    std::vector<double> download_tile(TileId id, int& d) const;

    // Encode one batch: K1 -> cuFFT -> K2 on the encode stream of this batch's parity, then the
    // library's NTT on the polys' (legacy default) stream; returns B device plaintexts (EVAL form,
    // NoiseFactor = sf, NoiseLevel 1, slots n), all ready when last_ready_event() has fired.
    std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> encode_group(const GroupJob& job);

    // The event recorded after the batch's NTTs (for cudaStreamWaitEvent on any other stream).
    cudaEvent_t last_ready_event() const { return ready_[(cur_ + 1) & 1].get(); }

    // Counters for the timing report. Device-side stage times are accumulated only when
    // FHE_GPU_ENCODE_TIMING=1 (the host then waits for every batch, so the overlap is lost).
    struct Stats {
        uint64_t plaintexts = 0, batches = 0;
        double k1_ms = 0, fft_ms = 0, k2_ms = 0, ntt_ms = 0;
    };
    const Stats& stats() const { return stats_; }

  private:
    struct Tile {
        DevicePtr<double> W;  // device, d*d doubles
        int d = 0;
        double max_abs = 0.0;  // bounds every coefficient the tile can produce: |c_k| <= max|W|
    };
    struct PlanRec {
        int batch;
        int parity;
        FftPlan plan;
    };

    const Tile& tile(TileId id) const;
    cufftHandle plan_for(int batch, int parity);

    FIDESlib::CKKS::Context& ctx_;
    FIDESlib::CKKS::ContextData& cc_;
    int n_, d_, t_, log2t_, device_;
    bool timing_ = false, debug_sync_ = false;
    // Members are destroyed bottom-up: the plans (which reference the work areas) go first, then
    // the buffers, then the streams and events; the destructor body drains the streams before that.
    Stream s_enc_[2];
    Event alloc_done_[2];  // legacy stream: limbs pooled + pointer tables uploaded
    Event enc_done_[2];    // K2 finished on s_enc[i]
    Event ready_[2];       // NTTs finished on the polys' stream
    Event tm_[2][7];       // timing marks (FHE_GPU_ENCODE_TIMING=1)
    DevicePtr<uint16_t> invperm_;          // device, n
    DevicePtr<cufftDoubleComplex> twist_;  // device, n
    DevicePtr<cufftDoubleComplex> in_[2];  // device, kMaxBatch * n each
    DevicePtr<unsigned char> work_[2];     // cuFFT work areas (bytes), one per parity
    std::size_t work_bytes_[2] = {0, 0};
    std::vector<Tile> tiles_;
    uint32_t generation_ = 1;
    std::vector<PlanRec> plans_;
    uint64_t cur_ = 0;
    Stats stats_;
};

}  // namespace gpu_encode
