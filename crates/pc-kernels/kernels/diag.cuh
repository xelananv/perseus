// The device diagonal encoder's gather (args.cuh DiagGatherArgs): the slot vector of a generated diagonal read straight
// from its tile (a linear's resident row-major float64 matrix); encode_many (encode.cuh) then encodes the batch.
// Plaintext (n, m, gg, jj) of a DiagLinear over n_slots = B t slots (B x B blocks, b x g steps, k = gg b + jj) is
// np.roll(np.repeat(blk[(i + k) % B, i], t), gg b t), blk the (n, m) block (the port's filling.LazyDiagonals). The roll
// moves element e to e + gg b t, so slot v holds element e = (v - gg b t) mod n_slots of the repeated diagonal, i.e.
// entry i = e / t of the diagonal: W[n B + ((i + k) mod B)][m B + i]. The same arithmetic is pc-program
// compile/diag.rs TileInfo::slot / addr and the port's filling.diag_slot_index, both unit-tested on the CPU.
#pragma once
#include "arith.cuh"

namespace pc {
namespace diag {
constexpr int BX = 256;

// grid (N/2 / BX, njobs): thread s of block row y writes slot s of job y (coalesced: consecutive threads, consecutive
// double2). The t consecutive slots of one diagonal entry read one address: a warp of 32 slots touches max(1, 32 / t)
// elements of the tile (one at the model's t = 32).
__global__ void __launch_bounds__(BX)
gather(const DiagJobRow* __restrict__ jobs, double2* __restrict__ v, uint64_t vs) {
    const uint32_t s = blockIdx.x * BX + threadIdx.x;
    const uint32_t ns = (uint32_t)(N / 2);  // n_slots, a power of two dividing 2^32
    if (s >= ns) return;
    const DiagJobRow J = jobs[blockIdx.y];
    const uint32_t gb = J.gg * J.b;                              // gg b < B
    // (s - gg b t) mod n_slots: the u32 difference wraps mod 2^32, a multiple of n_slots, so the mask is the mod
    const uint32_t i = ((s - (gb << J.log_t)) & (ns - 1)) >> J.log_t;   // < n_slots / t = B
    uint32_t r = i + gb + J.jj;                                   // i + k < 2B: one subtract is the mod B
    if (r >= J.side) r -= J.side;
    // (n B + r) d_out + m B + i in 64 bits: n B + r < d_in and m B + i < d_out (the host range-checked the row)
    const size_t at = ((size_t)J.n * J.side + r) * (size_t)J.d_out + (size_t)J.m * J.side + i;
    const double* __restrict__ w = reinterpret_cast<const double*>(J.w);
    v[(size_t)blockIdx.y * vs + s] = make_double2(__ldg(w + at), 0.0);
}

// Device traffic per generated plaintext at N = 2^16 (n_slots 2^15), L limbs of W = 4-byte words, B = 1,024:
//   gather   the tile's B entries of the diagonal, 8 B = 8 KiB (32 KiB moved: consecutive entries are d_out + 1
//            elements apart, one 32-byte sector each), and 16 n_slots = 512 KiB of slots written;
//   encode   place 0.5 MiB in, 1 MiB out; 16 FFT passes x 2 MiB = 32 MiB; coeffs 1 MiB in, 0.5 MiB out; scatter
//            0.5 MiB in, 4 L N = 0.25 L MiB out: with the gather about 36 MiB + 0.25 L MiB;
//   NTT      two passes over L rows, each reading and writing them: about L MiB;
// so about 36 + 1.25 L MiB per plaintext (61 MiB at L = 20) before its product (a mul_table and an add, read by
// pc-gpu Ops). Launches per chunk of nb jobs: gather 1, encode_many 19 kernels and 1 memset, NTT 2 per 128 rows of
// nb L; products 1 + 2 (nb - 1) for the first chunk of a sum, 2 nb for the others (pc-gpu diag_gen.rs).
int launch(const DiagGatherArgs* a, cudaStream_t s) {
    if (a->njobs < 1 || a->njobs > 65535 || a->v_stride < (uint64_t)N || (a->v_stride & 1u) || a->jobs == nullptr)
        return -1;
    const dim3 g((N / 2 + BX - 1) / BX, a->njobs);
    gather<<<g, BX, 0, s>>>(a->jobs, (double2*)a->v, a->v_stride / 2);
    return (int)cudaGetLastError();
}
}  // namespace diag
}  // namespace pc
