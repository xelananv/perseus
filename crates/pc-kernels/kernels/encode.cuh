// CKKS packed encode on the device, as FIDESlib's encodeOnDevice (DeviceEncode.cu): slot placement at the 5^j orbit,
// radix-2 Stockham FFT in double, coefficient extraction with the running max |c|, OpenFHE's 2^61 split, and the
// per-prime scatter (exact: |m| mod p in 64-bit integer arithmetic, then one ModMul by deg_factor * 2^split).
// Output is in coefficient form; the caller NTTs it. Semantics in args.cuh EncodeArgs / EncodeManyArgs.
// One set of kernels serves both launchers: job j = blockIdx.y works on its own buffers (the job strides), so `launch`
// (encode) is `launch_many` (encode_many) at batch 1 and job j of a batch equals one encode of its buffers bit for bit.
#pragma once
#include "arith.cuh"

namespace pc {
namespace encode {
constexpr int BX = 256;

// slot j sits at zeta^{5^j}, its conjugate at zeta^{-5^j}: FFT inputs (t-1)/2 and (2N-t-1)/2, t = 5^j mod 2N
// (vs, ws: double2 elements between two jobs' v and w)
__global__ void place(const double2* v, double2* w, uint64_t vs, uint64_t ws) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N / 2) return;
    v += (size_t)blockIdx.y * vs;
    w += (size_t)blockIdx.y * ws;
    const uint32_t m2n = 2u * N - 1;
    uint32_t t = 1, b = 5, e = (uint32_t)j;
    while (e) {
        if (e & 1u) t = (t * b) & m2n;
        b = (b * b) & m2n;
        e >>= 1;
    }
    const double2 x = v[j];
    w[(t - 1) / 2] = x;
    w[(2u * N - t - 1) / 2] = make_double2(x.x, -x.y);
}

// one radix-2 Stockham pass (forward, exp(-2 pi i ...)); s: double2 elements between two jobs' buffers
__global__ void fft_pass(const double2* in, double2* out, int Ns, uint64_t s) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = N / 2;
    if (j >= half) return;
    in += (size_t)blockIdx.y * s;
    out += (size_t)blockIdx.y * s;
    const double2 v0 = in[j], v1 = in[j + half];
    const int k = j % Ns;
    double sn, cs;
    sincospi(-(double)k / (double)Ns, &sn, &cs);
    const double2 t = make_double2(v1.x * cs - v1.y * sn, v1.x * sn + v1.y * cs);
    const int idx = (j / Ns) * Ns * 2 + k;
    out[idx] = make_double2(v0.x + t.x, v0.y + t.y);
    out[idx + Ns] = make_double2(v0.x - t.x, v0.y - t.y);
}

// c_k = Re(Y_k zeta^{-k} / N) * scale, and max |c_k| (non-negative doubles order as their bit patterns)
// (ys: double2 elements between two jobs' Y; cst: doubles between two jobs' c; one cmax word per job)
__global__ void coeffs(const double2* Y, double scale, double* c, unsigned long long* cmax, uint64_t ys, uint64_t cst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    Y += (size_t)blockIdx.y * ys;
    c += (size_t)blockIdx.y * cst;
    cmax += blockIdx.y;
    double a = 0.0;
    if (k < N) {
        double sn, cs;
        sincospi(-(double)k / (double)N, &sn, &cs);
        const double re = (Y[k].x * cs - Y[k].y * sn) / (double)N * scale;
        c[k] = re;
        a = fabs(re);
    }
    for (int o = 16; o > 0; o >>= 1) a = fmax(a, __shfl_down_sync(0xffffffff, a, o));
    if ((threadIdx.x & 31) == 0) atomicMax(cmax, (unsigned long long)__double_as_longlong(a));
}

// OpenFHE's split: logc = ceil(log2 max|c|) exactly through frexp; coefficients past 2^61 are rounded at c / 2^split
__device__ __forceinline__ int split_of(unsigned long long cmax_bits) {
    const double m = __longlong_as_double((long long)cmax_bits);
    if (!(m > 0.0)) return 0;
    int e;
    const double f = frexp(m, &e);
    const int logc = (f == 0.5) ? e - 1 : e;
    return logc > 61 ? logc - 61 : 0;
}

// (cs: doubles between two jobs' c; os: words between two jobs' out; nsplit: +1 per job with a nonzero split)
template <class W, class MM>
__global__ void scatter(const double* c, const unsigned long long* cmax, W* out, const W* deg_factor,
                        const PrimeConst* pr, int nl, uint64_t cs, uint64_t os, unsigned long long* nsplit) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    c += (size_t)blockIdx.y * cs;
    out += (size_t)blockIdx.y * os;
    const int split = split_of(cmax[blockIdx.y]);
    if (nsplit != nullptr && split > 0 && k == 0) atomicAdd(nsplit, 1ull);
    const long long m = llround(ldexp(c[k], -split));
    const uint64_t am = m < 0 ? (uint64_t)(-m) : (uint64_t)m;
    for (int l = 0; l < nl; ++l) {
        const PrimeConst& q = pr[l];
        const W p = (W)q.p;
        W f = deg_factor[l];
        for (int i = 0; i < split; ++i) f = add<W>(f, f, p);
        W r = (W)(am % (uint64_t)q.p);
        if (m < 0 && r) r = (W)(p - r);
        out[(size_t)l * N + k] = MM::mul(r, f, q);
    }
}

// encode_many: every kernel over grid.y = batch jobs (strides from doubles to double2 for the complex buffers)
template <class W, class MM>
int launch_many(const EncodeManyArgs* m, cudaStream_t s) {
    const EncodeArgs* a = &m->e;
    const int nb = m->batch;
    if (nb < 1 || nb > 65535) return -1;
    if (nb > 1 && ((m->v_stride | m->ab_stride) & 1u)) return -2;
    const uint64_t vs = m->v_stride / 2, ab2 = m->ab_stride / 2;
    const dim3 gh((N / 2 + BX - 1) / BX, nb), gn((N + BX - 1) / BX, nb);
    double2* A = (double2*)a->a;
    double2* B = (double2*)a->b;
    place<<<gh, BX, 0, s>>>((const double2*)a->v, A, vs, ab2);
    double2* in = A;
    double2* out = B;
    for (int Ns = 1; Ns < N; Ns <<= 1) {
        fft_pass<<<gh, BX, 0, s>>>(in, out, Ns, ab2);
        double2* t = in;
        in = out;
        out = t;
    }
    const cudaError_t e = cudaMemsetAsync(a->cmax, 0, (size_t)nb * sizeof(unsigned long long), s);
    if (e != cudaSuccess) return (int)e;
    coeffs<<<gn, BX, 0, s>>>(in, a->scale, (double*)a->c, a->cmax, ab2, m->c_stride);
    scatter<W, MM><<<gn, BX, 0, s>>>((const double*)a->c, a->cmax, (W*)a->out, (const W*)a->deg_factor, a->pr, a->nl,
                                     m->c_stride, m->out_stride, m->nsplit);
    return (int)cudaGetLastError();
}

// encode: one job (blockIdx.y = 0: the strides multiply zero)
template <class W, class MM>
int launch(const EncodeArgs* a, cudaStream_t s) {
    const EncodeManyArgs m = {*a, 0, 0, 0, 0, nullptr, 1, 0};
    return launch_many<W, MM>(&m, s);
}
}  // namespace encode
}  // namespace pc
