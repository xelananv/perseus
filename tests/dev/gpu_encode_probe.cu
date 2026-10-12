// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_encode_probe — parity of the device encoder (include/gpu_encode.h) against OpenFHE's CPU encode,
// on the library itself. No gtest: JSON lines; exit 0 iff every gate passes.
//
//  1. COEFF parity (skip_ntt): for B diagonals of one giant index, the device's natural-order residues
//     vs MakeCKKSPackedPlaintext's; the integer coefficient is recovered from the q0 pair by CRT on both
//     sides, so the comparison is in units of the rounded integer (|m| up to ~2^52: fp64 summation order
//     differs between cuFFT and FFTSpecialInv, so a few units of difference are the expected noise).
//  2. EVAL parity: the same plaintexts NTT'd by the encoder vs Plaintext(ctx, GetRawPlainText(cpu_pt)).
//  3. Product: a device plaintext wrapped as the bridge wraps it, times an encrypted random vector,
//     decrypted against the CPU-encoded product and against the plaintext product.
//
//   gpu_encode_probe [LEVEL=34] [G=3] [SEED=7]      (n32 env; bootstrap off, no rotation keys)
#include "fideslib_wrapper.h"
#include "gpu_encode.h"
#include "test_helpers.h"

#include <CKKS/Plaintext.cuh>
#include <CKKS/RNSPoly.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace {

using i128 = __int128;

// centred integer from the residues modulo the first two primes (the q0 pair: 56 bits > 53)
int64_t crt2_centred(uint64_t r0, uint64_t q0, uint64_t r1, uint64_t q1) {
    // m = r0 + q0 * t,  t = (r1 - r0) * q0^{-1} mod q1
    auto inv = [](uint64_t a, uint64_t m) {  // a^{-1} mod m by Fermat (m prime)
        uint64_t e = m - 2, r = 1;
        a %= m;
        while (e) {
            if (e & 1) r = (uint64_t)((i128)r * a % m);
            a = (uint64_t)((i128)a * a % m);
            e >>= 1;
        }
        return r;
    };
    const uint64_t d = (r1 + q1 - (r0 % q1)) % q1;
    const uint64_t t = (uint64_t)((i128)d * inv(q0 % q1, q1) % q1);
    i128 m = (i128)r0 + (i128)q0 * t;
    const i128 M = (i128)q0 * q1;
    if (m > M / 2) m -= M;
    return (int64_t)m;
}

std::vector<std::vector<uint64_t>> coeff_residues(const lbcrypto::Plaintext& pt) {
    lbcrypto::DCRTPoly el = pt->GetElement<lbcrypto::DCRTPoly>();
    el.SetFormat(Format::COEFFICIENT);
    std::vector<std::vector<uint64_t>> out;
    for (const auto& tower : el.GetAllElements()) {
        const auto& v = tower.GetValues();
        std::vector<uint64_t> r(v.GetLength());
        for (size_t k = 0; k < r.size(); ++k) r[k] = v[k].ConvertToInt();
        out.push_back(std::move(r));
    }
    return out;
}

// the slot vector of diagonal (g, b) of tile W, as filling._Diagonal builds it: channel i's value over
// t slots, rolled by g*s*t; the value at channel i is W[(i + g*s + b) mod d][i]
std::vector<double> diagonal_slots(const std::vector<double>& W, int d, int n, int g, int s, int b) {
    const int t = n / d, shift = (g * s * t) % n;
    std::vector<double> z(n);
    for (int m = 0; m < n; ++m) {
        const int i = (((m - shift) % n + n) % n) / t;
        z[m] = W[static_cast<size_t>((i + g * s + b) % d) * d + i];
    }
    return z;
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t L = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 34u;
    const int g = argc > 2 ? std::atoi(argv[2]) : 3;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 7u;
    CKKSContextOptions o = ckks_options_from_env();
    o.enable_bootstrap = false;
    o.depth = 26;
    o.extra_rot_steps.clear();
    o.h_weight = 0;
    o.ckks_complex_payload = true;
    auto fhe = make_ckks_context(o);
    CC cc = fhe->cc;
    auto& fctx = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
    FIDESlib::CKKS::ContextData& cd = *fctx;
    auto& occ = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    const int n = static_cast<int>(cc->GetRingDimension() / 2);
    const int d = 1024, t = n / d, s = 32, B = s;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-0.3, 0.3);
    std::vector<double> W(static_cast<size_t>(d) * d), y(n);
    for (auto& v : W) v = U(rng);
    for (auto& v : y) v = U(rng);

    gpu_encode::GpuEncoder enc(fctx, n, d);
    const gpu_encode::TileId tile = enc.register_tile(W, d);
    const double sf = cc->ScalingFactorReal(L);
    gpu_encode::GroupJob job;
    job.tiles.assign(B, tile);
    for (int b = 0; b < B; ++b) job.baby.push_back(b);
    job.g = g;
    job.s = s;
    job.level = static_cast<int>(L);
    job.sf = sf;

    // CPU references for every diagonal of the group
    std::vector<Ptx> refs(B);
    for (int b = 0; b < B; ++b) refs[b] = cc->MakeCKKSPackedPlaintext(diagonal_slots(W, d, n, g, s, b), 1, L);
    // ABI: ContextData fields after `precom` are at other offsets in a non-NCCL TU (Context.cuh:164);
    // `param` is before the shift.
    const auto& q = cd.param.primes;   // PrimeRecord: .p
    std::printf("{\"probe\":\"setup\",\"level\":%u,\"g\":%d,\"B\":%d,\"sf\":%.6e,\"q0\":%lu,\"q1\":%lu}\n", L, g, B, sf,
                (unsigned long)q[0].p, (unsigned long)q[1].p);
    std::fflush(stdout);

    // 1. COEFF parity. The integer m = llround(sf * c) differs by one unit wherever cuFFT's and
    //    FFTSpecialInv's fp64 summation orders land on opposite sides of a rounding boundary; the
    //    relative error that matters is in the infinity norm, max|m_dev - m_cpu| / max|m_cpu| (one
    //    unit against |m| ~ 0.3 sf ~ 2^52), not per coefficient (a 1-unit difference on a tiny
    //    coefficient is the same rounding noise).
    job.skip_ntt = true;
    auto coeff = enc.encode_group(job);
    cudaDeviceSynchronize();
    std::map<int64_t, size_t> hist;   // |integer difference| -> count
    size_t total = 0, res_equal = 0, res_total = 0, limbs_cpu = 0, limbs_dev = 0;
    double max_rel = 0.0;
    int64_t max_diff = 0, max_int = 0;
    std::vector<std::vector<std::vector<uint64_t>>> coeff_dev(B);   // kept for the EVAL self-check
    for (int b = 0; b < B; ++b) {
        auto& dev = coeff_dev[b];
        coeff[b]->c0.store(dev);
        auto& ref_o = std::any_cast<lbcrypto::Plaintext&>(refs[b]->cpu);
        const auto cpu = coeff_residues(ref_o);
        if (b == 0) {
            limbs_cpu = cpu.size();
            limbs_dev = dev.size();
        }
        const size_t N = cpu[0].size();
        for (size_t k = 0; k < N; ++k) {
            const int64_t mc = crt2_centred(cpu[0][k], q[0].p, cpu[1][k], q[1].p);
            const int64_t md = crt2_centred(dev[0][k], q[0].p, dev[1][k], q[1].p);
            const int64_t dd = std::llabs(mc - md);
            ++hist[std::min<int64_t>(dd, 100)];
            ++total;
            max_diff = std::max<int64_t>(max_diff, dd);
            max_int = std::max<int64_t>(max_int, std::llabs(mc));
            if (mc != 0) max_rel = std::max(max_rel, (double)dd / std::fabs((double)mc));
        }
        for (size_t l = 0; l < cpu.size() && l < dev.size(); ++l)
            for (size_t k = 0; k < N; ++k) {
                res_equal += (cpu[l][k] == dev[l][k]);
                ++res_total;
            }
    }
    const double rel_inf = max_int ? (double)max_diff / (double)max_int : 0.0;
    const size_t diff10plus = [&] { size_t c = 0; for (auto& [v, cnt] : hist) if (v >= 10) c += cnt; return c; }();
    std::printf("{\"probe\":\"coeff\",\"limbs_cpu\":%zu,\"limbs_dev\":%zu,\"coefficients\":%zu,"
                "\"exact\":%zu,\"diff1\":%zu,"
                "\"diff2_9\":%zu,\"diff10plus\":%zu,\"max_abs_diff_units\":%lld,\"max_abs_int\":%lld,\"rel_inf\":%.3e,"
                "\"max_rel_per_coeff\":%.3e,\"residues_equal\":%zu,\"of\":%zu}\n",
                limbs_cpu, limbs_dev, total, hist[0], hist[1],
                [&] { size_t c = 0; for (int64_t v = 2; v < 10; ++v) c += hist[v]; return c; }(), diff10plus,
                (long long)max_diff, (long long)max_int, rel_inf, max_rel, res_equal, res_total);
    std::fflush(stdout);

    // 2. EVAL parity. (a) self-consistency: the device's EVAL words must equal the library's own NTT of
    //    the device's COEFF residues (RNSPoly::load + NTT, the path gpu_encode_layout_probe validated) — this is
    //    what checks the grow/aux/stream choreography. (b) against the CPU reference's EVAL words:
    //    informational only, since one differing coefficient changes every word after the NTT.
    job.skip_ntt = false;
    auto evalp = enc.encode_group(job);
    cudaDeviceSynchronize();
    const FIDESlib::CKKS::RawPlainText raw0 =
        FIDESlib::CKKS::GetRawPlainText(occ, std::any_cast<lbcrypto::Plaintext&>(refs[0]->cpu));   // the moduli
    size_t eq_self = 0, eq_ref = 0, tot = 0;
    for (int b = 0; b < B; ++b) {
        std::vector<std::vector<uint64_t>> dev;
        evalp[b]->c0.store(dev);
        FIDESlib::CKKS::RNSPoly lib(cd, -1, false, false);
        lib.load(coeff_dev[b], raw0.moduli);
        lib.NTT(cd.param.batch, true);
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> self;
        lib.store(self);
        auto& ref_o = std::any_cast<lbcrypto::Plaintext&>(refs[b]->cpu);
        FIDESlib::CKKS::Plaintext gp_ref(fctx, FIDESlib::CKKS::GetRawPlainText(occ, ref_o));
        FIDESlib::CKKS::RawPlainText raw;
        gp_ref.store(raw);
        for (size_t l = 0; l < dev.size(); ++l)
            for (size_t k = 0; k < dev[l].size(); ++k) {
                eq_self += (l < self.size() && k < self[l].size() && self[l][k] == dev[l][k]);
                eq_ref += (l < raw.sub_0.size() && k < raw.sub_0[l].size() && raw.sub_0[l][k] == dev[l][k]);
                ++tot;
            }
    }
    std::printf("{\"probe\":\"eval\",\"words_equal_self\":%zu,\"words_equal_cpu_ref\":%zu,\"of\":%zu}\n",
                eq_self, eq_ref, tot);
    std::fflush(stdout);

    // 3. product through the facade, wrapped as the bridge wraps it
    using PtxImpl = typename Ptx::element_type;
    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(y, 1, L), fhe->pk());
    double max_dev_ref = 0.0, max_dev_plain = 0.0, max_plain = 0.0;
    for (int b : {0, B / 2, B - 1}) {
        const uint32_t handle = cc->RegisterDevicePlaintext(std::static_pointer_cast<void>(evalp[b]));
        Ptx mypt = std::make_shared<PtxImpl>(CC(cc));
        mypt->cpu = refs[b]->cpu;
        mypt->gpu = handle;
        mypt->loaded = true;
        const auto d_ref = test_helpers::decrypt_slots(*fhe, cc->EvalMult(ct, refs[b]));
        const auto d_dev = test_helpers::decrypt_slots(*fhe, cc->EvalMult(ct, mypt));
        const auto z = diagonal_slots(W, d, n, g, s, b);
        for (int i = 0; i < n; ++i) {
            max_dev_ref = std::max(max_dev_ref, std::fabs(d_ref[i] - d_dev[i]));
            max_dev_plain = std::max(max_dev_plain, std::fabs(d_dev[i] - y[i] * z[i]));
            max_plain = std::max(max_plain, std::fabs(y[i] * z[i]));
        }
        cc->EvictDevicePlaintext(handle);
    }
    std::printf("{\"probe\":\"product\",\"max_abs_dev_minus_ref\":%.3e,\"rel_dev_minus_ref\":%.3e,"
                "\"max_abs_dev_minus_plain\":%.3e,"
                "\"max_abs_plain\":%.3e}\n",
                max_dev_ref, max_plain > 0 ? max_dev_ref / max_plain : 0.0, max_dev_plain, max_plain);
    const auto& st = enc.stats();
    std::printf("{\"probe\":\"stats\",\"plaintexts\":%lu,\"batches\":%lu,\"k1_ms\":%.3f,\"fft_ms\":%.3f,"
                "\"k2_ms\":%.3f,\"ntt_ms\":%.3f}\n",
                (unsigned long)st.plaintexts, (unsigned long)st.batches, st.k1_ms, st.fft_ms, st.k2_ms, st.ntt_ms);
    // gates: no integer off by 10 or more units, infinity-norm relative error <= 1e-13, the device's EVAL
    // words identical to the library's NTT of its own coefficients, the product within 1e-9 of the
    // CPU-encoded plaintext's
    const bool ok = diff10plus == 0 && rel_inf <= 1e-13 && eq_self == tot && tot > 0 && max_dev_ref < 1e-9;
    std::printf("{\"probe\":\"verdict\",\"pass\":%s}\n", ok ? "true" : "false");
    return ok ? 0 : 1;
}
