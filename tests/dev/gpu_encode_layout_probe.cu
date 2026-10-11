// SPDX-License-Identifier: MIT
// gpu_encode_layout_probe — R1 encoder groundwork, parity step 0 (2026-10-11). No gtest: prints JSON lines.
//
// Two questions the device encoder's K2 and wrap depend on, answered on the library itself:
//  A. In which index order must COEFF-form residues be written into an RNSPoly's limbs so that the
//     library's own NTT yields exactly what `Plaintext(ctx, GetRawPlainText(pt))` holds? Candidates:
//     natural order (OpenFHE's) and 16-bit bit-reversed (the order GetRawPlainText imports with).
//  B. Does a device plaintext built that way, registered with RegisterDevicePlaintext and wrapped in
//     a facade Plaintext (cpu = metadata, gpu = handle, loaded = true), multiply identically to the
//     CPU-encoded reference through the ordinary EvalMult?
//
//   gpu_encode_layout_probe [LEVEL=34]      (n32 env from scripts/local_env.sh; bootstrap off, no rotation keys)
#include "fideslib_wrapper.h"
#include "test_helpers.h"

#include <CKKS/Limb.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/RNSPoly.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <variant>
#include <vector>

namespace {

uint32_t brev16(uint32_t k) {
    uint32_t r = 0;
    for (int b = 0; b < 16; ++b) r |= ((k >> b) & 1u) << (15 - b);
    return r;
}

// COEFF-form residues of an OpenFHE plaintext, per tower, natural order (what the encoder produces).
std::vector<std::vector<uint64_t>> coeff_residues(const lbcrypto::Plaintext& pt) {
    lbcrypto::DCRTPoly el = pt->GetElement<lbcrypto::DCRTPoly>();
    el.SetFormat(lbcrypto::Format::COEFFICIENT);
    std::vector<std::vector<uint64_t>> out;
    for (const auto& tower : el.GetAllElements()) {
        const auto& v = tower.GetValues();
        std::vector<uint64_t> r(v.GetLength());
        for (size_t k = 0; k < r.size(); ++k) r[k] = v[k].ConvertToInt();
        out.push_back(std::move(r));
    }
    return out;
}

// An RNSPoly whose limb l holds `res[l]` written at index idx(k); NTT'd on its own stream, synced.
FIDESlib::CKKS::RNSPoly build_poly(FIDESlib::CKKS::ContextData& cd,
                                   const std::vector<std::vector<uint64_t>>& res, bool reversed) {
    const int nlimbs = static_cast<int>(res.size());
    FIDESlib::CKKS::RNSPoly poly(cd, nlimbs - 1, false, false);
    for (int l = 0; l < nlimbs; ++l) {
        const size_t N = res[l].size();
        std::vector<uint32_t> h(N);
        for (size_t k = 0; k < N; ++k) h[reversed ? brev16(static_cast<uint32_t>(k)) : k] = static_cast<uint32_t>(res[l][k]);
        auto& limb = poly.GPU[cd.limbGPUid[l].x].limb[cd.limbGPUid[l].y];
        std::get<FIDESlib::CKKS::Limb<uint32_t>>(limb).load_convert(h);
    }
    poly.NTT(cd.batch, true);   // the default algorithm (ALGO_SHOUP), as the library's own loads use
    cudaDeviceSynchronize();
    return poly;
}

size_t mismatches(const std::vector<std::vector<uint64_t>>& a, const std::vector<std::vector<uint64_t>>& b) {
    size_t m = 0;
    for (size_t l = 0; l < a.size() && l < b.size(); ++l)
        for (size_t k = 0; k < a[l].size() && k < b[l].size(); ++k) m += (a[l][k] != b[l][k]);
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t L = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 34u;
    CKKSContextOptions o = ckks_options_from_env();
    o.enable_bootstrap = false;   // the same 54-prime chain (depth 26 either way), no bootstrap keys or tables
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

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-0.3, 0.3);
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = U(rng);
    for (auto& v : y) v = U(rng);

    Ptx ref = cc->MakeCKKSPackedPlaintext(x, 1, L);
    auto& ref_o = std::any_cast<lbcrypto::Plaintext&>(ref->cpu);
    // the library's own import of the same plaintext = the target layout
    FIDESlib::CKKS::Plaintext gp_ref(fctx, FIDESlib::CKKS::GetRawPlainText(occ, ref_o));
    FIDESlib::CKKS::RawPlainText raw_ref;
    gp_ref.store(raw_ref);
    const auto res = coeff_residues(ref_o);
    std::printf("{\"probe\":\"layout\",\"level\":%u,\"nlimbs\":%zu,\"N\":%zu,\"ref_limbs\":%d,\"sf\":%.6e}\n", L,
                res.size(), res.empty() ? 0 : res[0].size(), raw_ref.numRes, ref_o->GetScalingFactor());

    // A. which order reproduces the library's limbs after its NTT
    size_t mm_nat = 0, mm_rev = 0;
    {
        auto p_nat = build_poly(cd, res, false);
        std::vector<std::vector<uint64_t>> out;
        p_nat.store(out);
        mm_nat = mismatches(out, raw_ref.sub_0);
    }
    FIDESlib::CKKS::RNSPoly p_rev = build_poly(cd, res, true);
    {
        std::vector<std::vector<uint64_t>> out;
        p_rev.store(out);
        mm_rev = mismatches(out, raw_ref.sub_0);
    }
    const size_t total = res.size() * (res.empty() ? 0 : res[0].size());
    std::printf("{\"probe\":\"order\",\"mismatch_natural\":%zu,\"mismatch_bitrev16\":%zu,\"of\":%zu}\n", mm_nat, mm_rev, total);

    // B. the wrap: a device plaintext from the matching order, through the facade's EvalMult
    const bool use_rev = mm_rev <= mm_nat;
    auto gp2 = std::make_shared<FIDESlib::CKKS::Plaintext>(fctx);
    if (use_rev) {
        gp2->c0.swap(p_rev);
    } else {
        auto p2 = build_poly(cd, res, false);
        gp2->c0.swap(p2);
    }
    gp2->NoiseFactor = ref_o->GetScalingFactor();
    gp2->NoiseLevel = ref_o->GetNoiseScaleDeg();
    gp2->slots = n;
    cudaDeviceSynchronize();
    const uint32_t handle = cc->RegisterDevicePlaintext(std::static_pointer_cast<void>(gp2));
    using PtxImpl = typename Ptx::element_type;
    Ptx mypt = std::make_shared<PtxImpl>(CC(cc));
    mypt->cpu = ref->cpu;   // metadata only (level, scale, slots); the device data is gp2's
    mypt->gpu = handle;
    mypt->loaded = true;

    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(y, 1, L), fhe->pk());
    Ctx r_ref = cc->EvalMult(ct, ref);
    Ctx r_dev = cc->EvalMult(ct, mypt);
    const auto d_ref = test_helpers::decrypt_slots(*fhe, r_ref);
    const auto d_dev = test_helpers::decrypt_slots(*fhe, r_dev);
    double max_dd = 0, max_plain = 0, max_ref = 0;
    for (int i = 0; i < n; ++i) {
        max_dd = std::max(max_dd, std::abs(d_ref[i] - d_dev[i]));
        max_plain = std::max(max_plain, std::abs(d_dev[i] - x[i] * y[i]));
        max_ref = std::max(max_ref, std::abs(x[i] * y[i]));
    }
    std::printf("{\"probe\":\"wrap\",\"order\":\"%s\",\"max_abs_dev_minus_ref\":%.3e,\"max_abs_dev_minus_plain\":%.3e,"
                "\"max_abs_plain\":%.3e,\"pt_level\":%u}\n",
                use_rev ? "bitrev16" : "natural", max_dd, max_plain, max_ref, ref_o->GetLevel());
    cc->EvictDevicePlaintext(handle);
    return (mm_nat == 0 || mm_rev == 0) && max_dd < 1e-9 ? 0 : 1;
}
