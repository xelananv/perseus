// SPDX-License-Identifier: MIT
// gpu_encode_bridge.cu — see include/gpu_encode_bridge.h.
#include "gpu_encode_bridge.h"

#include "fideslib_wrapper.h"
#include "inference.h"

#include <CKKS/Plaintext.cuh>

#include <any>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace gpu_encode {

Bridge::Bridge(Inference& inf) : inf_(inf) {
    auto& fctx = std::any_cast<FIDESlib::CKKS::Context&>(inf.cc()->gpu);
    context_ = static_cast<const void*>(&*inf.cc());
    slots_ = inf.slots;
    dim_ = inf.size.dim;   // the port opens its session with dim = hidDim = d (gpu_smoke.open_session)
    enc_ = std::make_unique<GpuEncoder>(fctx, slots_, dim_);   // the shared_ptr itself: Plaintext(Context&) wants it
}

Bridge::~Bridge() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (enc_) enc_->release_tiles();
}

void Bridge::check_session() const {
    if (static_cast<const void*>(&*inf_.cc()) != context_ || inf_.slots != slots_ || inf_.size.dim != dim_)
        throw std::runtime_error("gpu_encode: the session changed under its device encoder (a stale bridge); "
                                 "close the previous session before opening another");
}

TileId Bridge::register_tile(const std::vector<double>& W, int d) {
    std::lock_guard<std::mutex> lk(mtx_);
    check_session();
    return enc_->register_tile(W, d);
}

void Bridge::release_tiles() {
    std::lock_guard<std::mutex> lk(mtx_);
    enc_->release_tiles();
}

GpuEncoder::Stats Bridge::stats() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return enc_->stats();
}

const std::any& Bridge::meta_for_level(uint32_t level) {
    auto it = meta_.find(level);
    if (it == meta_.end()) {
        // the facade reads a plaintext's level and scale from its host object (Plaintext.cpp:51-57);
        // one zero encode per level serves every device plaintext made at that level
        std::vector<double> zeros(static_cast<size_t>(inf_.slots), 0.0);
        it = meta_.emplace(level, inf_.cc()->MakeCKKSPackedPlaintext(zeros, 1, level)).first;
    }
    return it->second->cpu;
}

std::vector<Ptx> Bridge::encode_group(const std::vector<TileId>& tile_ids, const std::vector<int>& baby, int g, int s,
                                      const Ctx& ref, const std::vector<std::string>& names) {
    std::lock_guard<std::mutex> lk(mtx_);
    check_session();
    const size_t B = tile_ids.size();
    if (baby.size() != B || names.size() != B)
        throw std::invalid_argument("encode_group: tiles, baby, names differ in length");
    if (B == 0 || B > static_cast<size_t>(kMaxBatch))
        throw std::invalid_argument("encode_group: batch size out of range");
    if (!ref) throw std::invalid_argument("encode_group: null reference ciphertext");
    const auto [lv, nsd] = inf_.encode_like_params(ref);
    if (nsd != 1) throw std::runtime_error("encode_group: the device encoder makes degree-1 plaintexts only");
    GroupJob job;
    job.tiles = tile_ids;
    job.baby = baby;
    job.g = g;
    job.s = s;
    job.level = static_cast<int>(lv);
    job.sf = inf_.cc()->ScalingFactorReal(lv);

    auto gps = enc_->encode_group(job);   // K1 -> cuFFT -> K2 -> NTT, ready event recorded
    // the legacy-stream bridge: every later launch on a blocking stream (the ciphertext ops) waits
    cuda_check(cudaStreamWaitEvent(nullptr, enc_->last_ready_event(), 0), "encode_group", "cudaStreamWaitEvent");

    const std::any& meta = meta_for_level(lv);
    using PtxImpl = typename Ptx::element_type;
    std::vector<Ptx> out;
    out.reserve(B);
    for (size_t b = 0; b < B; ++b) {
        const uint32_t handle = inf_.cc()->RegisterDevicePlaintext(std::static_pointer_cast<void>(gps[b]));
        Ptx pt = std::make_shared<PtxImpl>(CC(inf_.cc()));
        pt->cpu = meta;
        pt->gpu = handle;
        pt->loaded = true;
        if (inf_.fhe) {
            inf_.fhe->tag_plaintext_top(pt);
            inf_.fhe->name_pt_if_absent(pt, names[b]);
        }
        out.push_back(std::move(pt));
    }
    return out;
}

void Bridge::evict(Ptx& pt) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!pt) return;
    inf_.evict_plaintext(pt);
    if (inf_.fhe) inf_.fhe->forget_pt(pt);
    pt->cpu.reset();   // no host values: a product after eviction fails instead of multiplying by zeros
}

namespace {
std::mutex g_mtx;
// Leaked on purpose: a bridge holds facade plaintexts and CUDA handles, which must be released while
// the session is alive (drop_bridge from Session.close); a registry destroyed at static destruction
// would free them after the context and the CUDA runtime are gone (observed: "double free or
// corruption" at interpreter exit). A bridge whose session was never closed leaks with the process.
auto& registry() {
    static auto* r = new std::unordered_map<const Inference*, std::shared_ptr<Bridge>>();
    return *r;
}
}  // namespace

std::shared_ptr<Bridge> bridge_for(Inference& inf) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto& reg = registry();
    auto it = reg.find(&inf);
    if (it == reg.end()) it = reg.emplace(&inf, std::make_shared<Bridge>(inf)).first;
    return it->second;
}

void drop_bridge(Inference& inf) {
    std::shared_ptr<Bridge> gone;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto& reg = registry();
        auto it = reg.find(&inf);
        if (it == reg.end()) return;
        gone = std::move(it->second);
        reg.erase(it);
    }
    gone.reset();   // outside the lock: frees tiles, metadata plaintexts, the encoder (when no call holds it)
}

}  // namespace gpu_encode
