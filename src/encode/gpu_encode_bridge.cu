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
    const int n = inf.slots;
    const int d = inf.size.dim;   // the port opens its session with dim = hidDim = d (gpu_smoke.open_session)
    enc_ = std::make_unique<GpuEncoder>(*fctx, n, d);
}

Bridge::~Bridge() { release_tiles(); }

int Bridge::register_tile(const std::string& name, const std::vector<double>& W, int d) {
    tiles_.push_back(enc_->register_tile(W, d));
    tile_names_.push_back(name);
    return static_cast<int>(tiles_.size()) - 1;
}

void Bridge::release_tiles() {
    for (auto& t : tiles_) enc_->release_tile(t);
    tiles_.clear();
    tile_names_.clear();
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

std::vector<Ptx> Bridge::encode_group(const std::vector<int>& tile_ids, const std::vector<int>& baby, int g, int s,
                                      const Ctx& ref, const std::vector<std::string>& names) {
    const size_t B = tile_ids.size();
    if (baby.size() != B || names.size() != B) throw std::invalid_argument("encode_group: tiles, baby, names differ in length");
    if (B == 0 || B > static_cast<size_t>(kMaxBatch)) throw std::invalid_argument("encode_group: batch size out of range");
    const auto [lv, nsd] = inf_.encode_like_params(ref);
    if (nsd != 1) throw std::runtime_error("encode_group: the device encoder makes degree-1 plaintexts only");
    GroupJob job;
    job.tiles.reserve(B);
    for (int id : tile_ids) {
        if (id < 0 || id >= static_cast<int>(tiles_.size())) throw std::out_of_range("encode_group: unknown tile id");
        job.tiles.push_back(tiles_[id]);
    }
    job.baby = baby;
    job.g = g;
    job.s = s;
    job.level = static_cast<int>(lv);
    job.sf = inf_.cc()->ScalingFactorReal(lv);

    auto gps = enc_->encode_group(job);   // K1 -> cuFFT -> K2 -> NTT, ready event recorded
    // the legacy-stream bridge: every later launch on a blocking stream (the ciphertext ops) waits
    cudaStreamWaitEvent(nullptr, enc_->last_ready_event(), 0);

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

void Bridge::evict(Ptx& pt) { inf_.evict_plaintext(pt); }

namespace {
std::mutex g_mtx;
std::unordered_map<const Inference*, std::unique_ptr<Bridge>> g_bridges;
}  // namespace

Bridge& bridge_for(Inference& inf) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_bridges.find(&inf);
    if (it == g_bridges.end()) it = g_bridges.emplace(&inf, std::make_unique<Bridge>(inf)).first;
    return *it->second;
}

void drop_bridge(Inference& inf) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_bridges.erase(&inf);
}

}  // namespace gpu_encode
