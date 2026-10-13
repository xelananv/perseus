// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_encode_bridge.cu — see include/gpu_encode_bridge.h.
#include "gpu_encode_bridge.h"

#include "fideslib_wrapper.h"
#include "inference.h"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Plaintext.cuh>

#include <any>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace gpu_encode {

namespace {

using PtxImpl = typename Ptx::element_type;

// The deleter of a device plaintext's wrap: when the last reference goes without evict(), the
// graph name and tag keyed by this address are forgotten first (another plaintext may take the
// address next), then the facade's destructor evicts the device copy. The runtime is held weakly:
// a handle that outlives its session frees its device copy and nothing else.
struct ForgetOnDrop {
    std::weak_ptr<CKKSContext> fhe;
    void operator()(PtxImpl* p) const noexcept {
        if (auto f = fhe.lock()) f->forget_pt_key(static_cast<const void*>(p));
        delete p;
    }
};

}  // namespace

Bridge::Bridge(Inference& inf) : inf_(inf), context_(inf.cc()) {
    auto& fctx = std::any_cast<FIDESlib::CKKS::Context&>(inf.cc()->gpu);
    slots_ = inf.slots;
    dim_ = inf.size.dim;   // the port opens its session with dim = hidDim = d (gpu_smoke.open_session)
    enc_ = std::make_unique<GpuEncoder>(fctx, slots_, dim_);   // the shared_ptr itself: Plaintext(Context&) wants it
}

Bridge::~Bridge() {
    // destructors do not throw: release_tiles() is noexcept and the encoder's own teardown reports
    // rather than throws a context error
    std::lock_guard<std::mutex> lk(mtx_);
    if (enc_) enc_->release_tiles();
}

void Bridge::check_session() const {
    // identity by ownership, then by value: a context destroyed and another allocated in its place
    // is a change too, and the weak reference says so before anything is read through it
    const auto live = context_.lock();
    if (!live || live.get() != &*inf_.cc() || inf_.slots != slots_ || inf_.size.dim != dim_)
        throw std::runtime_error("gpu_encode: the session changed under its device encoder (a stale bridge); "
                                 "close the previous session before opening another");
}

void Bridge::check_ciphertext_stream(const Ctx& ref) {
    // The ordering contract (gpu_encode.h): the legacy default stream, where the plaintext polys are
    // made and the NTTs run, orders after the ciphertext streams only while those are blocking ones.
    // Checked once, on the ciphertext the products will run on (the plaintext polys themselves are
    // always on the legacy stream, so their flags say nothing).
    if (stream_checked_ || !ref->loaded) return;
    auto gct = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(inf_.cc()->GetDeviceCiphertext(ref->gpu));
    if (!gct || gct->c0.GPU.empty()) return;
    if (!stream_is_blocking(gct->c0.GPU[0].getS().ptr(), "gpu_encode"))
        throw std::runtime_error("gpu_encode: the library's ciphertext stream is non-blocking; the legacy-stream "
                                 "ordering the device encoder relies on does not hold");
    stream_checked_ = true;
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

std::vector<double> Bridge::tile_values(TileId id, int& d) {
    std::lock_guard<std::mutex> lk(mtx_);
    check_session();
    return enc_->download_tile(id, d);
}

bool Bridge::fits(const Ctx& ref) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!ref) return false;
    return inf_.encode_like_params(ref).second == 1;
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

void Bridge::prune_dropped() {
    // a handle dropped without evict() has already freed its device copy and forgotten its name and
    // tag (ForgetOnDrop); only this table still knows the address
    for (auto it = issued_.begin(); it != issued_.end();)
        it = it->second.expired() ? issued_.erase(it) : std::next(it);
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
    check_ciphertext_stream(ref);
    const auto [lv, nsd] = inf_.encode_like_params(ref);
    if (nsd != 1)
        throw std::runtime_error("encode_group: the device encoder makes degree-1 plaintexts only (fits() is false "
                                 "for this ciphertext; the CPU route serves it)");
    prune_dropped();
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
    std::vector<Ptx> out;
    out.reserve(B);
    std::vector<uint32_t> registered;
    registered.reserve(B);
    try {
        for (size_t b = 0; b < B; ++b) {
            const uint32_t handle = inf_.cc()->RegisterDevicePlaintext(std::static_pointer_cast<void>(gps[b]));
            registered.push_back(handle);
            Ptx pt(new PtxImpl(CC(inf_.cc())), ForgetOnDrop{inf_.fhe});
            pt->cpu = meta;
            pt->gpu = handle;
            pt->loaded = true;
            if (inf_.fhe) {
                inf_.fhe->tag_plaintext_top(pt);
                inf_.fhe->name_pt_if_absent(pt, names[b]);
            }
            // insert_or_assign: a handle dropped on another thread between prune_dropped() and this allocation
            // can hand the address back, and its expired entry must not outlive the new facade
            issued_.insert_or_assign(static_cast<const void*>(pt.get()), std::weak_ptr<PtxImpl>(pt));
            out.push_back(std::move(pt));
        }
    } catch (...) {
        // a wrap that failed midway leaves no device plaintext without a facade to evict it, and no
        // name or tag behind
        for (auto& pt : out) {
            issued_.erase(static_cast<const void*>(pt.get()));
            inf_.evict_plaintext(pt);
            if (inf_.fhe) inf_.fhe->forget_pt(pt);
        }
        for (size_t i = out.size(); i < registered.size(); ++i) inf_.cc()->EvictDevicePlaintext(registered[i]);
        throw;
    }
    return out;
}

void Bridge::evict(Ptx& pt) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!pt) return;
    const void* key = static_cast<const void*>(pt.get());
    auto it = issued_.find(key);
    if (it == issued_.end())
        throw std::invalid_argument("evict_ptx: not a device plaintext of this encoder (or already evicted)");
    const auto live = it->second.lock();
    if (!live || live.get() != pt.get()) {
        // the address of a handle that was dropped, taken since by another plaintext: not ours
        issued_.erase(it);
        throw std::invalid_argument("evict_ptx: not a device plaintext of this encoder (the handle it was issued to "
                                    "is gone)");
    }
    issued_.erase(it);
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
