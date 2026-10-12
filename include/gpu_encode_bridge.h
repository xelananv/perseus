// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Alexander Ananyev
// gpu_encode_bridge.h — the device encoder (gpu_encode.h) as a session service: tiles registered
// once, plaintexts made per giant group at the level and scale a product with a given ciphertext
// needs, wrapped as facade plaintexts the ordinary fhe->mult consumes (such a wrap multiplies
// bit-identically to a CPU-encoded plaintext: tests/dev/gpu_encode_layout_probe.cu).
//
// Every public method holds the bridge's mutex: the bindings release the GIL around their calls
// and the encoder's buffers, plans and batch parity are shared state. A device plaintext is freed
// by evict() (the upstream convention of explicit eviction) or, failing that, when the last
// reference to its facade goes: the facade's destructor evicts the device copy and a deleter on the
// wrap forgets the graph name and tag keyed by its address, so neither outlives the object. After
// evict() the wrap is unloaded and carries no host metadata, so a later product refuses it instead
// of multiplying by the metadata's zeros.
#pragma once

#include "ckks_types.h"
#include "gpu_encode.h"

#include <any>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct Inference;

namespace gpu_encode {

class Bridge {
  public:
    explicit Bridge(Inference& inf);
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    // A d x d folded tile (row-major, y = x @ W), resident on the device; returns its id.
    TileId register_tile(const std::vector<double>& W, int d);
    void release_tiles();
    // The resident tile back on the host (row-major, d x d; `d` is set): what the CPU route encodes
    // from when a group cannot take the device route (fits() false, or the route switched off).
    std::vector<double> tile_values(TileId id, int& d);

    // True when encode_group can serve a product with `ref`. The encoder makes degree-1 plaintexts:
    // every encode on the composite chain and, on a one-prime chain, every input without a pending
    // rescale (encode_like_params); for the rest the caller takes the CPU route.
    bool fits(const Ctx& ref) const;

    // B facade plaintexts: entry b is diagonal (g, baby[b]) of tiles[b], encoded at the level and
    // scale `encode_like_params(ref)` gives (the ciphertext the products will multiply), named
    // `names[b]` in the graph, tagged as the CPU route tags a dense real vector. Device-resident
    // and ready when this returns.
    std::vector<Ptx> encode_group(const std::vector<TileId>& tile_ids, const std::vector<int>& baby, int g, int s,
                                  const Ctx& ref, const std::vector<std::string>& names);

    // Free the device plaintext behind a wrapped Ptx and forget its graph name and tag. Only a
    // plaintext this bridge issued and still holds is accepted: a handle that was dropped and whose
    // address another plaintext took since is refused.
    void evict(Ptx& pt);

    GpuEncoder::Stats stats() const;

  private:
    using PtxImpl = typename Ptx::element_type;
    void check_session() const;                       // the Inference still runs the context this bridge was made for
    void check_ciphertext_stream(const Ctx& ref);     // once: the ciphertext streams are blocking ones
    const std::any& meta_for_level(uint32_t level);   // one host metadata plaintext per level
    void prune_dropped();                             // drop the entries of handles that went away

    Inference& inf_;
    std::weak_ptr<typename CC::element_type> context_;   // the session's crypto context at construction, by ownership
    int slots_ = 0, dim_ = 0;
    bool stream_checked_ = false;
    mutable std::mutex mtx_;
    std::unique_ptr<GpuEncoder> enc_;
    std::unordered_map<uint32_t, Ptx> meta_;   // level -> a zero plaintext made once (level, scale, slots)
    // the device plaintexts this bridge made and has not evicted, by address, each with a weak
    // reference that tells a live handle from a dropped one whose address was reused
    std::unordered_map<const void*, std::weak_ptr<PtxImpl>> issued_;
};

// The session's bridge, created on first use. Callers keep the shared_ptr for the duration of a
// call, so a concurrent drop_bridge destroys the bridge only after the call returns.
std::shared_ptr<Bridge> bridge_for(Inference& inf);
void drop_bridge(Inference& inf);

}  // namespace gpu_encode
