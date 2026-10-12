// SPDX-License-Identifier: MIT
// gpu_encode_bridge.h — the device encoder (gpu_encode.h) as a session service: tiles registered
// once, plaintexts made per giant group at the level and scale a product with a given ciphertext
// needs, wrapped as facade plaintexts the ordinary fhe->mult consumes (such a wrap multiplies
// bit-identically to a CPU-encoded plaintext: tests/dev/gpu_encode_layout_probe.cu).
//
// Every public method holds the bridge's mutex: the bindings release the GIL around their calls
// and the encoder's buffers, plans and batch parity are shared state. A device plaintext is freed
// by evict() (the facade keeps the upstream convention of explicit eviction); after it the wrap is
// unloaded and carries no host metadata, so any later product refuses it instead of multiplying by
// the metadata's zeros.
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

    // B facade plaintexts: entry b is diagonal (g, baby[b]) of tiles[b], encoded at the level and
    // scale `encode_like_params(ref)` gives (the ciphertext the products will multiply), named
    // `names[b]` in the graph, tagged dense. Device-resident and ready when this returns.
    std::vector<Ptx> encode_group(const std::vector<TileId>& tile_ids, const std::vector<int>& baby, int g, int s,
                                  const Ctx& ref, const std::vector<std::string>& names);

    // Free the device plaintext behind a wrapped Ptx and forget its graph name and tag.
    void evict(Ptx& pt);

    GpuEncoder::Stats stats() const;

  private:
    void check_session() const;                       // the Inference is still the one this bridge was made for
    const std::any& meta_for_level(uint32_t level);   // one host metadata plaintext per level

    Inference& inf_;
    const void* context_ = nullptr;   // identity of the session's crypto context at construction
    int slots_ = 0, dim_ = 0;
    mutable std::mutex mtx_;
    std::unique_ptr<GpuEncoder> enc_;
    std::unordered_map<uint32_t, Ptx> meta_;   // level -> a zero plaintext made once (level, scale, slots)
};

// The session's bridge, created on first use. Callers keep the shared_ptr for the duration of a
// call, so a concurrent drop_bridge destroys the bridge only after the call returns.
std::shared_ptr<Bridge> bridge_for(Inference& inf);
void drop_bridge(Inference& inf);

}  // namespace gpu_encode
