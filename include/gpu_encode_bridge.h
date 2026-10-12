// SPDX-License-Identifier: MIT
// gpu_encode_bridge.h — the device encoder (gpu_encode.h) as a session service: tiles registered
// once, plaintexts made per giant group at the level and scale a product with a given ciphertext
// needs, wrapped as facade plaintexts the ordinary fhe->mult consumes (such a wrap multiplies
// bit-identically to a CPU-encoded plaintext: tests/dev/gpu_encode_layout_probe.cu).
#pragma once

#include "ckks_types.h"
#include "gpu_encode.h"

#include <string>
#include <unordered_map>
#include <vector>

struct Inference;

namespace gpu_encode {

class Bridge {
  public:
    explicit Bridge(Inference& inf);
    ~Bridge();

    // A d x d folded tile (row-major, y = x @ W), resident on the device; returns its id.
    int register_tile(const std::string& name, const std::vector<double>& W, int d);
    void release_tiles();

    // B facade plaintexts: entry b is diagonal (g, baby[b]) of tiles[b], encoded at the level and
    // scale `encode_like_params(ref)` gives (the ciphertext the products will multiply), named
    // `names[b]` in the graph, tagged dense. Device-resident and ready when this returns.
    std::vector<Ptx> encode_group(const std::vector<int>& tile_ids, const std::vector<int>& baby, int g, int s,
                                  const Ctx& ref, const std::vector<std::string>& names);

    // Free the device plaintext behind a wrapped Ptx (the host metadata is shared and stays).
    void evict(Ptx& pt);

    const GpuEncoder::Stats& stats() const { return enc_->stats(); }

  private:
    const std::any& meta_for_level(uint32_t level);   // one host metadata plaintext per level

    Inference& inf_;
    std::unique_ptr<GpuEncoder> enc_;
    std::vector<TileHandle> tiles_;
    std::vector<std::string> tile_names_;
    std::unordered_map<uint32_t, Ptx> meta_;   // level -> a zero plaintext made once (level, scale, slots)
};

// The session's bridge (one Inference per process); created on first use.
Bridge& bridge_for(Inference& inf);
void drop_bridge(Inference& inf);

}  // namespace gpu_encode
