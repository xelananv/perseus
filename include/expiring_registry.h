#pragma once

#include <cstddef>

// Address-keyed registries guarded by a weak_ptr — the runtime's ciphertext tag registry and
// the slot-layout tracker — are written once per new object and read back by the object's
// address. The guard keeps a stale entry from mislabeling a recycled address. But the objects
// are made with std::make_shared, which puts the object and its reference counts in one block
// that is released only when the last weak reference goes: every entry then keeps its dead
// object's block allocated, no address is ever recycled, the guard's erase path never runs,
// and the registry holds one entry per object ever made. A long decode adds hundreds of
// thousands of entries per token, the map rehashes on the standard library's growth ladder —
// a full relink of every entry, on the thread that feeds the GPU — and host memory grows with
// the number of ciphertexts ever created.
//
// sweep_if_doubled() is called on the insert path. It walks the map once each time the map
// has doubled since the previous sweep and drops every expired entry, so the cost is a couple
// of node visits per insert, and the map stays below max(floor, twice the size the previous
// sweep left): at most twice the peak live count, not twice the current one.
namespace expiring {

// Erase every entry for which expired(entry) holds. Returns the number erased.
template <class Map, class Expired>
std::size_t sweep(Map& m, Expired expired) {
    std::size_t n = 0;
    for (auto it = m.begin(); it != m.end();) {
        if (expired(it->second)) {
            it = m.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

// Sweep when the map holds at least `floor` entries and at least twice as many as it held
// after the previous sweep. `last` is the caller's record of that size; start it at 0.
template <class Map, class Expired>
std::size_t sweep_if_doubled(Map& m, std::size_t& last, Expired expired, std::size_t floor = 4096) {
    if (m.size() < floor || m.size() < 2 * last) return 0;
    const std::size_t n = sweep(m, expired);
    last = m.size();
    return n;
}

}  // namespace expiring
