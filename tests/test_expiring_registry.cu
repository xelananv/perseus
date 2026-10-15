// CPU unit test of the expiring-registry sweep (include/expiring_registry.h): an address-keyed
// map whose entries hold a weak_ptr to the object they describe, the shape of the runtime's
// ciphertext tag registry and of the slot-layout tracker. No CUDA, no FIDESlib.
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "expiring_registry.h"

namespace {

struct Entry {
    std::weak_ptr<int> wp;
    int label = 0;
};
using Registry = std::unordered_map<const void*, Entry>;

bool expired(const Entry& e) { return e.wp.expired(); }

void insert(Registry& r, const std::shared_ptr<int>& o, int label) {
    r[(const void*)o.get()] = Entry{o, label};
}

// The read side of both registries: a hit is valid only if the object recorded is the one asked
// about; otherwise the stale entry is dropped and the answer is "unknown" (-1).
int lookup(Registry& r, const std::shared_ptr<int>& o) {
    auto it = r.find((const void*)o.get());
    if (it == r.end()) return -1;
    auto live = it->second.wp.lock();
    if (!live || live.get() != o.get()) {
        r.erase(it);
        return -1;
    }
    return it->second.label;
}

}  // namespace

TEST(ExpiringRegistry, SweepErasesExactlyTheExpiredEntries) {
    Registry r;
    std::vector<std::shared_ptr<int>> keep;
    for (int i = 0; i < 10; ++i) {
        auto o = std::make_shared<int>(i);
        insert(r, o, i);
        if (i % 2 == 0) keep.push_back(o);
    }
    ASSERT_EQ(r.size(), 10u);
    EXPECT_EQ(expiring::sweep(r, expired), 5u);
    EXPECT_EQ(r.size(), 5u);
    for (const auto& o : keep) EXPECT_EQ(lookup(r, o), *o);
}

TEST(ExpiringRegistry, SweepIfDoubledRunsOncePerDoublingAboveTheFloor) {
    Registry r;
    std::size_t last = 0, sweeps = 0, erased = 0;
    const std::size_t floor = 8, inserts = 1000;
    // every object dies right after its insert: without the sweep the map would grow by one per
    // insert. The sweep runs on the insert path, while the new object is still alive, so each
    // sweep drops the floor - 1 dead entries before it and leaves the map at one.
    for (std::size_t i = 0; i < inserts; ++i) {
        auto o = std::make_shared<int>(int(i));
        insert(r, o, int(i));
        const std::size_t n = expiring::sweep_if_doubled(r, last, expired, floor);
        if (n) {
            ++sweeps;
            erased += n;
            EXPECT_EQ(n, floor - 1);
            EXPECT_EQ(r.size(), 1u);
        }
        EXPECT_LE(r.size(), floor);
    }
    EXPECT_EQ(erased + r.size(), inserts);
    EXPECT_EQ(sweeps, 1 + (inserts - floor) / (floor - 1));   // the first at `floor`, then every floor - 1
}

TEST(ExpiringRegistry, TheMapStaysWithinTwiceThePeakLiveCountAtAmortisedCost) {
    Registry r;
    std::size_t last = 0, visits = 0;
    const std::size_t floor = 8, live = 100, inserts = 100000;
    const auto counting = [&](const Entry& e) {
        ++visits;
        return e.wp.expired();
    };
    std::vector<std::shared_ptr<int>> keep;
    for (int i = 0; i < int(live); ++i) {
        keep.push_back(std::make_shared<int>(i));
        insert(r, keep.back(), i);
    }
    for (std::size_t i = 0; i < inserts; ++i) {
        auto o = std::make_shared<int>(-1);
        insert(r, o, -1);
        expiring::sweep_if_doubled(r, last, counting, floor);
        ASSERT_LE(r.size(), 2 * live + floor);
    }
    for (const auto& o : keep) EXPECT_EQ(lookup(r, o), *o);   // the live entries survive every sweep
    // The doubling half of the trigger is what keeps the sweep amortised: with 100 live entries a
    // sweep walks about 200 nodes once per 100 inserts, two visits per insert. A trigger that
    // ignored `last` would walk the whole map on every insert above the floor (100 visits each).
    EXPECT_LE(visits, 3 * inserts);
}

TEST(ExpiringRegistry, ARecycledAddressIsNeverMislabelled) {
    Registry r;
    // an object made without make_shared frees its storage on expiry, so its address can be reused
    auto a = std::shared_ptr<int>(new int(7));
    insert(r, a, 7);
    a.reset();
    auto b = std::shared_ptr<int>(new int(8));   // may or may not land on a's address
    EXPECT_EQ(lookup(r, b), -1);                 // never 7: the read-side guard, or the sweep, stands between
    const std::size_t stale = r.size();          // whatever is left is a's expired entry
    EXPECT_EQ(expiring::sweep(r, expired), stale);
    EXPECT_TRUE(r.empty());
}
