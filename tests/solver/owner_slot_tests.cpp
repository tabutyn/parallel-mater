// SPDX-License-Identifier: MIT
#include "simulation/owner_slot.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); } } while(false)

namespace {

struct TestOwner {
    static inline std::size_t live_count{};

    explicit TestOwner(std::size_t payload_size)
        : payload(payload_size, std::byte{0x5a}) {
        ++live_count;
    }
    ~TestOwner() { --live_count; }

    std::uint32_t generation{1U};
    bool alive{true};
    std::vector<std::byte> payload;
};

struct TestHandle {
    std::size_t index{};
    std::uint32_t generation{};
};

using Slot = parallel_mater::simulation::OwnerSlot<TestOwner>;

[[nodiscard]] bool valid(const std::vector<Slot> &slots,
                         TestHandle handle) noexcept {
    return handle.index < slots.size() && slots[handle.index] &&
           slots[handle.index]->alive &&
           slots[handle.index]->generation == handle.generation;
}

[[nodiscard]] std::size_t first_available(
    const std::vector<Slot> &slots) noexcept {
    std::size_t index = 0U;
    while (index < slots.size() && slots[index] && slots[index]->alive)
        ++index;
    return index;
}

TestHandle publish(std::vector<Slot> &slots, std::size_t index,
                   std::size_t payload_size) {
    auto owner = std::make_unique<TestOwner>(payload_size);
    owner->generation = slots[index].generation();
    owner->alive = true;
    const TestHandle handle{index, owner->generation};
    slots[index] = std::move(owner);
    return handle;
}

} // namespace

int main() {
    std::vector<Slot> slots(2U);
    const auto *const fixed_data = slots.data();
    const auto fixed_size = slots.size();
    const auto fixed_capacity = slots.capacity();

    const TestHandle old = publish(slots, 0U, 64U * 1024U);
    const TestHandle other = publish(slots, 1U, 32U * 1024U);
    CHECK(TestOwner::live_count == 2U);
    CHECK(valid(slots, old) && valid(slots, other));
    CHECK(first_available(slots) == slots.size());

    slots[0].retire();
    CHECK(TestOwner::live_count == 1U);
    CHECK(!slots[0]);
    CHECK(!valid(slots, old));
    CHECK(slots[0].generation() != old.generation);

    const TestHandle replacement = publish(slots, 0U, 128U * 1024U);
    CHECK(replacement.index == old.index);
    CHECK(replacement.generation != old.generation);
    CHECK(valid(slots, replacement));
    CHECK(!valid(slots, old));
    slots[0].retire();
    slots[1].retire();
    CHECK(TestOwner::live_count == 0U);

    std::uint32_t previous_generation = slots[0].generation();
    for (std::size_t iteration = 0U; iteration < 4096U; ++iteration) {
        const TestHandle current = publish(slots, 0U, 4U * 1024U);
        CHECK(current.generation == previous_generation);
        CHECK(valid(slots, current));
        slots[0].retire();
        CHECK(TestOwner::live_count == 0U);
        CHECK(!valid(slots, current));
        CHECK(slots[0].generation() != previous_generation);
        previous_generation = slots[0].generation();
        CHECK(slots.data() == fixed_data);
        CHECK(slots.size() == fixed_size);
        CHECK(slots.capacity() == fixed_capacity);
    }

    std::puts("PASS: owner slots reclaim payloads, stay bounded and reject stale handles");
}
