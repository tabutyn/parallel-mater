// SPDX-License-Identifier: MIT
// Exercise the real Metal host scheduler with a non-GPU failure-injection device.
#include <driver/metal/schedule_context.hpp>

#include <iostream>
#include <utility>

namespace compute = parallel_mater::device::compute;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
struct State {
    bool fail_encode{}, fail_submit{};
    unsigned encoders{}, accepted_fills{};
};
class Buffer final : public compute::Buffer {
public:
    explicit Buffer(std::size_t size) : bytes_(size) {}
    compute::ResourceIdentity identity() const noexcept override { return {1, device_address()}; }
    std::size_t size() const noexcept override { return bytes_.size(); }
    void *host_address() const noexcept override { return const_cast<std::byte *>(bytes_.data()); }
    std::uint64_t device_address() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(bytes_.data());
    }
    void *native_handle() const noexcept override { return host_address(); }
private:
    std::vector<std::byte> bytes_;
};
class Completion final : public compute::Submission {
public:
    bool ready() const override { return done_; }
    void wait() const override { done_ = true; }
    double elapsed_seconds() const override { return 0; }
    void *native_handle() const noexcept override { return nullptr; }
private:
    mutable bool done_{};
};
class Encoder final : public compute::CommandEncoder {
public:
    explicit Encoder(State &state) : state_(state) { ++state_.encoders; }
    void dispatch(const compute::Arguments &, compute::Extent) override {}
    void copy(const compute::BufferPtr &, std::size_t, const compute::BufferPtr &,
              std::size_t, std::size_t) override {}
    compute::BufferPtr upload(const compute::BufferPtr &, std::size_t,
                             std::span<const std::byte>) override { return {}; }
    void fill(const compute::BufferPtr &, std::size_t, std::size_t, std::uint8_t) override {
        if (std::exchange(state_.fail_encode, false)) throw std::bad_alloc();
        ++fills_;
    }
    compute::SubmissionPtr submit() override {
        if (std::exchange(state_.fail_submit, false)) throw std::bad_alloc();
        auto token = std::make_shared<Completion>();
        state_.accepted_fills += fills_;
        return token;
    }
private:
    State &state_;
    unsigned fills_{};
};
class Device final : public compute::Device {
public:
    explicit Device(State &state) : state_(state) {}
    std::string name() const override { return "failure-injection mock"; }
    compute::BufferPtr allocate(std::size_t size, bool) override { return std::make_shared<Buffer>(size); }
    compute::KernelPtr load_kernel(const std::string &, const std::string &, const std::string &) override {
        throw std::logic_error("mock does not execute shaders");
    }
    std::unique_ptr<compute::CommandList> commands() override {
        return queue_.commands(std::make_unique<Encoder>(state_));
    }
    void *native_device() const noexcept override { return nullptr; }
    void *native_queue() const noexcept override { return nullptr; }
private:
    State &state_;
    compute::SubmissionQueue queue_;
};
void recovery() {
    State state;
    parallel_mater::metal::ScheduleContext context(std::make_unique<Device>(state));
    auto *buffer = context.allocate(64);
    const auto fill = [&] { context.clear(buffer, 17, 64); };
    require(context.checked(fill, false) == cudaSuccess, "initial recording failed");
    state.fail_submit = true;
    require(context.checked([&] { (void)context.submit(); }, false) == cudaErrorMemoryAllocation,
        "submit failure not reported");
    require(state.accepted_fills == 0, "failed submission accepted work");
    require(context.checked([&] { fill(); context.wait(); }, false) == cudaSuccess,
        "scheduler retained cancelled command after failed submit");
    require(state.accepted_fills == 1 && state.encoders == 2, "submit recovery did not use fresh recording");

    state.fail_encode = true;
    require(context.checked(fill, false) == cudaErrorMemoryAllocation, "encoding failure not reported");
    require(context.checked([&] { fill(); context.wait(); }, false) == cudaSuccess,
        "scheduler retained cancelled command after failed encoding");
    require(state.accepted_fills == 2, "encoding recovery did not accept replacement work");

    require(context.checked(fill, false) == cudaSuccess, "recording before validation failure failed");
    const auto encoders = state.encoders;
    require(context.checked([&] { context.clear(buffer, 0, 65); }, false) == cudaErrorInvalidValue,
        "invalid transfer was accepted");
    require(context.checked([&] { context.wait(); }, false) == cudaSuccess &&
        state.encoders == encoders && state.accepted_fills == 3,
        "validation failure discarded earlier valid recording");
    require(context.error(false) == cudaSuccess, "public service failure poisoned sticky launch error");
    context.free(buffer);
}
} // namespace
int main() {
    try {
        recovery();
        std::cout << "Metal scheduler encoding/submission failure recovery passed (mock device)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
