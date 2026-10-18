// SPDX-License-Identifier: MIT
#include <device/compute.hpp>

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace compute = parallel_mater::device::compute;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function &&function) {
    try { function(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid handoff accepted");
}

struct CompletionState {
    bool done{};
    unsigned host_waits{};
};

class Buffer final : public compute::Buffer {
public:
    compute::ResourceIdentity identity() const noexcept override { return {7, 1}; }
    std::size_t size() const noexcept override { return 4; }
    void *host_address() const noexcept override { return nullptr; }
    std::uint64_t device_address() const noexcept override { return 0x1000; }
    void *native_handle() const noexcept override { return nullptr; }
};

class Completion final : public compute::Submission {
public:
    Completion(compute::QueueIdentity identity, std::shared_ptr<CompletionState> state)
        : identity_(identity), state_(std::move(state)) {}
    bool ready() const override { return state_->done; }
    void wait() const override { ++state_->host_waits; state_->done = true; }
    double elapsed_seconds() const override { return 0; }
    void *native_handle() const noexcept override { return state_.get(); }
    compute::QueueIdentity queue_identity() const noexcept override { return identity_; }
private:
    compute::QueueIdentity identity_;
    std::shared_ptr<CompletionState> state_;
};

struct EncoderState {
    std::vector<std::string> events;
    std::vector<compute::QueueIdentity> prior;
    std::shared_ptr<CompletionState> completion = std::make_shared<CompletionState>();
};

class Encoder final : public compute::CommandEncoder {
public:
    Encoder(compute::QueueIdentity identity, std::shared_ptr<EncoderState> state,
            bool support_cross_queue = true)
        : identity_(identity), state_(std::move(state)), support_cross_queue_(support_cross_queue) {}
    compute::QueueIdentity queue_identity() const noexcept override { return identity_; }
    void wait(std::span<const compute::SubmissionPtr> prior) override {
        state_->events.emplace_back("wait");
        for (const auto &completion : prior) {
            if (!support_cross_queue_ && completion->queue_identity().queue != identity_.queue)
                throw std::invalid_argument("adapter does not support cross-queue handoff");
            state_->prior.push_back(completion->queue_identity());
        }
    }
    void barrier(std::span<const compute::ResourceDependency>) override {}
    void dispatch(const compute::Arguments &, compute::Extent) override {}
    void copy(const compute::BufferPtr &, std::size_t, const compute::BufferPtr &,
              std::size_t, std::size_t) override {}
    void fill(const compute::BufferPtr &, std::size_t, std::size_t, std::uint8_t) override {
        state_->events.emplace_back("encode");
    }
    compute::BufferPtr upload(const compute::BufferPtr &, std::size_t,
                              std::span<const std::byte>) override { return {}; }
    compute::SubmissionPtr submit() override {
        state_->events.emplace_back("submit");
        return std::make_shared<Completion>(identity_, state_->completion);
    }
private:
    compute::QueueIdentity identity_;
    std::shared_ptr<EncoderState> state_;
    bool support_cross_queue_;
};

void release_count(void *context, std::uintptr_t handle) noexcept {
    auto *values = static_cast<std::array<std::uintptr_t, 2> *>(context);
    ++(*values)[0];
    (*values)[1] = handle;
}

void native_ownership() {
    std::array<std::uintptr_t, 2> released{};
    {
        auto borrowed = compute::NativeObject::borrowed(
            compute::NativeApi::d3d12, compute::NativeObjectKind::buffer, 17);
        require(borrowed.api() == compute::NativeApi::d3d12 &&
                borrowed.ownership() == compute::NativeOwnership::borrowed &&
                borrowed.handle() == 17, "borrowed native identity lost");
        borrowed.reset();
        require(released[0] == 0, "borrowed native object was released");
    }
    {
        auto owned = compute::NativeObject::owned(
            compute::NativeApi::vulkan, compute::NativeObjectKind::device,
            23, &released, release_count);
        auto moved = std::move(owned);
        require(!owned && moved.handle() == 23, "native ownership move failed");
    }
    require(released[0] == 1 && released[1] == 23,
            "owned native object was not released exactly once");
    rejects([] { (void)compute::NativeObject::borrowed(
        compute::NativeApi::d3d12, compute::NativeObjectKind::queue, 0); });
    rejects([] { (void)compute::NativeObject::owned(
        compute::NativeApi::d3d12, compute::NativeObjectKind::queue, 1, nullptr, nullptr); });
}

void prior_next_handoff_and_retention() {
    compute::SubmissionQueue queue;
    auto prior_state = std::make_shared<CompletionState>();
    std::weak_ptr<compute::Submission> prior_witness;
    auto encoder_state = std::make_shared<EncoderState>();
    std::unique_ptr<compute::CommandList> commands;
    {
        auto prior = std::make_shared<Completion>(
            compute::QueueIdentity{compute::NativeApi::mock, 7, 11}, prior_state);
        prior_witness = prior;
        const std::array<compute::SubmissionPtr, 1> prerequisites{prior};
        commands = queue.commands(
            std::make_unique<Encoder>(
                compute::QueueIdentity{compute::NativeApi::mock, 7, 19}, encoder_state),
            prerequisites);
    }

    require(!prior_witness.expired(), "recording did not retain prior completion");
    commands->fill(std::make_shared<Buffer>(), 0, 4, 0);
    require(encoder_state->events == std::vector<std::string>({"wait", "encode"}) &&
            encoder_state->prior == std::vector<compute::QueueIdentity>{
                {compute::NativeApi::mock, 7, 11}},
            "prior completion was not encoded before consumer work");
    require(prior_state->host_waits == 0, "handoff introduced a CPU wait");

    auto next = commands->submit();
    commands.reset();
    require(next->queue_identity() ==
                compute::QueueIdentity{compute::NativeApi::mock, 7, 19},
            "next completion queue identity lost");
    require(!prior_witness.expired(), "next completion did not retain prior completion");
    require(encoder_state->events ==
                std::vector<std::string>({"wait", "encode", "submit"}),
            "handoff submission order mismatch");
    encoder_state->completion->done = true;
    require(next->ready(), "next completion did not become ready");
    require(prior_witness.expired(), "terminal next completion retained prior completion");
    require(prior_state->host_waits == 0, "completion observation waited on prior CPU-side");
}

void compatibility_and_error_paths() {
    compute::SubmissionQueue queue;
    auto same_device = std::make_shared<Completion>(
        compute::QueueIdentity{compute::NativeApi::mock, 3, 5},
        std::make_shared<CompletionState>());
    auto other_device = std::make_shared<Completion>(
        compute::QueueIdentity{compute::NativeApi::mock, 4, 5},
        std::make_shared<CompletionState>());
    auto other_api = std::make_shared<Completion>(
        compute::QueueIdentity{compute::NativeApi::vulkan, 3, 5},
        std::make_shared<CompletionState>());

    auto foreign_device_state = std::make_shared<EncoderState>();
    rejects([&] {
        const std::array<compute::SubmissionPtr, 1> prior{other_device};
        (void)queue.commands(std::make_unique<Encoder>(
            compute::QueueIdentity{compute::NativeApi::mock, 3, 9},
            foreign_device_state), prior);
    });
    require(foreign_device_state->events.empty(),
            "foreign-device handoff reached adapter wait");
    auto foreign_api_state = std::make_shared<EncoderState>();
    rejects([&] {
        const std::array<compute::SubmissionPtr, 1> prior{other_api};
        (void)queue.commands(std::make_unique<Encoder>(
            compute::QueueIdentity{compute::NativeApi::mock, 3, 9},
            foreign_api_state), prior);
    });
    require(foreign_api_state->events.empty(), "foreign-API handoff reached adapter wait");
    auto null_state = std::make_shared<EncoderState>();
    rejects([&] {
        const std::array<compute::SubmissionPtr, 1> prior{compute::SubmissionPtr{}};
        (void)queue.commands(std::make_unique<Encoder>(
            compute::QueueIdentity{compute::NativeApi::mock, 3, 9},
            null_state), prior);
    });
    require(null_state->events.empty(), "null handoff reached adapter wait");
    rejects([&] {
        (void)queue.commands(std::make_unique<Encoder>(
            compute::QueueIdentity{}, std::make_shared<EncoderState>()));
    });
    auto unsupported_state = std::make_shared<EncoderState>();
    rejects([&] {
        const std::array<compute::SubmissionPtr, 1> prior{same_device};
        (void)queue.commands(std::make_unique<Encoder>(
            compute::QueueIdentity{compute::NativeApi::mock, 3, 9},
            unsupported_state, false), prior);
    });
    require(unsupported_state->events == std::vector<std::string>{"wait"},
            "unsupported adapter handoff submitted consumer work");
}
} // namespace

int main() {
    try {
        native_ownership();
        prior_next_handoff_and_retention();
        compatibility_and_error_paths();
        std::cout << "Borrowed native ownership and prior/next queue handoff passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
