// SPDX-License-Identifier: MIT
#include <device/compute.hpp>

#include <array>
#include <condition_variable>
#include <iostream>
#include <thread>

namespace compute = parallel_mater::device::compute;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Exception = std::exception, class Function> void rejects(Function &&function) {
    try { function(); }
    catch (const Exception &) { return; }
    throw std::runtime_error("invalid lifetime operation accepted");
}
class Buffer final : public compute::Buffer {
public:
    compute::ResourceIdentity identity() const noexcept override { return {1, device_address()}; }
    std::size_t size() const noexcept override { return 64; }
    void *host_address() const noexcept override { return nullptr; }
    std::uint64_t device_address() const noexcept override {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this));
    }
    void *native_handle() const noexcept override { return nullptr; }
};

// Deliberately owns NO resources: only the common lifetime layer can keep
// these weak witnesses alive. No GPU or native API is involved in this test.
struct State {
    bool done{}, failed{}, fail_submit{}, null_submit{}, fail_encode{};
    unsigned submitted{}, waited{}, destroyed{};
    std::vector<std::weak_ptr<const void>> resources;
    std::vector<std::byte> snapshot;
    std::mutex wait_mutex;
    std::condition_variable wait_condition;
    bool block_wait{}, waiting{}, allow_wait{};
    template<class T> void observe(const std::shared_ptr<T> &resource) { resources.emplace_back(resource); }
    bool alive() const {
        for (const auto &resource : resources) if (resource.expired()) return false;
        return true;
    }
    bool expired() const {
        for (const auto &resource : resources) if (!resource.expired()) return false;
        return true;
    }
};
class Completion final : public compute::Submission {
public:
    explicit Completion(std::shared_ptr<State> state) : state_(std::move(state)) {}
    bool ready() const override {
        if (state_->done && state_->failed) throw std::runtime_error("deferred GPU failure");
        return state_->done;
    }
    void wait() const override {
        ++state_->waited;
        if (state_->block_wait) {
            std::unique_lock lock(state_->wait_mutex);
            state_->waiting = true;
            state_->wait_condition.notify_one();
            state_->wait_condition.wait(lock, [&] { return state_->allow_wait; });
        }
        if (!state_->done) require(state_->alive(), "pending resource freed before native wait");
        state_->done = true;
        (void)ready();
    }
    double elapsed_seconds() const override {
        if (!ready()) throw std::logic_error("timestamps not ready");
        return 0.125;
    }
    void *native_handle() const noexcept override { return state_.get(); }
private:
    std::shared_ptr<State> state_;
};
class Encoder final : public compute::CommandEncoder {
public:
    explicit Encoder(std::shared_ptr<State> state) : state_(std::move(state)) {}
    ~Encoder() override { ++state_->destroyed; }
    void dispatch(const compute::Arguments &args, compute::Extent) override {
        state_->observe(args.kernel());
        for (const auto &binding : args.bindings()) state_->observe(binding.range.buffer);
        fail_if_requested();
    }
    void copy(const compute::BufferPtr &dst, std::size_t, const compute::BufferPtr &src,
              std::size_t, std::size_t) override {
        state_->observe(dst);
        state_->observe(src);
        fail_if_requested();
    }
    void fill(const compute::BufferPtr &dst, std::size_t, std::size_t, std::uint8_t) override {
        state_->observe(dst);
        fail_if_requested();
    }
    compute::BufferPtr upload(const compute::BufferPtr &dst, std::size_t,
                             std::span<const std::byte> bytes) override {
        auto staging = std::make_shared<Buffer>();
        state_->snapshot.assign(bytes.begin(), bytes.end());
        state_->observe(dst);
        state_->observe(staging);
        fail_if_requested();
        return staging;
    }
    compute::SubmissionPtr submit() override {
        require(state_->alive(), "resource freed before native acceptance");
        if (state_->fail_submit) throw std::bad_alloc();
        if (state_->null_submit) return {};
        auto result = std::make_shared<Completion>(state_);
        ++state_->submitted;
        return result;
    }
private:
    void fail_if_requested() const {
        if (state_->fail_encode) throw std::runtime_error("encoding failure");
    }
    std::shared_ptr<State> state_;
};
std::unique_ptr<compute::CommandList> commands(compute::SubmissionQueue &queue,
                                             const std::shared_ptr<State> &state) {
    return queue.commands(std::make_unique<Encoder>(state));
}
void encode_all_resources(compute::CommandList &command) {
    auto direct = std::make_shared<Buffer>();
    auto nested = std::make_shared<Buffer>();
    auto indirect = std::make_shared<Buffer>();
    auto kernel = std::make_shared<compute::Kernel>();
    compute::Field pointer;
    pointer.size = pointer.alignment = sizeof(std::uint64_t);
    pointer.pointer = true;
    pointer.pointee_size = pointer.pointee_alignment = 4;
    compute::Field record;
    record.offset = 8;
    record.size = record.alignment = 8;
    record.fields.emplace("pointer", pointer);
    kernel->argument_size = 16;
    kernel->parameters.emplace("direct", pointer);
    kernel->parameters.emplace("record", record);
    const std::uint64_t unused = 0;
    compute::Record value(record, unused);
    value.bind("pointer", nested);
    compute::Arguments args(kernel);
    args.bind("direct", direct);
    args.record("record", value);
    args.reference("indirect", indirect);
    command.dispatch(args, {1});
    command.copy(direct, 0, nested, 0, 8);
    command.fill(indirect, 0, 8, 17);
    std::array<std::byte, 8> source;
    source.fill(std::byte{37});
    command.upload(indirect, 8, source);
    source.fill(std::byte{99});
}
void dropped_handles_and_tokens() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = commands(queue, state);
    encode_all_resources(*command);
    require(command->is_recording(), "valid command lost recording state");
    require(state->alive(), "command did not retain direct/record/indirect/kernel/staging owners");
    require(state->snapshot[0] == std::byte{37}, "upload did not snapshot caller memory");
    auto completion = command->submit();
    require(!command->is_recording(), "submitted command still recording");
    command.reset();
    require(!completion->ready(), "unfinished submission reported ready");
    require(completion->native_handle() == state.get(), "native completion handle lost");
    rejects<std::logic_error>([&] { (void)completion->elapsed_seconds(); });
    completion.reset();
    queue.collect();
    require(state->waited == 0 && state->alive(), "dropping token waited or freed pending owners");
    state->done = true;
    queue.collect();
    require(state->expired(), "completed dropped-token owners not reclaimed");
}
void observed_completion_and_shared_resources() {
    compute::SubmissionQueue queue;
    auto buffer = std::make_shared<Buffer>();
    const std::weak_ptr<Buffer> weak = buffer;
    auto first = std::make_shared<State>(), second = std::make_shared<State>();
    auto a = commands(queue, first), b = commands(queue, second);
    a->fill(buffer, 0, 4, 1);
    b->fill(buffer, 0, 4, 2);
    auto token_a = a->submit(), token_b = b->submit();
    buffer.reset();
    first->done = true;
    require(token_a->ready() && !weak.expired(), "first completion freed another submission's buffer");
    token_b->wait();
    require(weak.expired(), "wait retained completed owners while token remained alive");
    require(token_b->elapsed_seconds() == 0.125, "native timing lost");
    rejects<std::logic_error>([&] { (void)b->submit(); });
    rejects<std::logic_error>([&] { b->fill({}, 0, 4, 0); });
}
void cancellation_and_failures() {
    compute::SubmissionQueue queue;
    for (unsigned mode = 0; mode < 4; ++mode) {
        auto state = std::make_shared<State>();
        auto command = commands(queue, state);
        encode_all_resources(*command);
        if (mode == 0) command.reset(); // Unsubmitted list cancels, never waits.
        if (mode == 1) {
            state->fail_submit = true;
            rejects<std::bad_alloc>([&] { (void)command->submit(); });
        }
        if (mode == 2) {
            state->null_submit = true;
            rejects<std::runtime_error>([&] { (void)command->submit(); });
        }
        if (mode == 3) {
            state->fail_encode = true;
            auto buffer = std::make_shared<Buffer>();
            rejects<std::runtime_error>([&] { command->fill(buffer, 0, 4, 0); });
        }
        require(state->expired() && state->destroyed == 1 && state->submitted == 0 &&
            state->waited == 0, "cancel/failure leaked or waited on unsubmitted resources");
        if (command) {
            require(!command->is_recording(), "failed command still recording");
            rejects<std::logic_error>([&] { (void)command->submit(); });
        }
    }
    for (bool observe_with_wait : {false, true}) {
        auto state = std::make_shared<State>();
        auto command = commands(queue, state);
        encode_all_resources(*command);
        auto token = command->submit();
        state->failed = true;
        if (observe_with_wait) rejects<std::runtime_error>([&] { token->wait(); });
        else {
            state->done = true;
            rejects<std::runtime_error>([&] { (void)token->ready(); });
        }
        require(state->expired(), "terminal execution failure retained owners");
        queue.collect();
        rejects<std::runtime_error>([&] { (void)token->ready(); });
    }
}
void validation_and_reclamation() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto buffer = std::make_shared<Buffer>();
    auto command = commands(queue, state);
    command->fill(buffer, 0, 4, 0);
    rejects<std::invalid_argument>([&] { command->fill(buffer, 63, 2, 0); });
    rejects<std::invalid_argument>([&] { command->copy(buffer, 0, buffer, 1, 4); });
    rejects<std::invalid_argument>([&] { command->upload(buffer, SIZE_MAX, std::array<std::byte, 8>{}); });
    rejects<std::invalid_argument>([&] { command->fill({}, 0, 0, 0); });
    command->fill(buffer, 64, 0, 0);
    (void)command->submit();
    buffer.reset();
    state->done = true;
    auto next_state = std::make_shared<State>();
    auto next = commands(queue, next_state);
    require(state->expired(), "new recording did not reclaim completed dropped-token work");
    (void)next->submit(); // Empty command lists have valid completion ownership.
}
void device_teardown() {
    std::array<std::shared_ptr<State>, 3> states;
    compute::SubmissionPtr retained_token;
    std::unique_ptr<compute::CommandList> unsubmitted;
    auto unsubmitted_state = std::make_shared<State>();
    {
        // This owner is the last member of the native device; its destructor
        // drains before the native queue/device members are destroyed.
        compute::SubmissionQueue device_queue;
        for (auto &state : states) {
            state = std::make_shared<State>();
            auto command = commands(device_queue, state);
            encode_all_resources(*command);
            retained_token = command->submit();
        }
        states[0]->failed = true; // A drain error must not skip later work.
        unsubmitted = commands(device_queue, unsubmitted_state);
        encode_all_resources(*unsubmitted);
    }
    for (const auto &state : states)
        require(state->waited == 1 && state->done && state->expired(), "device teardown failed to drain owners");
    require(retained_token->ready(), "completion cannot be used after device teardown");
    rejects<std::logic_error>([&] { (void)unsubmitted->submit(); });
    rejects<std::logic_error>([&] { encode_all_resources(*unsubmitted); });
    unsubmitted.reset();
    require(unsubmitted_state->expired() && unsubmitted_state->submitted == 0,
        "surviving command enqueued after device teardown or leaked its owners");
}
void concurrent_poll_does_not_wait() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = commands(queue, state);
    encode_all_resources(*command);
    auto completion = command->submit();
    state->block_wait = true;
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        try { completion->wait(); } catch (...) { wait_error = std::current_exception(); }
    });
    {
        std::unique_lock lock(state->wait_mutex);
        state->wait_condition.wait(lock, [&] { return state->waiting; });
    }
    // Native wait is deliberately blocked until AFTER these operations. A
    // blocking ready()/collect() implementation deadlocks and fails the timeout.
    const bool pending = !completion->ready();
    rejects<std::logic_error>([&] { (void)completion->elapsed_seconds(); });
    queue.collect();
    auto next = commands(queue, std::make_shared<State>());
    (void)next->submit();
    {
        std::lock_guard lock(state->wait_mutex);
        state->allow_wait = true;
    }
    state->wait_condition.notify_one();
    waiter.join();
    if (wait_error) std::rethrow_exception(wait_error);
    require(pending && state->expired(), "concurrent polling violated completion ownership");
}
} // namespace

int main() {
    try {
        dropped_handles_and_tokens();
        observed_completion_and_shared_resources();
        cancellation_and_failures();
        validation_and_reclamation();
        device_teardown();
        concurrent_poll_does_not_wait();
        std::cout << "Common command/completion lifetime, cancellation and teardown passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
