// SPDX-License-Identifier: MIT
#include <device/compute.hpp>

#include <array>
#include <iostream>
#include <string>

namespace compute = parallel_mater::device::compute;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects_invalid(Function &&function) {
    try { function(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid dependency operation accepted");
}

class Buffer final : public compute::Buffer {
public:
    explicit Buffer(std::uint64_t allocation, std::size_t size = 64,
                    std::uint64_t device = 1, std::uint64_t address = 0)
        : identity_{device, allocation}, size_(size),
          address_(address ? address : 0x100000 + allocation * 0x1000) {}
    compute::ResourceIdentity identity() const noexcept override { return identity_; }
    std::size_t size() const noexcept override { return size_; }
    void *host_address() const noexcept override { return nullptr; }
    std::uint64_t device_address() const noexcept override { return address_; }
    void *native_handle() const noexcept override { return nullptr; }
private:
    compute::ResourceIdentity identity_;
    std::size_t size_;
    std::uint64_t address_;
};

struct Event {
    std::string name;
    std::vector<compute::ResourceDependency> dependencies;
};
struct State {
    std::vector<Event> events;
    unsigned waits{};
};
class Completion final : public compute::Submission {
public:
    explicit Completion(std::shared_ptr<State> state) : state_(std::move(state)) {}
    bool ready() const override { return true; }
    void wait() const override { ++state_->waits; }
    double elapsed_seconds() const override { return 0; }
    void *native_handle() const noexcept override { return nullptr; }
private:
    std::shared_ptr<State> state_;
};
class Encoder final : public compute::CommandEncoder {
public:
    explicit Encoder(std::shared_ptr<State> state) : state_(std::move(state)) {}
    void barrier(std::span<const compute::ResourceDependency> dependencies) override {
        state_->events.push_back({"barrier", {dependencies.begin(), dependencies.end()}});
    }
    void dispatch(const compute::Arguments &, compute::Extent) override {
        state_->events.push_back({"dispatch", {}});
    }
    void copy(const compute::BufferPtr &, std::size_t, const compute::BufferPtr &,
              std::size_t, std::size_t) override {
        state_->events.push_back({"copy", {}});
    }
    void fill(const compute::BufferPtr &, std::size_t, std::size_t, std::uint8_t) override {
        state_->events.push_back({"fill", {}});
    }
    compute::BufferPtr upload(const compute::BufferPtr &, std::size_t,
                              std::span<const std::byte>) override {
        state_->events.push_back({"upload", {}});
        return {};
    }
    compute::SubmissionPtr submit() override {
        state_->events.push_back({"submit", {}});
        return std::make_shared<Completion>(state_);
    }
private:
    std::shared_ptr<State> state_;
};

compute::KernelPtr pointer_kernel(bool two = false) {
    auto kernel = std::make_shared<compute::Kernel>();
    kernel->name = "dependencies";
    kernel->argument_size = two ? 16 : 8;
    compute::Field pointer;
    pointer.size = pointer.alignment = 8;
    pointer.pointer = true;
    pointer.pointee_size = pointer.pointee_alignment = 4;
    kernel->parameters.emplace("first", pointer);
    if (two) {
        pointer.offset = 8;
        kernel->parameters.emplace("second", pointer);
    }
    return kernel;
}

compute::Arguments access(const compute::BufferPtr &buffer, std::size_t offset,
                          std::size_t size, compute::ResourceAccess mode) {
    compute::Arguments arguments(pointer_kernel());
    arguments.bind("first", buffer, offset, size, mode);
    return arguments;
}

void expect_dependency(const compute::ResourceDependency &dependency,
                       compute::ResourceIdentity identity, std::size_t offset,
                       std::size_t size, compute::ResourceAccess prior,
                       compute::ResourceAccess next) {
    require(dependency.prior.identity() == identity &&
        dependency.next.identity() == identity, "dependency identity mismatch");
    require(dependency.prior.offset == offset && dependency.next.offset == offset &&
        dependency.prior.size == size && dependency.next.size == size,
        "dependency intersection mismatch");
    require(dependency.prior.access == prior && dependency.next.access == next,
        "dependency access transition mismatch");
}

void raw_war_waw_and_order() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = queue.commands(std::make_unique<Encoder>(state));
    auto buffer = std::make_shared<Buffer>(11);

    command->fill(buffer, 0, 16, 1);
    auto read = access(buffer, 8, 16, compute::ResourceAccess::read);
    command->dispatch(read, {1});
    auto repeated_read = access(buffer, 12, 4, compute::ResourceAccess::read);
    command->dispatch(repeated_read, {1});
    command->fill(buffer, 4, 12, 2);
    auto completion = command->submit();

    require(state->events.size() == 7, "unexpected dependency event count");
    require(state->events[0].name == "fill" && state->events[1].name == "barrier" &&
        state->events[2].name == "dispatch" && state->events[3].name == "dispatch" &&
        state->events[4].name == "barrier" && state->events[5].name == "fill" &&
        state->events[6].name == "submit", "dependency was not recorded before its consumer");
    require(state->events[1].dependencies.size() == 1, "RAW dependency count mismatch");
    expect_dependency(state->events[1].dependencies[0], buffer->identity(), 8, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
    require(state->events[4].dependencies.size() == 2, "WAR/WAW dependency count mismatch");
    expect_dependency(state->events[4].dependencies[0], buffer->identity(), 4, 4,
        compute::ResourceAccess::write, compute::ResourceAccess::write);
    expect_dependency(state->events[4].dependencies[1], buffer->identity(), 8, 8,
        compute::ResourceAccess::read, compute::ResourceAccess::write);
    require(completion->ready() && state->waits == 0,
        "resource dependency introduced a host wait");
    queue.collect();
    require(state->waits == 0, "nonblocking collection introduced a host wait");
}

void independent_and_read_only_stay_unordered() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = queue.commands(std::make_unique<Encoder>(state));
    auto a = std::make_shared<Buffer>(21);
    auto b = std::make_shared<Buffer>(22);
    auto c = std::make_shared<Buffer>(23);

    auto read_a = access(a, 0, 16, compute::ResourceAccess::read);
    command->dispatch(read_a, {1});
    auto overlapping_read_a = access(a, 8, 16, compute::ResourceAccess::read);
    command->dispatch(overlapping_read_a, {1});
    command->fill(b, 0, 16, 3);
    command->fill(a, 32, 8, 4);
    command->copy(c, 0, a, 0, 8);
    auto completion = command->submit();

    for (const auto &event : state->events)
        require(event.name != "barrier", "independent/read-only work gained a barrier");
    require(completion->ready() && state->waits == 0,
        "independent resources introduced a global wait");
    queue.collect();
}

void partial_history_and_same_command_aliases() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = queue.commands(std::make_unique<Encoder>(state));
    auto buffer = std::make_shared<Buffer>(31);
    auto alias = std::make_shared<Buffer>(31, buffer->size(), 1, buffer->device_address());

    command->fill(buffer, 0, 16, 1);
    auto upper = access(buffer, 8, 8, compute::ResourceAccess::read);
    command->dispatch(upper, {1});
    auto lower = access(alias, 0, 8, compute::ResourceAccess::read);
    command->dispatch(lower, {1});

    compute::Arguments mixed(pointer_kernel(true));
    mixed.bind("first", buffer, 24, 16, compute::ResourceAccess::read);
    mixed.bind("second", alias, 32, 16, compute::ResourceAccess::write);
    command->dispatch(mixed, {1});
    auto mixed_read = access(buffer, 28, 16, compute::ResourceAccess::read);
    command->dispatch(mixed_read, {1});
    (void)command->submit();

    std::vector<const Event *> barriers;
    for (const auto &event : state->events)
        if (event.name == "barrier") barriers.push_back(&event);
    require(barriers.size() == 3, "range history or same-command alias scheduling mismatch");
    expect_dependency(barriers[0]->dependencies.at(0), buffer->identity(), 8, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
    expect_dependency(barriers[1]->dependencies.at(0), buffer->identity(), 0, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
    require(barriers[2]->dependencies.size() == 2,
        "mixed same-command access was not normalized into exact slices");
    expect_dependency(barriers[2]->dependencies[0], buffer->identity(), 32, 8,
        compute::ResourceAccess::read_write, compute::ResourceAccess::read);
    expect_dependency(barriers[2]->dependencies[1], buffer->identity(), 40, 4,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
}

void invalid_alias_keeps_prior_recording() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = queue.commands(std::make_unique<Encoder>(state));
    auto buffer = std::make_shared<Buffer>(41);
    auto inconsistent = std::make_shared<Buffer>(41, buffer->size(), 1,
                                                  buffer->device_address() + 4);
    command->fill(buffer, 0, 8, 1);
    rejects_invalid([&] { command->fill(inconsistent, 0, 8, 2); });
    require(command->is_recording(), "dependency validation discarded prior commands");
    command->fill(buffer, 0, 8, 3);
    (void)command->submit();
    require(state->events.size() == 4 && state->events[0].name == "fill" &&
        state->events[1].name == "barrier" && state->events[2].name == "fill" &&
        state->events[3].name == "submit",
        "valid work did not survive rejected alias metadata");
}

void transfers_supply_read_and_write_accesses() {
    compute::SubmissionQueue queue;
    auto state = std::make_shared<State>();
    auto command = queue.commands(std::make_unique<Encoder>(state));
    auto source = std::make_shared<Buffer>(51);
    auto destination = std::make_shared<Buffer>(52);

    command->fill(source, 0, 8, 1);
    command->copy(destination, 0, source, 0, 8);
    const std::array bytes{
        std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
        std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8},
    };
    command->upload(destination, 0, bytes);
    auto read = access(destination, 0, 8, compute::ResourceAccess::read);
    command->dispatch(read, {1});
    (void)command->submit();

    require(state->events.size() == 8, "transfer dependency event count mismatch");
    require(state->events[0].name == "fill" && state->events[1].name == "barrier" &&
        state->events[2].name == "copy" && state->events[3].name == "barrier" &&
        state->events[4].name == "upload" && state->events[5].name == "barrier" &&
        state->events[6].name == "dispatch" && state->events[7].name == "submit",
        "transfer dependency ordering mismatch");
    expect_dependency(state->events[1].dependencies.at(0), source->identity(), 0, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
    expect_dependency(state->events[3].dependencies.at(0), destination->identity(), 0, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::write);
    expect_dependency(state->events[5].dependencies.at(0), destination->identity(), 0, 8,
        compute::ResourceAccess::write, compute::ResourceAccess::read);
}
} // namespace

int main() {
    try {
        raw_war_waw_and_order();
        independent_and_read_only_stay_unordered();
        partial_history_and_same_command_aliases();
        invalid_alias_keeps_prior_recording();
        transfers_supply_read_and_write_accesses();
        std::cout << "Common range dependency scheduling passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
