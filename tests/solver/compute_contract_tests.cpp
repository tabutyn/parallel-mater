// SPDX-License-Identifier: MIT
#include <device/compute.hpp>
#include <simulation/avbd_schedule.hpp>

#include <array>
#include <iostream>
#include <tuple>

namespace compute = parallel_mater::device::compute;
using namespace parallel_mater::simulation;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function &&function) {
    try { function(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid compute operation accepted");
}
class TestBuffer final : public compute::Buffer {
public:
    explicit TestBuffer(compute::ResourceIdentity identity = {7, 11},
                        std::size_t size = 128, std::uint64_t address = 0x1000)
        : identity_(identity), size_(size), address_(address) {}
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
class RecordingCommandList final : public compute::CommandEncoder {
public:
    compute::QueueIdentity queue_identity() const noexcept override {
        return {compute::NativeApi::mock, 1, 1};
    }
    void wait(std::span<const compute::SubmissionPtr>) override {}
    void barrier(std::span<const compute::ResourceDependency>) override {}
    void dispatch(const compute::Arguments &arguments, compute::Extent) override {
        arguments.validate();
        observed = arguments.bindings();
    }
    void copy(const compute::BufferPtr &, std::size_t, const compute::BufferPtr &,
              std::size_t, std::size_t) override {}
    void fill(const compute::BufferPtr &, std::size_t, std::size_t, std::uint8_t) override {}
    compute::BufferPtr upload(const compute::BufferPtr &, std::size_t,
                             std::span<const std::byte>) override { return {}; }
    compute::SubmissionPtr submit() override { return {}; }
    std::vector<compute::ResourceBinding> observed;
};
void arguments_and_records() {
    auto kernel = std::make_shared<compute::Kernel>();
    kernel->name = "contract";
    kernel->argument_size = 32;
    compute::Field scalar;
    scalar.size = scalar.alignment = 4;
    compute::Field pointer;
    pointer.size = pointer.alignment = 8;
    pointer.pointer = true;
    pointer.pointee_size = pointer.pointee_alignment = 4;
    pointer.offset = 8;
    kernel->parameters.emplace("count", scalar);
    kernel->parameters.emplace("values", pointer);
    compute::Arguments args(kernel);
    rejects([&] { args.validate(); });
    rejects([&] { args.set("missing", 1U); });
    rejects([&] { args.set("count", std::uint64_t{1}); });
    rejects([&] { args.set("values", std::uint64_t{0x1000}); });
    auto buffer = std::make_shared<TestBuffer>();
    rejects([&] { args.bind("count", buffer); });
    rejects([&] { args.bind("values", buffer, 1); });
    rejects([&] { args.bind("values", buffer, 128); });
    rejects([&] { args.bind("values", nullptr, 1); });
    rejects([&] { args.bind("values", nullptr, 0, 4); });
    rejects([&] { args.bind("values", buffer, 4, 2); });
    rejects([&] { args.bind("values", buffer, 4, 125); });
    rejects([&] { args.bind("values", buffer, 4, 4,
        static_cast<compute::ResourceAccess>(0)); });
    args.set("count", 31U);
    args.bind("values", buffer, 4, 32, compute::ResourceAccess::read);
    args.validate();
    std::uint64_t encoded;
    std::memcpy(&encoded, args.bytes().data() + 8, 8);
    require(encoded == 0x1004, "device address offset lost");
    require(args.resources().size() == 1, "buffer residency lost");
    const auto direct = args.bindings();
    require(direct.size() == 1 && direct[0].name == "values" &&
        direct[0].kind == compute::ResourceBindingKind::parameter &&
        direct[0].range.offset == 4 && direct[0].range.size == 32 &&
        direct[0].range.access == compute::ResourceAccess::read,
        "typed parameter range lost");
    args.bind("values", nullptr);
    require(args.resources().empty(), "rebound pointer retained stale buffer");

    args.bind("values", buffer, 4, 32, compute::ResourceAccess::read);
    auto alias = std::make_shared<TestBuffer>(buffer->identity(), buffer->size(), buffer->device_address());
    auto other_device = std::make_shared<TestBuffer>(compute::ResourceIdentity{8, 11}, 128, 0x1000);
    args.reference("table.output", alias, 16, 16, compute::ResourceAccess::write);
    args.reference("other.input", other_device, 16, 16, compute::ResourceAccess::read);
    rejects([&] { args.reference("", buffer, 0, 4); });
    rejects([&] { args.reference("bad.range", buffer, 127, 2); });
    RecordingCommandList mock;
    mock.dispatch(args, {1});
    require(mock.observed.size() == 3, "mock adapter missed a resource binding");
    require(mock.observed[0].name == "table.output" &&
        mock.observed[0].kind == compute::ResourceBindingKind::indirect &&
        mock.observed[0].range.offset == 16 && mock.observed[0].range.size == 16 &&
        mock.observed[0].range.access == compute::ResourceAccess::write,
        "mock adapter lost complete indirect resource metadata");
    const auto parameter = mock.observed[2];
    require(compute::aliases(parameter.range, mock.observed[0].range) &&
        compute::overlaps(parameter.range, mock.observed[0].range),
        "same-allocation overlapping aliases not identified");
    require(!compute::aliases(parameter.range, mock.observed[1].range),
        "allocation values on different devices incorrectly alias");
    require(!compute::overlaps(compute::resource_range(buffer, 0, 4),
        compute::resource_range(alias, 4, 4)), "adjacent half-open ranges overlap");

    compute::Arguments inconsistent(kernel);
    inconsistent.set("count", 1U);
    inconsistent.bind("values", buffer, 0, 4);
    inconsistent.reference("bad.alias",
        std::make_shared<TestBuffer>(buffer->identity(), 64, buffer->device_address()), 0, 4);
    rejects([&] { (void)inconsistent.bindings(); });

    struct Host { std::uint32_t flag, padding; std::uint64_t pointers[2]; } host{1, 0, {0xdead, 0xbeef}};
    compute::Field layout;
    layout.size = sizeof(Host);
    layout.alignment = alignof(Host);
    layout.fields.emplace("flag", scalar);
    compute::Field array;
    array.offset = 8;
    array.size = 16;
    array.alignment = 8;
    array.element_count = 2;
    array.element_stride = 8;
    array.element = std::make_shared<compute::Field>(pointer);
    layout.fields.emplace("pointers", array);
    compute::Record record(layout, host);
    rejects([&] { (void)record.bytes(); });
    record.bind("pointers[0]", buffer, 4, 32, compute::ResourceAccess::read);
    rejects([&] { (void)record.bytes(); });
    record.bind("pointers[1]", nullptr);
    Host relocated{};
    std::memcpy(&relocated, record.bytes().data(), sizeof(Host));
    require(relocated.flag == 1 && relocated.pointers[0] == 0x1004 && relocated.pointers[1] == 0,
        "record relocation failed");
    require(host.pointers[0] == 0xdead && host.pointers[1] == 0xbeef, "host record mutated");
    kernel->parameters.clear();
    kernel->parameters.emplace("record", layout);
    compute::Arguments nested(kernel);
    rejects([&] { nested.set("record", host); });
    nested.record("record", record);
    nested.validate();
    require(nested.resources().size() == 1, "record residency lost");
    const auto record_bindings = nested.bindings();
    require(record_bindings.size() == 1 && record_bindings[0].name == "record.pointers[0]" &&
        record_bindings[0].kind == compute::ResourceBindingKind::record_pointer &&
        record_bindings[0].range.offset == 4 && record_bindings[0].range.size == 32 &&
        record_bindings[0].range.access == compute::ResourceAccess::read,
        "typed record-pointer range lost");
    auto wrong_layout = layout;
    wrong_layout.fields.at("pointers").offset = 0;
    compute::Record wrong(wrong_layout, host);
    wrong.bind("pointers[0]", buffer);
    wrong.bind("pointers[1]", buffer);
    rejects([&] { nested.record("record", wrong); });
}
void schedule() {
    using Step = std::tuple<AvbdStage, std::uint32_t, std::uint32_t, std::uint32_t>;
    std::vector<Step> steps;
    encode_avbd_schedule(2, 3, [&](auto... values) { steps.emplace_back(values...); });
    require(steps.size() == 30, "AVBD schedule dispatch count wrong");
    require(steps.front() == Step{AvbdStage::solve_color, 0, 0, 0}, "AVBD first color wrong");
    require(steps[3] == Step{AvbdStage::advance_duals, 0, 0, 0}, "AVBD dual ordering wrong");
    require(steps[7] == Step{AvbdStage::advance_duals, 0, 1, 0}, "AVBD iteration ordering wrong");
    require(steps[8] == Step{AvbdStage::free_mean, 0, 0, 0} &&
        steps[9] == Step{AvbdStage::free_subtract, 0, 0, 0} &&
        steps[10] == Step{AvbdStage::free_clear, 0, 0, 0}, "AVBD island dependency ordering wrong");
    require(steps[14] == Step{AvbdStage::impact_gaps, 1, 0, 0} &&
        steps[15] == Step{AvbdStage::impact_bodies, 1, 0, 0} &&
        steps[16] == Step{AvbdStage::impact_constraints, 1, 0, 0}, "AVBD impact ordering wrong");
    require(steps.back() == Step{AvbdStage::finish_phase, 1, 0, 0}, "AVBD finish ordering wrong");
    steps.clear();
    encode_avbd_schedule(0, 0, [&](auto... values) { steps.emplace_back(values...); });
    require(steps.size() == 14, "AVBD empty schedule bookkeeping wrong");
}
}
int main() {
    try {
        arguments_and_records();
        schedule();
        std::cout << "Portable compute arguments, relocation and AVBD schedule passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
