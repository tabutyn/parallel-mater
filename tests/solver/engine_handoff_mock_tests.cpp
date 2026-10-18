// SPDX-License-Identifier: MIT
#include <device/compute.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace compute = parallel_mater::device::compute;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function &&function) {
    try { function(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid engine import/handoff accepted");
}

struct Counters {
    unsigned native_devices_destroyed{};
    unsigned native_queues_destroyed{};
    unsigned native_buffers_destroyed{};
    unsigned wrappers_created{};
    unsigned wrappers_destroyed{};
};
struct NativeDevice {
    std::shared_ptr<Counters> counters;
    ~NativeDevice() { ++counters->native_devices_destroyed; }
};
struct NativeQueue {
    NativeDevice *device{};
    std::shared_ptr<Counters> counters;
    ~NativeQueue() { ++counters->native_queues_destroyed; }
};
struct NativeBuffer {
    NativeDevice *device{};
    std::size_t bytes{};
    std::shared_ptr<Counters> counters;
    ~NativeBuffer() { ++counters->native_buffers_destroyed; }
};

class Engine final {
public:
    explicit Engine(std::shared_ptr<Counters> counters) : counters_(std::move(counters)) {}
    NativeDevice *device() {
        auto value = std::make_unique<NativeDevice>();
        value->counters = counters_;
        devices_.push_back(std::move(value));
        return devices_.back().get();
    }
    NativeQueue *queue(NativeDevice *device) {
        auto value = std::make_unique<NativeQueue>();
        value->device = device;
        value->counters = counters_;
        queues_.push_back(std::move(value));
        return queues_.back().get();
    }
    NativeBuffer *buffer(NativeDevice *device, std::size_t bytes) {
        auto value = std::make_unique<NativeBuffer>();
        value->device = device;
        value->bytes = bytes;
        value->counters = counters_;
        buffers_.push_back(std::move(value));
        return buffers_.back().get();
    }
    bool owns(const NativeDevice *value) const { return contains(devices_, value); }
    bool owns(const NativeQueue *value) const { return contains(queues_, value); }
    bool owns(const NativeBuffer *value) const { return contains(buffers_, value); }
    void clear() {
        buffers_.clear();
        queues_.clear();
        devices_.clear();
    }
private:
    template<class T> static bool contains(const std::vector<std::unique_ptr<T>> &values,
                                           const T *value) {
        for (const auto &candidate : values) if (candidate.get() == value) return true;
        return false;
    }
    std::shared_ptr<Counters> counters_;
    std::vector<std::unique_ptr<NativeDevice>> devices_;
    std::vector<std::unique_ptr<NativeQueue>> queues_;
    std::vector<std::unique_ptr<NativeBuffer>> buffers_;
};

compute::NativeObject borrow(compute::NativeObjectKind kind, const void *value,
                             compute::NativeApi api = compute::NativeApi::mock) {
    return compute::NativeObject::borrowed(
        api, kind, reinterpret_cast<std::uintptr_t>(value));
}

enum class QueueRole { physics, renderer };
struct Job {
    std::uint64_t id{};
    QueueRole role{};
    std::vector<Job *> prior;
    bool done{};
};
struct JobSnapshot {
    std::uint64_t id{};
    QueueRole role{};
    std::vector<std::uint64_t> prior;
};
struct GpuState {
    std::uint64_t next_id{1};
    unsigned accepted{};
    unsigned encoded_waits{};
    unsigned native_waits{};
    unsigned signals{};
    unsigned tokens_destroyed{};
    std::unordered_set<Job *> live;
    std::vector<JobSnapshot> submissions;
    std::vector<std::uint64_t> completion_order;

    void complete(Job *job) {
        require(job && live.contains(job), "completion referenced a dead native job");
        if (job->done) return;
        for (auto *dependency : job->prior) complete(dependency);
        job->done = true;
        ++signals;
        completion_order.push_back(job->id);
    }
};

class Completion final : public compute::Submission {
public:
    Completion(compute::QueueIdentity identity, std::shared_ptr<GpuState> state,
               std::shared_ptr<Job> job)
        : identity_(identity), state_(std::move(state)), job_(std::move(job)) {}
    ~Completion() override {
        state_->live.erase(job_.get());
        ++state_->tokens_destroyed;
    }
    bool ready() const override { return job_->done; }
    void wait() const override {
        ++state_->native_waits;
        state_->complete(job_.get());
    }
    double elapsed_seconds() const override {
        if (!ready()) throw std::logic_error("mock GPU timestamps requested before completion");
        return 0;
    }
    void *native_handle() const noexcept override { return job_.get(); }
    compute::QueueIdentity queue_identity() const noexcept override { return identity_; }
private:
    compute::QueueIdentity identity_;
    std::shared_ptr<GpuState> state_;
    std::shared_ptr<Job> job_;
};

class ImportedBuffer final : public compute::Buffer {
public:
    ImportedBuffer(compute::NativeObject native, NativeBuffer *buffer,
                   std::shared_ptr<Counters> counters)
        : native_(std::move(native)), buffer_(buffer), counters_(std::move(counters)) {
        ++counters_->wrappers_created;
    }
    ~ImportedBuffer() override { ++counters_->wrappers_destroyed; }
    compute::ResourceIdentity identity() const noexcept override {
        return {reinterpret_cast<std::uintptr_t>(buffer_->device), native_.handle()};
    }
    std::size_t size() const noexcept override { return buffer_->bytes; }
    void *host_address() const noexcept override { return nullptr; }
    std::uint64_t device_address() const noexcept override { return native_.handle(); }
    void *native_handle() const noexcept override {
        return reinterpret_cast<void *>(native_.handle());
    }
private:
    compute::NativeObject native_;
    NativeBuffer *buffer_;
    std::shared_ptr<Counters> counters_;
};

class Encoder final : public compute::CommandEncoder {
public:
    Encoder(compute::QueueIdentity identity, QueueRole role, std::shared_ptr<GpuState> state)
        : identity_(identity), role_(role), state_(std::move(state)) {}
    compute::QueueIdentity queue_identity() const noexcept override { return identity_; }
    void wait(std::span<const compute::SubmissionPtr> prior) override {
        for (const auto &completion : prior) {
            auto *job = static_cast<Job *>(completion->native_handle());
            if (!job || !state_->live.contains(job))
                throw std::invalid_argument("foreign mock completion");
            prior_.push_back(job);
            ++state_->encoded_waits;
        }
    }
    void barrier(std::span<const compute::ResourceDependency>) override {}
    void dispatch(const compute::Arguments &, compute::Extent) override {}
    void copy(const compute::BufferPtr &destination, std::size_t,
              const compute::BufferPtr &source, std::size_t, std::size_t) override {
        imported(destination);
        imported(source);
        ++operations_;
    }
    void fill(const compute::BufferPtr &destination, std::size_t, std::size_t,
              std::uint8_t) override {
        imported(destination);
        ++operations_;
    }
    compute::BufferPtr upload(const compute::BufferPtr &, std::size_t,
                              std::span<const std::byte>) override {
        throw std::logic_error("mock engine adapter does not stage host uploads");
    }
    compute::SubmissionPtr submit() override {
        require(operations_ == 1, "mock frame must contain one physics/render operation");
        auto job = std::make_shared<Job>();
        job->id = state_->next_id++;
        job->role = role_;
        job->prior = prior_;
        state_->live.insert(job.get());
        JobSnapshot snapshot{job->id, role_, {}};
        for (const auto *dependency : prior_) snapshot.prior.push_back(dependency->id);
        state_->submissions.push_back(std::move(snapshot));
        ++state_->accepted;
        return std::make_shared<Completion>(identity_, state_, std::move(job));
    }
private:
    static void imported(const compute::BufferPtr &value) {
        if (!std::dynamic_pointer_cast<ImportedBuffer>(value))
            throw std::invalid_argument("foreign mock buffer");
    }
    compute::QueueIdentity identity_;
    QueueRole role_;
    std::shared_ptr<GpuState> state_;
    std::vector<Job *> prior_;
    unsigned operations_{};
};

class MockAdapter final {
public:
    MockAdapter(Engine &engine, compute::NativeObject device,
                compute::NativeObject physics, compute::NativeObject renderer,
                std::shared_ptr<Counters> counters, std::shared_ptr<GpuState> gpu)
        : engine_(engine), counters_(std::move(counters)), gpu_(std::move(gpu)),
          device_(std::move(device)), physics_(std::move(physics)),
          renderer_(std::move(renderer)) {
        require_native(device_, compute::NativeObjectKind::device);
        require_native(physics_, compute::NativeObjectKind::queue);
        require_native(renderer_, compute::NativeObjectKind::queue);
        native_device_ = reinterpret_cast<NativeDevice *>(device_.handle());
        auto *physics_queue = reinterpret_cast<NativeQueue *>(physics_.handle());
        auto *renderer_queue = reinterpret_cast<NativeQueue *>(renderer_.handle());
        if (!engine_.owns(native_device_) || !engine_.owns(physics_queue) ||
            !engine_.owns(renderer_queue) || physics_queue->device != native_device_ ||
            renderer_queue->device != native_device_)
            throw std::invalid_argument("foreign mock device or queue");
    }

    compute::BufferPtr import_buffer(compute::NativeObject native) {
        require_native(native, compute::NativeObjectKind::buffer);
        auto *buffer = reinterpret_cast<NativeBuffer *>(native.handle());
        if (!engine_.owns(buffer) || buffer->device != native_device_ || buffer->bytes == 0)
            throw std::invalid_argument("foreign or empty mock buffer");
        auto wrapper = std::make_shared<ImportedBuffer>(
            std::move(native), buffer, counters_);
        imports_.push_back(wrapper);
        return wrapper;
    }

    std::unique_ptr<compute::CommandList> commands(
        QueueRole role, std::span<const compute::SubmissionPtr> prior = {}) {
        const auto queue = role == QueueRole::physics ? physics_.handle() : renderer_.handle();
        compute::QueueIdentity identity{
            compute::NativeApi::mock, device_.handle(), queue};
        return submissions_.commands(
            std::make_unique<Encoder>(identity, role, gpu_), prior);
    }
private:
    static void require_native(const compute::NativeObject &object,
                               compute::NativeObjectKind kind) {
        if (!object || object.api() != compute::NativeApi::mock || object.kind() != kind ||
            object.ownership() != compute::NativeOwnership::borrowed)
            throw std::invalid_argument("incompatible mock native object");
    }

    Engine &engine_;
    std::shared_ptr<Counters> counters_;
    std::shared_ptr<GpuState> gpu_;
    NativeDevice *native_device_{};
    compute::NativeObject device_;
    compute::NativeObject physics_;
    compute::NativeObject renderer_;
    std::vector<compute::BufferPtr> imports_;
    // Last member: drain accepted work before imported wrappers/native views.
    compute::SubmissionQueue submissions_;
};

void engine_handoff() {
    auto counters = std::make_shared<Counters>();
    auto gpu = std::make_shared<GpuState>();
    Engine engine(counters);
    auto *device = engine.device();
    auto *physics_queue = engine.queue(device);
    auto *renderer_queue = engine.queue(device);
    auto *scene_native = engine.buffer(device, 64);
    auto *output_native = engine.buffer(device, 64);
    auto *foreign_device = engine.device();
    auto *foreign_queue = engine.queue(foreign_device);
    auto *foreign_buffer = engine.buffer(foreign_device, 64);

    rejects([&] {
        MockAdapter invalid(engine, borrow(compute::NativeObjectKind::device, device),
            borrow(compute::NativeObjectKind::queue, physics_queue),
            borrow(compute::NativeObjectKind::queue, foreign_queue), counters, gpu);
    });
    require(counters->native_devices_destroyed == 0 &&
            counters->native_queues_destroyed == 0,
            "failed adapter import destroyed borrowed engine objects");

    std::vector<std::weak_ptr<compute::Submission>> tokens;
    std::unique_ptr<MockAdapter> adapter = std::make_unique<MockAdapter>(
        engine, borrow(compute::NativeObjectKind::device, device),
        borrow(compute::NativeObjectKind::queue, physics_queue),
        borrow(compute::NativeObjectKind::queue, renderer_queue), counters, gpu);
    rejects([&] { (void)adapter->import_buffer(
        borrow(compute::NativeObjectKind::buffer, scene_native, compute::NativeApi::vulkan)); });
    rejects([&] { (void)adapter->import_buffer(
        borrow(compute::NativeObjectKind::queue, scene_native)); });
    rejects([&] { (void)adapter->import_buffer(
        borrow(compute::NativeObjectKind::buffer, foreign_buffer)); });
    require(counters->wrappers_created == 0 && gpu->accepted == 0,
            "failed import created a wrapper or submitted work");

    auto scene = adapter->import_buffer(borrow(compute::NativeObjectKind::buffer, scene_native));
    auto output = adapter->import_buffer(borrow(compute::NativeObjectKind::buffer, output_native));
    compute::SubmissionPtr prior;
    for (std::uint8_t frame = 0; frame < 3; ++frame) {
        const std::array<compute::SubmissionPtr, 1> prior_frame{prior};
        auto physics = prior
            ? adapter->commands(QueueRole::physics, prior_frame)
            : adapter->commands(QueueRole::physics);
        physics->fill(scene, 0, scene->size(), frame);
        auto physics_done = physics->submit();
        tokens.push_back(physics_done);

        const std::array<compute::SubmissionPtr, 1> physics_prior{physics_done};
        auto renderer = adapter->commands(QueueRole::renderer, physics_prior);
        renderer->copy(output, 0, scene, 0, scene->size());
        prior = renderer->submit();
        tokens.push_back(prior);
        physics_done.reset();
    }
    prior.reset();

    require(gpu->accepted == 6 && gpu->signals == 0 && gpu->native_waits == 0,
            "simultaneous frames completed or waited on the CPU");
    require(gpu->encoded_waits == 5 && gpu->submissions.size() == 6,
            "physics/renderer dependency count mismatch");
    for (std::size_t index = 0; index < gpu->submissions.size(); ++index) {
        const auto expected_role = index % 2 == 0 ? QueueRole::physics : QueueRole::renderer;
        require(gpu->submissions[index].role == expected_role,
                "physics/renderer submission order mismatch");
        require(gpu->submissions[index].prior.size() == (index == 0 ? 0U : 1U),
                "frame handoff dependency count mismatch");
        if (index != 0)
            require(gpu->submissions[index].prior[0] == gpu->submissions[index - 1].id,
                    "frame handoff did not use the immediately prior completion");
        require(!tokens[index].expired(),
                "caller-token release broke transitive in-flight ownership");
    }

    scene.reset();
    output.reset();
    adapter.reset();
    require(gpu->signals == 6 && gpu->native_waits == 6 &&
            gpu->tokens_destroyed == 6 && gpu->live.empty(),
            "adapter teardown did not complete each accepted job exactly once");
    require(gpu->completion_order ==
                std::vector<std::uint64_t>({1, 2, 3, 4, 5, 6}),
            "native dependency completion order mismatch");
    for (const auto &token : tokens)
        require(token.expired(), "teardown retained a dropped caller token");
    require(counters->wrappers_created == 2 && counters->wrappers_destroyed == 2,
            "imported wrappers were not destroyed exactly once");
    require(counters->native_devices_destroyed == 0 &&
            counters->native_queues_destroyed == 0 &&
            counters->native_buffers_destroyed == 0,
            "adapter teardown destroyed borrowed engine objects");

    engine.clear();
    require(counters->native_devices_destroyed == 2 &&
            counters->native_queues_destroyed == 3 &&
            counters->native_buffers_destroyed == 3,
            "engine did not retain sole native-object ownership");
}
} // namespace

int main() {
    try {
        engine_handoff();
        std::cout << "Engine mock physics/render handoff and teardown passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
