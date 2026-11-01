// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"
#include "resource_records.hpp"
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>

namespace parallel_mater::metal {
namespace {
thread_local ScheduleContext *active{};
std::atomic<std::uint64_t> profile_context_sequence{};
std::mutex profile_output_mutex;
struct Allocation { compute::BufferPtr buffer; ScheduleContext *owner{}; };
std::mutex allocation_mutex;
std::map<std::uintptr_t, Allocation> allocations;
struct Slice { compute::BufferPtr buffer; std::size_t offset{}; ScheduleContext *owner{}; };
Slice find(const void *pointer, std::size_t bytes = 1) {
    if (!pointer) return {};
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    std::lock_guard lock(allocation_mutex);
    auto it = allocations.upper_bound(address);
    if (it == allocations.begin()) return {};
    --it;
    const auto offset = address - it->first;
    if (offset >= it->second.buffer->size()) return {};
    if (bytes > it->second.buffer->size() - offset)
        throw std::invalid_argument("Metal allocation range exceeded");
    return {it->second.buffer, offset, it->second.owner};
}
void library_anchor() {}
}

struct ScheduleContext::ProfileState {
    struct KernelCounters {
        std::uint64_t dispatches{};
        std::uint64_t workgroups{};
    };
    using KernelMap = std::map<std::string, KernelCounters, std::less<>>;
    struct Snapshot {
        compute::CommandStatistics commands{};
        std::uint64_t readbacks{};
        std::uint64_t host_waits{};
        KernelMap kernels;
    };
    struct Phase {
        std::string name;
        Snapshot counters{};
        std::uint32_t declared_launches{};
    };
    struct Report {
        std::uint64_t context{};
        std::uint64_t schedule{};
        Snapshot counters{};
        std::uint64_t declared_launches{};
        std::vector<Phase> phases;
        std::vector<compute::SubmissionPtr> completions;
        double record_ms{};
        double submit_ms{};
        std::chrono::steady_clock::time_point submitted_at{};
    };

    explicit ProfileState(std::string output_path)
        : context(profile_context_sequence.fetch_add(
              1U, std::memory_order_relaxed) + 1U),
          output(std::move(output_path)) {}

    static compute::CommandStatistics add(
        compute::CommandStatistics left,
        const compute::CommandStatistics &right) noexcept {
        left.dispatches += right.dispatches;
        left.workgroups += right.workgroups;
        left.dependency_barriers += right.dependency_barriers;
        left.dependency_ranges += right.dependency_ranges;
        left.transfer_commands += right.transfer_commands;
        left.bytes_moved += right.bytes_moved;
        return left;
    }
    static Snapshot difference(const Snapshot &value,
                               const Snapshot &base) noexcept {
        Snapshot result{{
                    value.commands.dispatches - base.commands.dispatches,
                    value.commands.workgroups - base.commands.workgroups,
                    value.commands.dependency_barriers -
                        base.commands.dependency_barriers,
                    value.commands.dependency_ranges -
                        base.commands.dependency_ranges,
                    value.commands.transfer_commands -
                        base.commands.transfer_commands,
                    value.commands.bytes_moved - base.commands.bytes_moved,
                },
                value.readbacks - base.readbacks,
                value.host_waits - base.host_waits,
                {}};
        for (const auto &[name, counters] : value.kernels) {
            const auto found = base.kernels.find(name);
            const KernelCounters previous = found == base.kernels.end()
                ? KernelCounters{} : found->second;
            const KernelCounters delta{
                counters.dispatches - previous.dispatches,
                counters.workgroups - previous.workgroups,
            };
            if (delta.dispatches || delta.workgroups)
                result.kernels.emplace(name, delta);
        }
        return result;
    }

    std::uint64_t context{};
    std::string output;
    std::uint64_t schedule{};
    std::uint64_t readbacks{};
    std::uint64_t host_waits{};
    KernelMap kernels;
    Snapshot start{}, last{}, finish{};
    std::vector<Phase> phases;
    std::vector<compute::SubmissionPtr> completions;
    std::vector<Report> pending;
    std::chrono::steady_clock::time_point record_started{};
    double submit_ms{};
    bool active{};
    bool finished{};
};

namespace {
void write_kernel_counters(
    std::ostream &output, const ScheduleContext::ProfileState::KernelMap &kernels) {
    output << '[';
    bool first = true;
    for (const auto &[name, counters] : kernels) {
        if (!first) output << ',';
        first = false;
        output << "{\"name\":\"" << name << "\",\"dispatches\":"
               << counters.dispatches << ",\"workgroups\":"
               << counters.workgroups << '}';
    }
    output << ']';
}

void write_profile(const ScheduleContext::ProfileState &state,
                   const ScheduleContext::ProfileState::Report &report,
                   double gpu_ms, double wait_ms,
                   double submit_to_fence_ms) noexcept {
    try {
        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << std::fixed << std::setprecision(6)
             << "{\"schema\":\"parallel-mater-metal-schedule-profile/v1\""
             << ",\"context\":" << report.context
             << ",\"schedule\":" << report.schedule
             << ",\"submissions\":" << report.completions.size()
             << ",\"final\":true"
             << ",\"dispatches\":" << report.counters.commands.dispatches
             << ",\"workgroups\":" << report.counters.commands.workgroups
             << ",\"dependency_barriers\":"
             << report.counters.commands.dependency_barriers
             << ",\"dependency_ranges\":"
             << report.counters.commands.dependency_ranges
             << ",\"native_barrier_commands\":0"
             << ",\"transfer_commands\":"
             << report.counters.commands.transfer_commands
             << ",\"readbacks\":" << report.counters.readbacks
             << ",\"host_waits\":" << report.counters.host_waits
             << ",\"bytes_moved\":" << report.counters.commands.bytes_moved
             << ",\"declared_launches\":" << report.declared_launches
             << ",\"kernels\":";
        write_kernel_counters(json, report.counters.kernels);
        json
             << ",\"gpu_ms\":" << gpu_ms
             << ",\"gpu_timing_status\":\"supported\""
             << ",\"record_ms\":" << report.record_ms
             << ",\"submit_ms\":" << report.submit_ms
             << ",\"wait_ms\":" << wait_ms
             << ",\"submit_to_fence_ms\":" << submit_to_fence_ms
             << ",\"barrier_model\":\"tracked-resources-and-ordered-encoders\""
             << ",\"instrumentation\":{\"timestamp_queries\":0,"
                "\"forced_waits\":0,\"result_bytes\":0}"
             << ",\"phases\":[";
        for (std::size_t index = 0U; index < report.phases.size(); ++index) {
            if (index != 0U) json << ',';
            const auto &phase = report.phases[index];
            json << "{\"name\":\"" << phase.name
                 << "\",\"dispatches\":" << phase.counters.commands.dispatches
                 << ",\"workgroups\":" << phase.counters.commands.workgroups
                 << ",\"dependency_barriers\":"
                 << phase.counters.commands.dependency_barriers
                 << ",\"dependency_ranges\":"
                 << phase.counters.commands.dependency_ranges
                 << ",\"native_barrier_commands\":0"
                 << ",\"transfer_commands\":"
                 << phase.counters.commands.transfer_commands
                 << ",\"readbacks\":" << phase.counters.readbacks
                 << ",\"host_waits\":" << phase.counters.host_waits
                 << ",\"bytes_moved\":" << phase.counters.commands.bytes_moved
                 << ",\"declared_launches\":" << phase.declared_launches
                 << ",\"kernels\":";
            write_kernel_counters(json, phase.counters.kernels);
            json
                 << ",\"gpu_timing_status\":\"aggregate-only\"}";
        }
        json << "]}";
        std::lock_guard output_lock(profile_output_mutex);
        if (state.output.empty()) {
            std::fprintf(stderr, "MetalScheduleProfile %s\n", json.str().c_str());
            return;
        }
        std::ofstream stream(state.output, std::ios::app);
        stream.imbue(std::locale::classic());
        stream << json.str() << '\n';
        stream.flush();
        if (!stream)
            std::fprintf(stderr,
                "Parallel Mater Metal: could not write schedule profile %s\n",
                state.output.c_str());
    } catch (const std::exception &error) {
        std::fprintf(stderr,
            "Parallel Mater Metal: schedule profile write failed: %s\n",
            error.what());
    }
}
}

ScheduleContext *context(cudaStream_t stream) noexcept {
    return stream ? static_cast<ScheduleContext *>(stream) : active;
}
ScheduleContext::ScheduleContext(std::unique_ptr<compute::Device> device) : device_(std::move(device)) {
    if (!device_) throw std::runtime_error("Apple GPU unavailable (no CPU fallback)");
    if (const char *enabled = std::getenv("PM_METAL_SCHEDULE_PROFILE");
        enabled && std::strcmp(enabled, "1") == 0) {
        const char *path = std::getenv("PM_METAL_SCHEDULE_PROFILE_FILE");
        profile_ = std::make_unique<ProfileState>(path ? path : "");
    }
    Dl_info library{};
    if (!dladdr(reinterpret_cast<const void *>(&library_anchor), &library) || !library.dli_fname)
        throw std::runtime_error("cannot locate Metal driver library");
    const auto root = std::filesystem::path(library.dli_fname).parent_path() / "slang-metal";
    std::ifstream manifest(root / "manifest.txt");
    if (!manifest) throw std::runtime_error("bundled Metal shader manifest missing: " + root.string());
    std::string line;
    while (std::getline(manifest, line)) {
        auto first = line.find('|'), second = line.find('|', first + 1);
        if (first == std::string::npos || second == std::string::npos)
            throw std::runtime_error("invalid bundled Metal manifest");
        paths_.emplace(line.substr(first + 1, second - first - 1), (root / line.substr(second + 1)).string());
    }
}
ScheduleContext::~ScheduleContext() {
    try { wait(); } catch (...) {}
    std::lock_guard lock(allocation_mutex);
    std::erase_if(allocations, [this](const auto &entry) { return entry.second.owner == this; });
    if (active == this) active = nullptr;
}
void ScheduleContext::activate() noexcept { active = this; }
void *ScheduleContext::allocate(std::size_t bytes) {
    if (!bytes) throw std::invalid_argument("empty Metal allocation");
    auto buffer = device_->allocate(bytes, true);
    auto *pointer = buffer->host_address();
    if (!pointer) throw std::bad_alloc();
    std::lock_guard lock(allocation_mutex);
    allocations.emplace(reinterpret_cast<std::uintptr_t>(pointer), Allocation{buffer, this});
    slices_.emplace(pointer, CachedSlice{buffer, 0U});
    return pointer;
}
void ScheduleContext::free(void *pointer) {
    if (!pointer) return;
    wait();
    launch_cache_.clear();
    launch_cursor_ = 0U;
    std::lock_guard lock(allocation_mutex);
    const auto it = allocations.find(reinterpret_cast<std::uintptr_t>(pointer));
    if (it == allocations.end() || it->second.owner != this)
        throw std::invalid_argument("foreign Metal allocation");
    allocations.erase(it);
    slices_.clear();
}
void *ScheduleContext::native_resource(const void *pointer) const {
    const auto slice = find(pointer);
    return slice.owner == this && slice.offset == 0 ? slice.buffer->native_handle() : nullptr;
}
compute::KernelPtr ScheduleContext::kernel(const std::string &entry) {
    if (auto found = kernels_.find(entry); found != kernels_.end()) return found->second;
    const auto &path = paths_.at(entry);
    auto result = device_->load_kernel(path + ".metallib", path + ".json", entry);
    std::vector<Parameter> parameters;
    parameters.reserve(result->parameters.size());
    for (const auto &[name, field] : result->parameters) {
        parameters.push_back({name, &field});
    }
    std::sort(parameters.begin(), parameters.end(), [](const auto &a, const auto &b) {
        return a.field->offset < b.field->offset;
    });
    parameters_.emplace(result.get(), std::move(parameters));
    kernels_.emplace(entry, result);
    return result;
}
ScheduleContext::CachedSlice ScheduleContext::slice(
    const void *pointer, std::size_t bytes) {
    if (!pointer) return {};
    if (auto cached = slices_.find(pointer); cached != slices_.end()) {
        if (bytes > cached->second.buffer->size() - cached->second.offset)
            throw std::invalid_argument("Metal allocation range exceeded");
        return cached->second;
    }
    const auto found = find(pointer, bytes);
    if (!found.buffer || found.owner != this) return {};
    CachedSlice result{found.buffer, found.offset};
    slices_.emplace(pointer, result);
    return result;
}
void ScheduleContext::bind(compute::Arguments &arguments, const std::string &name,
                           const compute::Field &field, const void *pointer) {
    if (!pointer) { arguments.bind_positional(name, field, {}); return; }
    auto slice = this->slice(pointer, field.pointee_size);
    if (!slice.buffer) throw std::invalid_argument("unregistered Metal shader pointer: " + name);
    if (name == "meshes" || name == "paint_fields") {
        const auto extent = table_counts.find(pointer);
        if (extent == table_counts.end())
            throw std::invalid_argument("Metal resource table extent missing: " + name);
        const auto count = extent->second;
        if (count == 0) { arguments.bind_positional(name, field, {}); return; }
        const auto stride = name == "meshes" ? sizeof(simulation::TriangleMeshResource)
                                            : sizeof(simulation::PaintFieldResource);
        if (field.pointee_size != stride || count > (slice.buffer->size() - slice.offset) / stride)
            throw std::invalid_argument("Metal resource table layout/range mismatch: " + name);
        auto translated = device_->allocate(count * stride, true);
        auto *output = static_cast<std::byte *>(translated->host_address());
        for (std::size_t i = 0; i < count; ++i) {
            const auto host = name == "meshes"
                ? mesh_record(static_cast<const simulation::TriangleMeshResource *>(pointer)[i])
                : paint_record(static_cast<const simulation::PaintFieldResource *>(pointer)[i]);
            const auto record = relocate_record(field, host);
            const auto bytes = record.bytes();
            std::memcpy(output + i * stride, bytes.data(), bytes.size());
            for (const auto &binding : record.bindings())
                arguments.reference(name + "[]." + binding.name, binding.range.buffer,
                    binding.range.offset, binding.range.size, binding.range.access);
        }
        arguments.bind_positional(name, field, translated);
    } else {
        // Unknown pointer-bearing arrays need an explicit extent and layout.
        for (const auto &[member, child] : field.fields)
            if (compute::contains_pointers(child))
                throw std::invalid_argument("unrelocated Metal resource table: " + name + "." + member);
        arguments.bind_positional(name, field, slice.buffer, slice.offset);
    }
}
void ScheduleContext::meshes(const void *table, const void *parameters, std::uint32_t count) {
    std::lock_guard lock(mutex);
    const auto *bodies = static_cast<const simulation::BodyParameters *>(parameters);
    std::size_t size = 1;
    for (std::uint32_t i = 0; i < count; ++i)
        size = std::max(size, std::size_t(bodies[i].mesh.index) + 1);
    table_counts[table] = size;
    table_strides[table] = sizeof(simulation::TriangleMeshResource);
}
void ScheduleContext::record(compute::Arguments &arguments, const std::string &name,
                             const compute::Field &field,
                             const RelocatedRecord &host) {
    arguments.record_positional(name, field, relocate_record(field, host));
}
void ScheduleContext::paint_fields(const void *table, std::uint32_t count, const void *meshes) {
    std::lock_guard lock(mutex);
    table_counts[table] = count;
    table_strides[table] = sizeof(simulation::PaintFieldResource);
    if (meshes)
        table_strides[meshes] = sizeof(simulation::TriangleMeshResource);
    // Painting may reference a UV-bearing mesh distinct from collision geometry.
    // Include these resources in the relocated table, not only body mesh IDs.
    const auto *fields = static_cast<const simulation::PaintFieldResource *>(table);
    for (std::uint32_t i = 0; i < count; ++i)
        if (fields[i].alive && fields[i].options.mesh.generation != 0)
            table_counts[meshes] = std::max(table_counts[meshes],
                std::size_t(fields[i].options.mesh.index) + 1);
}
compute::Record ScheduleContext::relocate_record(compute::Field layout, const RelocatedRecord &host) {
    if (layout.pointer) {
        layout.size = layout.pointee_size;
        layout.alignment = layout.pointee_alignment;
        layout.pointer = false;
    }
    const std::string name = "resource";
    if (host.bytes.size() != layout.size)
        throw std::invalid_argument("Metal host record size mismatch: " + name);
    for (const auto &member : host.members) {
        const compute::Field *field = &layout;
        std::string_view path(member.path);
        std::size_t offset = 0;
        while (!path.empty()) {
            const auto dot = path.find('.');
            field = &field->fields.at(std::string(path.substr(0, dot)));
            offset += field->offset;
            if (dot == std::string_view::npos) break;
            path.remove_prefix(dot + 1);
        }
        if (offset != member.offset || field->size != member.size)
            throw std::invalid_argument("Metal host member layout mismatch: " + name + "." + member.path);
    }
    compute::Record translated(layout, host.bytes);
    const std::function<void(const compute::Field &, const std::string &, std::size_t)> relocate =
        [&](const compute::Field &field, const std::string &path, std::size_t offset) {
            if (field.pointer) {
                if (offset > host.bytes.size() || sizeof(void *) > host.bytes.size() - offset)
                    throw std::invalid_argument("Metal nested pointer outside host record");
                const void *pointer{};
                std::memcpy(&pointer, host.bytes.data() + offset, sizeof(pointer));
                const auto resource = slice(pointer, field.pointee_size);
                if (pointer && !resource.buffer)
                    throw std::invalid_argument("unregistered Metal record pointer: " + name + "." + path);
                translated.bind(path, resource.buffer, resource.offset);
            } else if (field.element) {
                for (std::size_t i = 0; i < field.element_count; ++i)
                    relocate(*field.element, path + "[" + std::to_string(i) + "]",
                             offset + i * field.element_stride);
            } else {
                for (const auto &[child_name, child] : field.fields)
                    relocate(child, path.empty() ? child_name : path + "." + child_name, offset + child.offset);
            }
        };
    relocate(layout, "", 0);
    return translated;
}
void ScheduleContext::dispatch(const compute::Arguments &arguments, dim3 groups) {
    if (!command_) command_ = device_->commands();
    command_->dispatch(arguments, {groups.x, groups.y, groups.z});
    if (profile_) {
        auto &kernel = profile_->kernels[arguments.kernel()->name];
        ++kernel.dispatches;
        kernel.workgroups += std::uint64_t{groups.x} * groups.y * groups.z;
    }
    ++dispatch_count;
}
compute::CommandStatistics ScheduleContext::command_statistics() const noexcept {
    return ProfileState::add(submitted_statistics_,
        command_ ? command_->statistics() : compute::CommandStatistics{});
}
void ScheduleContext::collect_profiles(bool wait_for_completion) {
    if (!profile_) return;
    auto &reports = profile_->pending;
    for (auto iterator = reports.begin(); iterator != reports.end();) {
        bool ready = true;
        const auto wait_started = std::chrono::steady_clock::now();
        for (const auto &completion : iterator->completions) {
            if (wait_for_completion) completion->wait();
            else if (!completion->ready()) { ready = false; break; }
        }
        if (!ready) { ++iterator; continue; }
        const double wait_ms = wait_for_completion
            ? std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - wait_started).count()
            : 0.0;
        double gpu_ms{};
        for (const auto &completion : iterator->completions)
            gpu_ms += completion->elapsed_seconds() * 1000.0;
        const double submit_to_fence_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                iterator->submitted_at).count();
        write_profile(*profile_, *iterator, gpu_ms, wait_ms,
                      submit_to_fence_ms);
        iterator = reports.erase(iterator);
    }
}
compute::SubmissionPtr ScheduleContext::submit() {
    launch_cursor_ = 0U;
    collect_profiles(false);
    std::erase_if(pending_, [](const auto &completion) { return completion->ready(); });
    if (!command_) command_ = device_->commands();
    // Allocate the scheduler's tracking slot before GPU acceptance too. A
    // post-submit vector growth failure must never lose work from wait().
    pending_.emplace_back();
    try {
        const auto statistics = command_->statistics();
        const auto submit_started = std::chrono::steady_clock::now();
        auto completion = command_->submit();
        const double submit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - submit_started).count();
        pending_.back() = completion;
        submitted_statistics_ = ProfileState::add(
            submitted_statistics_, statistics);
        command_.reset();
        if (profile_ && (profile_->active || profile_->finished)) {
            profile_->completions.push_back(completion);
            profile_->submit_ms += submit_ms;
            if (profile_->finished) {
                ProfileState::Report report;
                report.context = profile_->context;
                report.schedule = profile_->schedule;
                report.counters = ProfileState::difference(
                    profile_->finish, profile_->start);
                for (const auto &phase : profile_->phases)
                    report.declared_launches += phase.declared_launches;
                report.phases = std::move(profile_->phases);
                report.completions = std::move(profile_->completions);
                report.record_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    profile_->record_started).count();
                report.submit_ms = profile_->submit_ms;
                report.submitted_at = std::chrono::steady_clock::now();
                profile_->pending.push_back(std::move(report));
                profile_->submit_ms = 0.0;
                profile_->finished = false;
            }
        }
        return completion;
    } catch (...) {
        pending_.pop_back();
        throw;
    }
}
void ScheduleContext::wait() {
    std::lock_guard lock(mutex);
    if (command_) (void)submit();
    collect_profiles(true);
    for (const auto &completion : pending_) completion->wait();
    pending_.clear();
}
void ScheduleContext::copy(void *destination, const void *source, std::size_t bytes) {
    if (!bytes) return;
    if (!source || !destination) throw std::invalid_argument("null Metal copy pointer");
    const auto dst = find(destination, bytes), src = find(source, bytes);
    if ((dst.buffer && dst.owner != this) || (src.buffer && src.owner != this))
        throw std::invalid_argument("cross-device Metal copy");
    const bool same_range = dst.buffer && src.buffer &&
        dst.buffer == src.buffer && dst.offset == src.offset;
    const bool overlapping = dst.buffer && src.buffer &&
        dst.buffer == src.buffer &&
        dst.offset < src.offset + bytes && src.offset < dst.offset + bytes;
    if (same_range) {
        return;
    } else if (dst.buffer && src.buffer && !overlapping) {
        if (!command_) command_ = device_->commands();
        command_->copy(dst.buffer, dst.offset, src.buffer, src.offset, bytes);
    } else if (dst.buffer && !src.buffer) {
        if (!command_) command_ = device_->commands();
        command_->upload(dst.buffer, dst.offset, {static_cast<const std::byte *>(source), bytes});
    } else {
        // Preview readback boundary, including overlapping shared-arena copies.
        if (profile_) {
            ++profile_->host_waits;
            if (src.buffer) ++profile_->readbacks;
        }
        wait();
        std::memmove(destination, source, bytes);
    }
}
void ScheduleContext::clear(void *destination, int value, std::size_t bytes) {
    if (!bytes) return;
    auto dst = find(destination, bytes);
    if (!dst.buffer || dst.owner != this) throw std::invalid_argument("unregistered Metal clear pointer");
    if (!command_) command_ = device_->commands();
    command_->fill(dst.buffer, dst.offset, bytes, static_cast<std::uint8_t>(value));
}
bool ScheduleContext::schedule_profile_enabled() const noexcept {
    return profile_ != nullptr;
}
cudaError_t ScheduleContext::begin_schedule_profile() noexcept {
    if (!profile_) return cudaSuccess;
    return checked([&] {
        std::lock_guard lock(mutex);
        collect_profiles(false);
        if (profile_->active || profile_->finished)
            throw std::logic_error("Metal schedule profile already active");
        profile_->active = true;
        ++profile_->schedule;
        profile_->phases.clear();
        profile_->completions.clear();
        profile_->submit_ms = 0.0;
        profile_->start = profile_->last = {command_statistics(),
            profile_->readbacks, profile_->host_waits, profile_->kernels};
        profile_->record_started = std::chrono::steady_clock::now();
    }, false);
}
cudaError_t ScheduleContext::record_schedule_profile_stage(
    std::string_view name, std::uint32_t launches) noexcept {
    if (!profile_) return cudaSuccess;
    return checked([&] {
        std::lock_guard lock(mutex);
        if (!profile_->active)
            throw std::logic_error("Metal schedule profile is not active");
        if (name.empty() || name.find_first_of("\"\\\r\n") !=
                std::string_view::npos)
            throw std::invalid_argument("invalid Metal profile stage name");
        const ProfileState::Snapshot current{command_statistics(),
            profile_->readbacks, profile_->host_waits, profile_->kernels};
        profile_->phases.push_back({std::string(name),
            ProfileState::difference(current, profile_->last), launches});
        profile_->last = current;
    }, false);
}
cudaError_t ScheduleContext::end_schedule_profile() noexcept {
    if (!profile_) return cudaSuccess;
    return checked([&] {
        std::lock_guard lock(mutex);
        if (!profile_->active)
            throw std::logic_error("Metal schedule profile is not active");
        const ProfileState::Snapshot current{command_statistics(),
            profile_->readbacks, profile_->host_waits, profile_->kernels};
        const auto tail = ProfileState::difference(current, profile_->last);
        if (tail.commands.dispatches || tail.commands.workgroups ||
            tail.commands.dependency_barriers ||
            tail.commands.transfer_commands || tail.readbacks ||
            tail.host_waits)
            profile_->phases.push_back({"schedule_tail", tail, 0U});
        profile_->finish = current;
        profile_->active = false;
        profile_->finished = true;
    }, false);
}
cudaError_t ScheduleContext::remember(cudaError_t value) noexcept { error_ = value; return value; }
cudaError_t ScheduleContext::error(bool take) noexcept {
    std::lock_guard lock(mutex);
    const auto result = error_;
    if (take) error_ = cudaSuccess;
    return result;
}
} // namespace parallel_mater::metal

using parallel_mater::metal::context;
cudaError_t pmMetalAllocate(void **output, std::size_t size) noexcept {
    if (!output || !size || !context()) return cudaErrorInvalidValue;
    *output = nullptr;
    return context()->checked([&] { *output = context()->allocate(size); });
}
cudaError_t pmMetalFree(void *pointer) noexcept {
    if (!pointer) return cudaSuccess;
    try {
        auto slice = parallel_mater::metal::find(pointer);
        if (!slice.owner || slice.offset) return cudaErrorInvalidValue;
        return slice.owner->checked([&] { slice.owner->free(pointer); });
    } catch (...) { return cudaErrorInvalidValue; }
}
cudaError_t pmMetalMemcpyAsync(void *dst, const void *src, std::size_t size, cudaMemcpyKind, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] { execution->copy(dst, src, size); if (!stream) execution->wait(); });
}
cudaError_t pmMetalMemsetAsync(void *dst, int value, std::size_t size, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] { execution->clear(dst, value, size); if (!stream) execution->wait(); });
}
cudaError_t pmMetalStreamSynchronize(cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    return execution ? execution->checked([&] { execution->wait(); }) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmMetalRecordEvent(cudaEvent_t event, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!event || !execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] {
        event->submission = execution->submit(); event->stream = execution;
        event->recorded = std::chrono::steady_clock::now();
    });
}
cudaError_t pmMetalWaitEvent(cudaEvent_t event) noexcept {
    if (!event || !event->submission) return cudaErrorInvalidResourceHandle;
    try {
        event->submission->wait();
        return cudaSuccess;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Metal preview: %s\n", error.what());
        return cudaErrorLaunchFailure;
    } catch (...) {
        std::fprintf(stderr, "Metal preview: unknown event wait failure\n");
        return cudaErrorLaunchFailure;
    }
}
cudaError_t pmMetalQueryEvent(cudaEvent_t event) noexcept {
    if (!event || !event->submission) return cudaErrorInvalidResourceHandle;
    try {
        return event->submission->ready() ? cudaSuccess : cudaErrorNotReady;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Metal preview: %s\n", error.what());
        return cudaErrorLaunchFailure;
    } catch (...) {
        std::fprintf(stderr, "Metal preview: unknown event query failure\n");
        return cudaErrorLaunchFailure;
    }
}
cudaError_t pmMetalElapsedTime(float *milliseconds, cudaEvent_t begin,
                              cudaEvent_t end) noexcept {
    // The device epoch is deliberately a non-timing synchronization event;
    // timestamp slots are timing events measured from that same GPU epoch.
    if (!milliseconds || !begin || !end || !end->timing ||
        !begin->submission || !end->submission ||
        begin->stream != end->stream)
        return cudaErrorInvalidValue;
    try {
        const double seconds = end->submission->completion_seconds() -
            begin->submission->completion_seconds();
        if (seconds < 0.0) return cudaErrorInvalidValue;
        *milliseconds = static_cast<float>(seconds * 1000.0);
        return cudaSuccess;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Metal preview: %s\n", error.what());
        return cudaErrorLaunchFailure;
    } catch (...) {
        std::fprintf(stderr, "Metal preview: unknown event timing failure\n");
        return cudaErrorLaunchFailure;
    }
}
cudaError_t pmMetalPeekError(cudaStream_t stream, bool take) noexcept {
    auto *execution = context(stream);
    return execution ? execution->error(take) : cudaErrorInvalidResourceHandle;
}
