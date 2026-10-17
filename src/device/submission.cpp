// SPDX-License-Identifier: MIT
#include "submission.hpp"

#include <new>
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace parallel_mater::device {

PmResult SubmissionCompletion::wait() noexcept {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [this] { return status() != PM_NOT_READY; });
    return status();
}

void SubmissionCompletion::finish(PmResult result) noexcept {
    {
        std::lock_guard lock(mutex_);
        result_.store(result, std::memory_order_release);
    }
    changed_.notify_all();
}

SubmissionQueue::SubmissionQueue() : worker_([this] { run(); }) {}

SubmissionQueue::~SubmissionQueue() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
    worker_.join();
}

PmResult SubmissionQueue::enqueue(
    std::function<PmResult()> work,
    std::function<bool()> ready,
    const std::shared_ptr<SubmissionCompletion> &completion,
    const std::function<void()> &accepted) noexcept {
    try {
        std::lock_guard lock(mutex_);
        if (stopping_ || !completion || completion->status() != PM_NOT_READY)
            return PM_ERROR_INVALID_STATE;
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
        if (std::getenv("PM_TEST_FAIL_SUBMISSION") != nullptr)
            return PM_ERROR_OUT_OF_HOST_MEMORY;
#endif
        work_.push_back({std::move(work), std::move(ready), completion});
        // All allocations and validation precede this no-fail publication.
        // The worker cannot observe the work until its handles are pending.
        accepted();
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    changed_.notify_one();
    return PM_SUCCESS;
}

PmResult SubmissionQueue::wait_idle() noexcept {
    std::unique_lock lock(mutex_);
    idle_.wait(lock, [this] { return work_.empty() && !active_; });
    const auto result = first_error_;
    first_error_ = PM_SUCCESS;
    return result;
}

void SubmissionQueue::run() noexcept {
    for (;;) {
        Work work;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [this] { return stopping_ || !work_.empty(); });
            if (work_.empty()) return;
            // An unresolved timeline wait must not block submission of its
            // producer. Run the first ready job; explicit dependencies order
            // access to shared simulations and output storage.
            const auto ready = std::find_if(work_.begin(), work_.end(),
                [](const Work &item) { return item.ready(); });
            if (ready == work_.end()) {
                changed_.wait_for(lock, std::chrono::milliseconds(1));
                continue;
            }
            work = std::move(*ready);
            work_.erase(ready);
            active_ = true;
        }
        PmResult result;
        try {
            result = work.execute();
        } catch (const std::bad_alloc &) {
            result = PM_ERROR_OUT_OF_HOST_MEMORY;
        } catch (...) {
            result = PM_ERROR_INVALID_STATE;
        }
        if (result > PM_SUCCESS) result = PM_ERROR_INVALID_STATE;
        // Release captures before advertising completion and safe reuse.
        work.execute = {};
        work.ready = {};
        work.completion->finish(result);
        {
            std::lock_guard lock(mutex_);
            if (first_error_ == PM_SUCCESS && result != PM_SUCCESS)
                first_error_ = result;
            active_ = false;
        }
        idle_.notify_all();
    }
}

} // namespace parallel_mater::device
