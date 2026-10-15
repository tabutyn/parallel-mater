// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace parallel_mater::device {

// A host submission completes only after its device work and deferred checks.
// Unlike an unrecorded native event, a newly created completion is never ready.
class SubmissionCompletion {
  public:
    [[nodiscard]] PmResult status() const noexcept {
        return result_.load(std::memory_order_acquire);
    }
    [[nodiscard]] PmResult wait() noexcept;

  private:
    void finish(PmResult result) noexcept;
    std::atomic<PmResult> result_{PM_NOT_READY};
    std::mutex mutex_;
    std::condition_variable changed_;
    friend class SubmissionQueue;
};

// One ordered host executor per public queue. The task owns its recorded
// commands and must drain native work before returning, including on failure.
class SubmissionQueue {
  public:
    SubmissionQueue();
    ~SubmissionQueue();
    SubmissionQueue(const SubmissionQueue &) = delete;
    SubmissionQueue &operator=(const SubmissionQueue &) = delete;

    [[nodiscard]] PmResult enqueue(
        std::function<PmResult()> work,
        std::function<bool()> ready,
        const std::shared_ptr<SubmissionCompletion> &completion,
        const std::function<void()> &accepted) noexcept;
    [[nodiscard]] PmResult wait_idle() noexcept;

  private:
    struct Work {
        std::function<PmResult()> execute;
        std::function<bool()> ready;
        std::shared_ptr<SubmissionCompletion> completion;
    };
    void run() noexcept;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::condition_variable idle_;
    std::deque<Work> work_;
    bool active_{};
    bool stopping_{};
    PmResult first_error_{PM_SUCCESS};
    std::thread worker_;
};

} // namespace parallel_mater::device
