// SPDX-License-Identifier: MIT
#include "device/submission.hpp"
#include "simulation/reference_schedule.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

int main() {
    using namespace parallel_mater::device;
    SubmissionQueue queue;
    const auto ready = [] { return true; };
    const auto accepted = [] {};
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::vector<int> order;
    auto first = std::make_shared<SubmissionCompletion>();
    auto second = std::make_shared<SubmissionCompletion>();
    auto payload = std::make_shared<int>(1);
    const std::weak_ptr<int> retained = payload;
    CHECK(queue.enqueue([&, payload, gate] {
        gate.wait();
        order.push_back(*payload);
        return PM_SUCCESS;
    }, ready, first, accepted) == PM_SUCCESS);
    payload.reset();
    CHECK(queue.enqueue([&] {
        order.push_back(2);
        return PM_SUCCESS;
    }, ready, second, accepted) == PM_SUCCESS);
    CHECK(!retained.expired());
    CHECK(first->status() == PM_NOT_READY);
    CHECK(second->status() == PM_NOT_READY);
    release.set_value();
    CHECK(first->wait() == PM_SUCCESS);
    CHECK(retained.expired());
    CHECK(second->wait() == PM_SUCCESS);
    CHECK((order == std::vector<int>{1, 2}));

    auto failed = std::make_shared<SubmissionCompletion>();
    CHECK(queue.enqueue([] { return PM_ERROR_CAPACITY_EXCEEDED; }, ready,
                        failed, accepted) == PM_SUCCESS);
    CHECK(failed->wait() == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(queue.wait_idle() == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(queue.wait_idle() == PM_SUCCESS);
    CHECK(failed->status() == PM_ERROR_CAPACITY_EXCEEDED);

    // A wait submitted before its producer must release the worker to run
    // that producer, while keeping the consumer's storage pending.
    auto consumer = std::make_shared<SubmissionCompletion>();
    auto producer = std::make_shared<SubmissionCompletion>();
    CHECK(queue.enqueue([&] {
        order.push_back(4);
        return PM_SUCCESS;
    }, [producer] { return producer->status() != PM_NOT_READY; }, consumer,
        accepted) == PM_SUCCESS);
    CHECK(consumer->status() == PM_NOT_READY);
    CHECK(queue.enqueue([&] {
        order.push_back(3);
        return PM_SUCCESS;
    }, ready, producer, accepted) == PM_SUCCESS);
    CHECK(consumer->wait() == PM_SUCCESS);
    CHECK((order == std::vector<int>{1, 2, 3, 4}));

    auto exceptional = std::make_shared<SubmissionCompletion>();
    CHECK(queue.enqueue([]() -> PmResult { throw std::bad_alloc{}; }, ready,
                        exceptional, accepted) == PM_SUCCESS);
    CHECK(exceptional->wait() == PM_ERROR_OUT_OF_HOST_MEMORY);
    CHECK(queue.wait_idle() == PM_ERROR_OUT_OF_HOST_MEMORY);

    std::atomic<bool> destroyed_work{};
    {
        SubmissionQueue ending;
        auto completion = std::make_shared<SubmissionCompletion>();
        CHECK(ending.enqueue([&] {
            destroyed_work = true;
            return PM_SUCCESS;
        }, ready, completion, accepted) == PM_SUCCESS);
    }
    CHECK(destroyed_work.load());
    return 0;
}
