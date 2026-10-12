/* Proposed API usage: syntax-checked, NOT linkable against today's library.
 * Prerequisites: all resources created/bound/initialized; geometry built;
 * pipeline compatible; query pool has >= 2 timestamp slots; pool permits
 * individual reset; prior use of cmd, output and query slots has completed.
 * The simulation remains on this queue family. This example assumes output
 * uses the same family as its renderer; see the reference for ownership moves.
 */
#include "../api/parallel_mater.h"
#include <string.h>

#define TRY(call) do { PmResult r_ = (call); \
    if (r_ != PM_SUCCESS) return r_; } while (0)

/* REF_BEGIN: example_record */
PmResult record_frame(PmCommandBuffer cmd,
    PmPhysicsPipeline pipeline, PmSimulation sim,
    PmBuffer output, uint32_t bodyCount,
    PmQueryPool timestamps, const PmStepInfo* step)
{
    PmCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = PM_COMMAND_BUFFER_ONE_TIME_BIT;
    TRY(pmResetCommandBuffer(cmd));
    TRY(pmBeginCommandBuffer(cmd, &begin));
    TRY(pmCmdResetQueryPool(cmd, timestamps, 0, 2));
    TRY(pmCmdWriteTimestamp(cmd,
        PM_STAGE_ALL_COMMANDS_BIT, timestamps, 0));

    PmMemoryBarrier before = {
        PM_STAGE_ALL_COMMANDS_BIT, PM_ACCESS_WRITE_BIT,
        PM_STAGE_PHYSICS_BIT,
        PM_ACCESS_READ_BIT | PM_ACCESS_WRITE_BIT
    };
    TRY(pmCmdPipelineBarrier(cmd, 1, &before, 0, NULL));
    TRY(pmCmdBindPhysicsPipeline(cmd, pipeline));
    TRY(pmCmdStepSimulation(cmd, sim, step));

    PmMemoryBarrier after = {
        PM_STAGE_PHYSICS_BIT, PM_ACCESS_WRITE_BIT,
        PM_STAGE_TRANSFER_BIT, PM_ACCESS_READ_BIT
    };
    TRY(pmCmdPipelineBarrier(cmd, 1, &after, 0, NULL));
    PmBufferRange destination = {
        output, 0, (PmDeviceSize)bodyCount * 64u
    };
    TRY(pmCmdCopyRigidStates(
        cmd, sim, 0, bodyCount, &destination));
    TRY(pmCmdWriteTimestamp(cmd,
        PM_STAGE_ALL_COMMANDS_BIT, timestamps, 1));
    return pmEndCommandBuffer(cmd);
}
/* REF_END: example_record */

/* REF_BEGIN: example_submit */
PmResult submit_frame(PmQueue queue, PmCommandBuffer cmd,
    PmSemaphore renderDone, uint64_t renderValue,
    PmSemaphore physicsDone, uint64_t physicsValue,
    PmFence completion)
{
    PmSemaphoreSubmitInfo wait = {
        renderDone, renderValue, PM_STAGE_ALL_COMMANDS_BIT
    };
    PmSemaphoreSubmitInfo signal = {
        physicsDone, physicsValue, PM_STAGE_ALL_COMMANDS_BIT
    };
    PmSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitCount = 1;
    submit.pWaits = &wait;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalCount = 1;
    submit.pSignals = &signal;
    return pmQueueSubmit(queue, &submit, completion);
}
/* REF_END: example_submit */

/* No WAIT flag: the caller can poll after its frame fence/timeline completes. */
PmResult read_gpu_nanoseconds(PmDevice device, PmQueryPool timestamps,
    uint32_t validBits, double tickNanoseconds, double* nanoseconds)
{
    if (validBits == 0 || validBits > 64 || !nanoseconds ||
        !(tickNanoseconds > 0)) return PM_ERROR_INVALID_ARGUMENT;
    uint64_t samples[4] = {0, 0, 0, 0};
    TRY(pmGetQueryPoolResults(device, timestamps, 0, 2,
        sizeof(samples), samples, 2u * sizeof(uint64_t),
        PM_QUERY_RESULT_WITH_AVAILABILITY_BIT));
    if (!samples[1] || !samples[3]) return PM_NOT_READY;
    const uint64_t mask = validBits == 64 ? UINT64_MAX :
        ((UINT64_C(1) << validBits) - UINT64_C(1));
    *nanoseconds = (double)((samples[2] - samples[0]) & mask)
        * tickNanoseconds;
    return PM_SUCCESS;
}
