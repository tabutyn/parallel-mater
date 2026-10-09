// SPDX-License-Identifier: MIT
#include "avbd_cases.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef PARALLEL_MATER_SLANG_AVBD_PTX
#error "PARALLEL_MATER_SLANG_AVBD_PTX must name the generated PTX file"
#endif

namespace {
using namespace parallel_mater::avbd;

struct Expected {
    float values[8]{};
    unsigned solved{1};
    unsigned block_index{~0U};
    const char *name{};
};

struct SlangAvbdInput {
    unsigned operation{};
    float values[64]{};
};

struct SlangAvbdOutput {
    float values[8]{};
    unsigned solved{};
};

struct SlangBuffer {
    void *data{};
    std::size_t count{};
};

struct SlangGlobalParams {
    SlangBuffer inputs{};
    SlangBuffer outputs{};
    unsigned count{};
    unsigned padding{};
};

static_assert(sizeof(SlangAvbdInput) == 260);
static_assert(sizeof(SlangAvbdOutput) == 36);
static_assert(sizeof(SlangGlobalParams) == 40);

float &input_value(SlangAvbdInput &input, unsigned index) {
    return input.values[index];
}

float output_value(const SlangAvbdOutput &output, unsigned index) {
    return output.values[index];
}

bool close(float actual, float expected) {
    const float scale = std::max({1.0F, std::fabs(actual), std::fabs(expected)});
    return std::fabs(actual - expected) <= 2.5e-5F * scale;
}

void append_block_cases(std::vector<SlangAvbdInput> &inputs,
                        std::vector<Expected> &expected) {
    for (unsigned index = 0; index < avbd_cases::block_count; ++index) {
        const Block block = avbd_cases::make_block(index);
        SlangAvbdInput input{};
        input.operation = 0;
        for (unsigned i = 0; i < 6; ++i) {
            for (unsigned j = 0; j < 6; ++j)
                input_value(input, i * 6 + j) = block.h[i][j];
            input_value(input, 36 + i) = block.g[i];
        }
        Expected result{};
        result.name = "coupled block solve";
        result.block_index = index;
        Vector6 delta{};
        result.solved = solve(block, delta) ? 1U : 0U;
        for (unsigned i = 0; i < 6; ++i) result.values[i] = delta.v[i];
        inputs.push_back(input);
        expected.push_back(result);
    }
}

void append_contact_case(std::vector<SlangAvbdInput> &inputs,
                         std::vector<Expected> &expected,
                         ContactForce force, float friction) {
    SlangAvbdInput input{};
    input.operation = 1;
    input_value(input, 0) = force.normal;
    input_value(input, 1) = force.tangent0;
    input_value(input, 2) = force.tangent1;
    input_value(input, 3) = friction;
    const ContactForce projected = project_contact(force, friction);
    const ContactForce scales = contact_stiffness_scales(force, friction);
    Expected result{};
    result.name = "contact cone projection";
    result.values[0] = projected.normal;
    result.values[1] = projected.tangent0;
    result.values[2] = projected.tangent1;
    result.values[3] = scales.normal;
    result.values[4] = scales.tangent0;
    result.values[5] = scales.tangent1;
    inputs.push_back(input);
    expected.push_back(result);
}

void append_dual_case(std::vector<SlangAvbdInput> &inputs,
                      std::vector<Expected> &expected, Dual dual,
                      float error, float beta, float lower, float upper,
                      float stiffness) {
    SlangAvbdInput input{};
    input.operation = 2;
    input_value(input, 0) = dual.lambda;
    input_value(input, 1) = dual.penalty;
    input_value(input, 2) = error;
    input_value(input, 3) = beta;
    input_value(input, 4) = lower;
    input_value(input, 5) = upper;
    input_value(input, 6) = stiffness;
    const Dual warm = warm_start(dual, stiffness);
    const Dual advanced = update_dual(dual, error, beta, lower, upper, stiffness);
    Expected result{};
    result.name = "augmented-Lagrangian dual update";
    result.values[0] = warm.lambda;
    result.values[1] = warm.penalty;
    result.values[2] = force(dual, error, lower, upper);
    result.values[3] = advanced.lambda;
    result.values[4] = advanced.penalty;
    inputs.push_back(input);
    expected.push_back(result);
}

void append_row_case(std::vector<SlangAvbdInput> &inputs,
                     std::vector<Expected> &expected, Row row,
                     Vector6 a, Vector6 b, float dt,
                     const std::array<float, 6> &inertia, bool first) {
    SlangAvbdInput input{};
    input.operation = 3;
    for (unsigned i = 0; i < 6; ++i) {
        input_value(input, i) = row.a.v[i];
        input_value(input, 6 + i) = row.b.v[i];
        input_value(input, 22 + i) = a.v[i];
        input_value(input, 28 + i) = b.v[i];
        input_value(input, 35 + i) = inertia[i];
    }
    input_value(input, 12) = row.dual.lambda;
    input_value(input, 13) = row.dual.penalty;
    input_value(input, 14) = row.error;
    input_value(input, 15) = row.velocity;
    input_value(input, 16) = row.lower;
    input_value(input, 17) = row.upper;
    input_value(input, 18) = row.stiffness;
    input_value(input, 19) = row.damping;
    input_value(input, 20) = row.beta;
    input_value(input, 21) = row.reference_force;
    input_value(input, 34) = dt;
    input_value(input, 41) = first ? 1.0F : 0.0F;

    Block block{};
    for (unsigned i = 0; i < 6; ++i) block.h[i][i] = inertia[i];
    accumulate(block, row, first, a, b, dt);
    const Row advanced = advance(row, a, b);
    Expected result{};
    result.name = "row force and block accumulation";
    result.values[0] = row_error(row, a, b);
    result.values[1] = row_trial(row, a, b, dt);
    result.values[2] = row_force(row, a, b, dt);
    result.values[3] = advanced.dual.lambda;
    result.values[4] = advanced.dual.penalty;
    for (unsigned i = 0; i < 6; ++i) {
        result.values[5] += block.g[i];
        result.values[6] += block.h[i][i];
    }
    inputs.push_back(input);
    expected.push_back(result);
}
}

int main() {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;

    std::vector<SlangAvbdInput> host_inputs;
    std::vector<Expected> expected;
    append_block_cases(host_inputs, expected);
    append_contact_case(host_inputs, expected, {-2.0F, 3.0F, 4.0F}, 0.7F);
    append_contact_case(host_inputs, expected, {6.0F, 1.0F, -2.0F}, 0.5F);
    append_contact_case(host_inputs, expected, {2.0F, 8.0F, 6.0F}, 0.25F);
    append_contact_case(host_inputs, expected, {4.0F, 2.0F, -3.0F}, 0.0F);
    append_dual_case(host_inputs, expected, {3.0F, 7.0F}, 0.4F, 10.0F,
                     -100.0F, 100.0F, maximum_penalty);
    append_dual_case(host_inputs, expected, {90.0F, 9.0F}, 4.0F, 3.0F,
                     -100.0F, 100.0F, maximum_penalty);
    append_dual_case(host_inputs, expected, {12.0F, 15.0F}, -0.2F, 8.0F,
                     -25.0F, 25.0F, 40.0F);

    Row bilateral{};
    bilateral.a = {{1.0F, -2.0F, 0.5F, 0.1F, 0.2F, -0.3F}};
    bilateral.b = {{-1.0F, 1.5F, 0.25F, -0.4F, 0.0F, 0.8F}};
    bilateral.dual = {3.0F, 120.0F};
    bilateral.error = -0.04F;
    bilateral.velocity = 0.3F;
    bilateral.lower = -75.0F;
    bilateral.upper = 75.0F;
    bilateral.stiffness = maximum_penalty;
    bilateral.damping = 2.5F;
    bilateral.beta = 400.0F;
    bilateral.reference_force = 1.25F;
    append_row_case(host_inputs, expected, bilateral,
                    {{0.02F, -0.01F, 0.03F, 0.1F, -0.2F, 0.05F}},
                    {{-0.04F, 0.02F, 0.0F, -0.03F, 0.07F, 0.02F}},
                    1.0F / 240.0F, {2, 3, 4, 5, 6, 7}, true);

    Row unilateral = bilateral;
    unilateral.dual = {0.0F, 40.0F};
    unilateral.error = -2.0F;
    unilateral.lower = 0.0F;
    unilateral.upper = maximum_penalty;
    unilateral.damping = 0.0F;
    append_row_case(host_inputs, expected, unilateral, {}, {},
                    1.0F / 60.0F, {1, 2, 3, 4, 5, 6}, false);

    SlangAvbdInput *inputs{};
    SlangAvbdOutput *outputs{};
    const std::size_t count = host_inputs.size();
    if (cudaMallocManaged(&inputs, count * sizeof(*inputs)) != cudaSuccess
        || cudaMallocManaged(&outputs, count * sizeof(*outputs)) != cudaSuccess) return 1;
    std::copy(host_inputs.begin(), host_inputs.end(), inputs);

    if (cudaSetDevice(0) != cudaSuccess) return 1;
    SlangGlobalParams params{{inputs, count}, {outputs, count},
                             static_cast<unsigned>(count), 0};
    CUmodule module{};
    CUfunction function{};
    CUdeviceptr global_params{};
    std::size_t global_size{};
    if (cuModuleLoad(&module, PARALLEL_MATER_SLANG_AVBD_PTX) != CUDA_SUCCESS
        || cuModuleGetGlobal(&global_params, &global_size, module,
                             "SLANG_globalParams") != CUDA_SUCCESS
        || global_size != sizeof(params)
        || cuMemcpyHtoD(global_params, &params, sizeof(params)) != CUDA_SUCCESS
        || cuModuleGetFunction(&function, module,
                               "avbdConformanceMain") != CUDA_SUCCESS
        || cuLaunchKernel(function, static_cast<unsigned>((count + 63) / 64), 1, 1,
                          64, 1, 1, 0, nullptr, nullptr, nullptr) != CUDA_SUCCESS
        || cuCtxSynchronize() != CUDA_SUCCESS) return 1;

    bool passed = true;
    for (std::size_t fixture = 0; fixture < count; ++fixture) {
        if (expected[fixture].block_index != ~0U) {
            avbd_cases::Result actual_result{};
            avbd_cases::Result expected_result{};
            actual_result.solved = outputs[fixture].solved != 0;
            expected_result.solved = expected[fixture].solved != 0;
            for (unsigned component = 0; component < 8; ++component) {
                actual_result.value[component] = output_value(outputs[fixture], component);
                expected_result.value[component] = expected[fixture].values[component];
            }
            if (!avbd_cases::validate(expected[fixture].block_index, actual_result)
                || !avbd_cases::equivalent(expected[fixture].block_index,
                                            actual_result, expected_result)) {
                passed = false;
                std::fprintf(stderr, "Slang AVBD block fixture %u failed\n",
                             expected[fixture].block_index);
            }
            continue;
        }
        if (outputs[fixture].solved != expected[fixture].solved) {
            passed = false;
            std::fprintf(stderr, "Slang AVBD fixture %zu (%s): solved=%u expected=%u\n",
                         fixture, expected[fixture].name, outputs[fixture].solved,
                         expected[fixture].solved);
        }
        for (unsigned component = 0; component < 8; ++component) {
            const float actual = output_value(outputs[fixture], component);
            if (!close(actual, expected[fixture].values[component])) {
                passed = false;
                std::fprintf(stderr,
                    "Slang AVBD fixture %zu (%s) [%u]: %.9g expected %.9g\n",
                    fixture, expected[fixture].name, component, actual,
                    expected[fixture].values[component]);
            }
        }
    }
    cudaFree(outputs);
    cudaFree(inputs);
    cuModuleUnload(module);
    if (passed)
        std::printf("%zu Slang CUDA/portable AVBD fixtures passed\n", count);
    return passed ? 0 : 1;
}
