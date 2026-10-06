// SPDX-License-Identifier: MIT

#include "capture_format.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace parity = parallel_mater::test::parity;

namespace {

bool parse_capture(std::istream &stream, parity::Capture &capture,
                   std::string_view label) {
    std::string error;
    if (!parity::parse(stream, capture, error)) {
        std::cerr << label << ": " << error << '\n';
        return false;
    }
    return true;
}

int self_test() {
    std::ostringstream reference_text;
    parity::Writer reference_writer(reference_text, "coupled", "cuda");
    reference_writer.unsigned_value("rigid.count", 2U);
    reference_writer.unsigned_value("rigid[0].id", 42U);
    reference_writer.floating_value("rigid[0].position.x", 10.0);
    reference_writer.floating_value("rigid[0].velocity.x", -0.25);

    std::ostringstream candidate_text;
    parity::Writer candidate_writer(candidate_text, "coupled", "metal");
    candidate_writer.unsigned_value("rigid.count", 2U);
    candidate_writer.unsigned_value("rigid[0].id", 42U);
    candidate_writer.floating_value("rigid[0].position.x", 10.009);
    candidate_writer.floating_value("rigid[0].velocity.x", -0.25001);

    parity::Capture reference;
    parity::Capture candidate;
    std::istringstream reference_input(reference_text.str());
    std::istringstream candidate_input(candidate_text.str());
    if (!parse_capture(reference_input, reference, "self-test reference") ||
        !parse_capture(candidate_input, candidate, "self-test candidate")) {
        return EXIT_FAILURE;
    }
    if (!parity::compare(reference, candidate).matches) {
        std::cerr << "self-test rejected an in-tolerance capture\n";
        return EXIT_FAILURE;
    }

    candidate.records[1].unsigned_value = 43U;
    if (parity::compare(reference, candidate).matches) {
        std::cerr << "self-test accepted an exact-field mismatch\n";
        return EXIT_FAILURE;
    }
    candidate.records[1].unsigned_value = 42U;
    candidate.records[2].floating_value = 10.02;
    if (parity::compare(reference, candidate).matches) {
        std::cerr << "self-test accepted an out-of-tolerance float\n";
        return EXIT_FAILURE;
    }

    std::istringstream non_finite(
        "parallel-mater-parity-v1\nscenario bad\nbackend metal\nf x nan\n");
    parity::Capture rejected;
    std::string error;
    if (parity::parse(non_finite, rejected, error)) {
        std::cerr << "self-test accepted a non-finite float\n";
        return EXIT_FAILURE;
    }

    std::cout << "parity capture comparator self-test passed\n";
    return EXIT_SUCCESS;
}

bool parse_tolerance(std::string_view text, double &value) {
    std::string owned(text);
    char *end = nullptr;
    value = std::strtod(owned.c_str(), &end);
    return end == owned.c_str() + owned.size();
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        return self_test();
    }
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " REFERENCE CANDIDATE [--abs VALUE] [--rel VALUE]\n";
        return 2;
    }

    parity::ComparisonOptions options;
    for (int index = 3; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing comparison option value\n";
            return 2;
        }
        const std::string_view option(argv[index]);
        double value = 0.0;
        if (!parse_tolerance(argv[index + 1], value)) {
            std::cerr << "invalid tolerance: " << argv[index + 1] << '\n';
            return 2;
        }
        if (option == "--abs") {
            options.absolute_tolerance = value;
        } else if (option == "--rel") {
            options.relative_tolerance = value;
        } else {
            std::cerr << "unknown option: " << option << '\n';
            return 2;
        }
    }

    std::ifstream reference_file(argv[1]);
    std::ifstream candidate_file(argv[2]);
    if (!reference_file) {
        std::cerr << "cannot open reference capture: " << argv[1] << '\n';
        return 2;
    }
    if (!candidate_file) {
        std::cerr << "cannot open candidate capture: " << argv[2] << '\n';
        return 2;
    }

    parity::Capture reference;
    parity::Capture candidate;
    if (!parse_capture(reference_file, reference, argv[1]) ||
        !parse_capture(candidate_file, candidate, argv[2])) {
        return 2;
    }

    const parity::ComparisonResult result =
        parity::compare(reference, candidate, options);
    std::cout << result.message << '\n';
    return result.matches ? EXIT_SUCCESS : EXIT_FAILURE;
}
