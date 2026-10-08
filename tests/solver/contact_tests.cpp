// SPDX-License-Identifier: MIT
#include "contact_cases.hpp"
#include <iostream>

int main() {
    using parallel_mater::solver::contact_iteration_budget;
    if (contact_iteration_budget(false) != 8 || contact_iteration_budget(true) != 32) return 1;
    for (unsigned index = 0; index < contact_cases::count; ++index) {
        if (!contact_cases::close(contact_cases::evaluate(index), contact_cases::expected(index))) {
            std::cerr << "Contact equation fixture failed: " << index << '\n';
            return 1;
        }
    }
    std::cout << contact_cases::count << " portable contact fixtures passed\n";
}
