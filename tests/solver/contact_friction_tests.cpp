// SPDX-License-Identifier: MIT
#include "contact_friction_cases.hpp"

#include <iostream>

int main() {
    for (unsigned index = 0U; index < contact_friction_cases::count; ++index) {
        if (!contact_friction_cases::evaluate(index)) {
            std::cerr << "Contact-friction fixture failed: " << index << '\n';
            return 1;
        }
    }
    std::cout << contact_friction_cases::count
              << " contact-friction fixtures passed\n";
}
