// SPDX-License-Identifier: MIT
#pragma once

#include <simulation/types.hpp>

namespace parallel_mater::gallery {

struct Camera {
    Vec3 eye{7.5F, 5.0F, 9.0F};
    Vec3 target{0.0F, 1.5F, 0.0F};
    Vec3 up{0.0F, 1.0F, 0.0F};
    float vertical_field_of_view_degrees{48.0F};
};

} // namespace parallel_mater::gallery
