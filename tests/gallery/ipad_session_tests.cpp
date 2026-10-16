// SPDX-License-Identifier: MIT
#include "gallery_session.hpp"
#include <cstdio>
#include <stdexcept>
using parallel_mater::gallery::IPadGallerySession;
int main() {
    try {
        IPadGallerySession session(PM_IPAD_TEST_RESOURCES, "iPad13,8");
        const auto names = IPadGallerySession::scene_names();
        if (names.size() != 27) throw std::runtime_error("Scene catalog mismatch");
        session.resize(480, 360);
        for (unsigned i = 0; i < names.size(); ++i) {
            session.select(i);
            for (unsigned step = 0; step < 3; ++step) session.frame();
            session.validate_snapshot();
            if (session.frames != 3 || !session.timing.latest() || session.timing.latest()->physics_steps != 1)
                throw std::runtime_error("Frame accounting mismatch");
            session.paused = true; session.frame();
            if (session.frames != 3 || session.timing.latest()->physics_steps != 0)
                throw std::runtime_error("Paused frame advanced physics");
            session.paused = false;
            session.reset();
            if (session.frames != 0 || session.timing.latest())
                throw std::runtime_error("Reset retained stale timing");
            std::printf("PASS iPad session: %s\n", names[i].c_str());
        }
        bool rejected = false;
        const auto old_selection = session.selection();
        try { session.select(999); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected || session.selection() != old_selection)
            throw std::runtime_error("Invalid scene replaced live state");
        session.select(2); session.release(); session.frame(); session.release();
        if (session.frames != 0) throw std::runtime_error("Point restore did not reset");
        return 0;
    } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
