// SPDX-License-Identifier: MIT
// Scheduling ABI shared by Slang and C++; no solver equations here.
PM_AVBD_STAGE(impact_gaps, 0)
PM_AVBD_STAGE(impact_bodies, 1)
PM_AVBD_STAGE(impact_constraints, 2)
PM_AVBD_STAGE(solve_color, 3)
PM_AVBD_STAGE(advance_duals, 4)
PM_AVBD_STAGE(free_mean, 5)
PM_AVBD_STAGE(free_subtract, 6)
PM_AVBD_STAGE(free_clear, 7)
PM_AVBD_STAGE(apply_states, 8)
PM_AVBD_STAGE(project_hinges, 9)
PM_AVBD_STAGE(finish_phase, 10)
