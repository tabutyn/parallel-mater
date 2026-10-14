# SPDX-License-Identifier: MIT

foreach(required
        PM_SOURCE_DIR PM_OUTPUT_DIR PM_SLANGC PM_XCRUN PM_METALLIB PM_XXD)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "BuildMetalPhysics.cmake requires ${required}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${PM_OUTPUT_DIR}")
set(PM_MODULE_CACHE "${PM_OUTPUT_DIR}/module-cache")
set(PM_METAL_AIR_FILES)

function(pm_metal_module name source)
    cmake_parse_arguments(MODULE "" "" "DEFINES;ENTRIES" ${ARGN})
    set(msl "${PM_OUTPUT_DIR}/${name}.metal")
    set(air "${PM_OUTPUT_DIR}/${name}.air")
    set(slang_command
        "${PM_SLANGC}"
        "${PM_SOURCE_DIR}/src/slang/${source}"
        -I "${PM_SOURCE_DIR}/src/slang")
    foreach(define IN LISTS MODULE_DEFINES)
        list(APPEND slang_command "-D${define}")
    endforeach()
    foreach(entry IN LISTS MODULE_ENTRIES)
        list(APPEND slang_command -entry "${entry}")
    endforeach()
    list(APPEND slang_command
        -target metal
        -capability metallib_3_2
        -fp-mode precise
        -O2
        -line-directive-mode none
        -o "${msl}")
    execute_process(COMMAND ${slang_command} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "CLANG_MODULE_CACHE_PATH=${PM_MODULE_CACHE}"
            "${PM_XCRUN}" -sdk macosx metal
            -std=metal4.0 -mmacosx-version-min=26.0
            -fno-fast-math -ffp-contract=fast -Wno-unused-variable
            "-fmodules-cache-path=${PM_MODULE_CACHE}"
            -c "${msl}" -o "${air}"
        COMMAND_ERROR_IS_FATAL ANY)
    set(PM_METAL_AIR_FILES ${PM_METAL_AIR_FILES} "${air}" PARENT_SCOPE)
endfunction()

pm_metal_module(fluid metal_fluid.slang
    DEFINES PM_TARGET_METAL=1 PM_TARGET_CUDA=0
    ENTRIES
        pm_fluid_cell_keys pm_fluid_forces pm_fluid_integrate
        pm_fluid_radix_histogram_0 pm_fluid_radix_histogram_1
        pm_fluid_radix_histogram_2 pm_fluid_radix_histogram_3
        pm_fluid_radix_histogram_4 pm_fluid_radix_histogram_5
        pm_fluid_radix_histogram_6 pm_fluid_radix_histogram_7
        pm_fluid_radix_prefix_blocks pm_fluid_radix_prefix_buckets
        pm_fluid_capture_spawn_baseline
        pm_fluid_radix_scatter_0 pm_fluid_radix_scatter_1
        pm_fluid_radix_scatter_2 pm_fluid_radix_scatter_3
        pm_fluid_radix_scatter_4 pm_fluid_radix_scatter_5
        pm_fluid_radix_scatter_6 pm_fluid_radix_scatter_7)
pm_metal_module(rigid_integrate rigid_integrate.slang
    ENTRIES pm_noop pm_rigid_integrate pm_rigid_increment_substep
            pm_rigid_clear_accumulators)
pm_metal_module(rigid_broadphase rigid_broadphase.slang
    ENTRIES pm_rigid_world_bounds pm_rigid_pair_filter
            pm_rigid_pair_count_rows pm_rigid_pair_prefix_rows
            pm_rigid_pair_scatter_rows)
pm_metal_module(rigid_contact_cache rigid_contact_cache.slang
    ENTRIES pm_rigid_contact_match_cache pm_rigid_advance_substep)
pm_metal_module(rigid_narrowphase rigid_narrowphase.slang
    ENTRIES pm_rigid_contact_generate pm_rigid_contact_generate_cooperative)
pm_metal_module(rigid_constraints rigid_constraints.slang
    ENTRIES pm_avbd_collision_groups pm_avbd_prepare pm_avbd_solve)
pm_metal_module(particle_rigid particle_rigid.slang
    ENTRIES pm_particles_rigid)
pm_metal_module(cloth_rigid cloth_rigid.slang
    ENTRIES pm_cloth_rigid_surface_serial)
pm_metal_module(rope_step rope_step.slang
    ENTRIES pm_rope_step_serial)
pm_metal_module(cloth cloth.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_cloth_predict pm_cloth_damage pm_cloth_project_bonds
            pm_cloth_apply_bonds pm_cloth_limit_strain_serial
            pm_cloth_project_volume pm_cloth_finalize
            pm_cloth_surface_update pm_cloth_impact)
pm_metal_module(cloth_rope cloth_rope.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_rope_cloth_sample)
pm_metal_module(soft_body soft_body.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_soft_body_predict pm_soft_body_project_bonds
            pm_soft_body_apply_bonds pm_soft_body_shape_matching
            pm_soft_body_finalize pm_soft_body_damping_prepare
            pm_soft_body_damping_apply pm_soft_body_surface_update)
pm_metal_module(fluid_cloth fluid_cloth.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_fluid_cloth_forces pm_fluid_cloth_apply
            pm_fluid_cloth_surface_update pm_fluid_cloth_project)
pm_metal_module(fluid_rope fluid_rope.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_fluid_rope_contacts pm_fluid_rope_apply)
pm_metal_module(fluid_soft_body fluid_soft_body.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_fluid_soft_contact pm_fluid_soft_solve pm_fluid_soft_apply
            pm_fluid_soft_surface_update pm_fluid_soft_sweep_on
            pm_fluid_soft_recover pm_fluid_soft_recover_current
            pm_fluid_soft_copy_surface pm_fluid_soft_rigid_recover)
pm_metal_module(fluid_smoke fluid_smoke.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_fluid_smoke_update pm_fluid_smoke_compact)
pm_metal_module(fluid_source fluid_source.slang ENTRIES pm_fluid_source)
pm_metal_module(fluid_destroy fluid_destroy.slang ENTRIES pm_fluid_destroy)
pm_metal_module(smoke_particles smoke_particles.slang
    ENTRIES pm_smoke_grid_clear pm_smoke_particle_density
            pm_smoke_particle_vorticity pm_smoke_particle_forces
            pm_smoke_particle_integrate pm_smoke_grid_advect_particles)
pm_metal_module(smoke_emit smoke_emit.slang ENTRIES pm_smoke_emit)
pm_metal_module(smoke_grid_splat smoke_grid_splat.slang
    ENTRIES pm_smoke_grid_splat_generate
            pm_smoke_grid_splat_histogram_0 pm_smoke_grid_splat_histogram_1
            pm_smoke_grid_splat_histogram_2 pm_smoke_grid_splat_histogram_3
            pm_smoke_grid_splat_prefix_blocks
            pm_smoke_grid_splat_prefix_buckets
            pm_smoke_grid_splat_scatter_0 pm_smoke_grid_splat_scatter_1
            pm_smoke_grid_splat_scatter_2 pm_smoke_grid_splat_scatter_3
            pm_smoke_grid_splat_resolve)
pm_metal_module(smoke_grid_pressure smoke_grid_pressure.slang
    ENTRIES pm_smoke_grid_advect_forward pm_smoke_grid_advect_reverse
            pm_smoke_grid_correct_face pm_smoke_grid_apply_face_forces
            pm_smoke_pressure_cycle_begin pm_smoke_pressure_cycle_end
            pm_smoke_grid_pressure_begin pm_smoke_grid_mark_domain_boundaries
            pm_smoke_grid_divergence_parallel pm_smoke_grid_project_face_parallel
            pm_smoke_grid_apply_face_boundaries pm_smoke_grid_subgrid_force
            pm_smoke_grid_cell_diagnostics_pre
            pm_smoke_grid_cell_diagnostics_post
            pm_smoke_pressure_restrict_open_1 pm_smoke_pressure_restrict_open_2
            pm_smoke_pressure_restrict_open_3
            pm_smoke_pressure_smooth_0_forward
            pm_smoke_pressure_smooth_0_backward
            pm_smoke_pressure_smooth_1_forward
            pm_smoke_pressure_smooth_1_backward
            pm_smoke_pressure_smooth_2_forward
            pm_smoke_pressure_smooth_2_backward
            pm_smoke_pressure_smooth_3_forward
            pm_smoke_pressure_smooth_3_backward
            pm_smoke_pressure_residual_0 pm_smoke_pressure_residual_1
            pm_smoke_pressure_residual_2 pm_smoke_pressure_restrict_residual_0
            pm_smoke_pressure_restrict_residual_1
            pm_smoke_pressure_restrict_residual_2
            pm_smoke_pressure_clear_1 pm_smoke_pressure_clear_2
            pm_smoke_pressure_clear_3 pm_smoke_pressure_prolong_1
            pm_smoke_pressure_prolong_2 pm_smoke_pressure_prolong_3)
pm_metal_module(smoke_grid_obstacles smoke_grid_obstacles.slang
    ENTRIES pm_smoke_grid_raster_clear pm_smoke_grid_merge_obstacles)
pm_metal_module(smoke_surface smoke_surface.slang
    ENTRIES pm_smoke_surface_grid_raster pm_smoke_surface_grid_select
            pm_smoke_surface_grid_resolve pm_smoke_surface_grid_force
            pm_smoke_surface)
pm_metal_module(smoke_rigid_grid smoke_rigid_grid.slang
    ENTRIES pm_smoke_grid_raster_rigid pm_smoke_grid_select_rigid
            pm_smoke_grid_resolve_rigid)
pm_metal_module(smoke_rigid smoke_rigid.slang ENTRIES pm_smoke_rigid)
pm_metal_module(rope_smoke rope_smoke.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_smoke_rope_wind pm_smoke_rope)
pm_metal_module(rope_soft_metal rope_soft_metal.slang
    ENTRIES pm_rope_soft_pack pm_rope_soft_sample
            pm_rope_soft_anchor_weights pm_rope_soft_apply
            pm_rope_soft_surface_update pm_rope_soft_update_endpoint
            pm_rope_soft_finalize)
pm_metal_module(rope_predict rope_predict.slang ENTRIES pm_rope_predict)
pm_metal_module(particle_coupling particle_coupling.slang
    ENTRIES pm_particle_coupling_serial)
pm_metal_module(soft_contact_momentum soft_contact_momentum.slang
    ENTRIES pm_soft_contact_accumulators_clear pm_soft_body_measure_momentum
            pm_soft_body_restore_momentum pm_soft_contact_friction
            pm_soft_contact_finish)
pm_metal_module(soft_body_cloth soft_body_cloth.slang
    DEFINES PM_TARGET_METAL=1
    ENTRIES pm_soft_cloth_clear_counts pm_soft_cloth_contact
            pm_soft_cloth_apply_soft pm_soft_cloth_apply
            pm_soft_cloth_surface_update pm_soft_cloth_soft_surface_update)
pm_metal_module(paint paint.slang ENTRIES pm_paint)

set(metallib "${PM_OUTPUT_DIR}/parallel_mater.metallib")
set(metallib_include "${PM_OUTPUT_DIR}/parallel_mater_metallib.inc")
execute_process(
    COMMAND "${PM_METALLIB}" ${PM_METAL_AIR_FILES} -o "${metallib}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(
    COMMAND "${PM_XXD}" -i -n parallel_mater_metallib_data
            "${metallib}" "${metallib_include}"
    COMMAND_ERROR_IS_FATAL ANY)
