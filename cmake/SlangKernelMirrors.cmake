# SPDX-License-Identifier: MIT

function(parallel_mater_add_slang_kernel_mirrors)
    set(slangc "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
    if(NOT slangc)
        message(FATAL_ERROR
            "parallel_mater_require_slang() must run before mirror generation")
    endif()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/slang/mirrors")
    file(GLOB slang_dependencies CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang")
    set(outputs)

    function(add_mirror module)
        set(entries ${ARGN})
        set(entry_arguments)
        foreach(entry IN LISTS entries)
            list(APPEND entry_arguments -entry "${entry}" -stage compute)
        endforeach()
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang")
        set(cuda_output "${generated_dir}/${module}.cu")
        add_custom_command(
            OUTPUT "${cuda_output}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${slangc}" "${source}"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                -target cuda -fp-mode precise -warnings-as-errors all
                ${entry_arguments} -o "${cuda_output}"
            DEPENDS ${slang_dependencies} "${slangc}"
            COMMENT "Validating ${module}.slang as CUDA"
            COMMAND_EXPAND_LISTS VERBATIM)
        set(module_outputs "${cuda_output}")
        if(TARGET CUDA::nvrtc)
            set(ptx_output "${generated_dir}/${module}.ptx")
            add_custom_command(
                OUTPUT "${ptx_output}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
                COMMAND "${slangc}" "${source}"
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                    -target ptx -fp-mode precise -warnings-as-errors all
                    -nvrtc-path "$<TARGET_FILE_DIR:CUDA::nvrtc>/nvrtc"
                    ${entry_arguments} -o "${ptx_output}"
                DEPENDS ${slang_dependencies} "${slangc}" CUDA::nvrtc
                COMMENT "Validating ${module}.slang as PTX"
                COMMAND_EXPAND_LISTS VERBATIM)
            list(APPEND module_outputs "${ptx_output}")
        endif()
        set(outputs ${outputs} ${module_outputs} PARENT_SCOPE)
    endfunction()

    add_mirror(fluid
        fluid_emit_cells fluid_compute_forces fluid_integrate
        fluid_reserve_contact_events fluid_gather_contact_events
        fluid_source_vacancies fluid_source_emit fluid_destroy_flags
        fluid_gather fluid_copy_initial)
    add_mirror(cloth cloth_project_volume cloth_break_bonds
        cloth_limit_strain cloth_update_surface)
    add_mirror(cloth_rope rope_cloth_sample_anchor rope_cloth_apply_anchor)
    add_mirror(cloth_smoke smoke_cloth_wind smoke_cloth_contact)
    add_mirror(fluid_cloth fluid_cloth_containment_forces
        cloth_apply_fluid_forces fluid_project_inside_cloth)
    add_mirror(fluid_rope fluid_rope_contacts fluid_rope_apply)
    add_mirror(fluid_smoke fluid_smoke_drag fluid_smoke_grid_drag
        fluid_smoke_exchange fluid_clear_inactive_keep)
    add_mirror(fluid_soft_body fluid_soft_refit fluid_soft_detect
        fluid_soft_solve fluid_soft_apply fluid_soft_recover)
    add_mirror(geometry_cloth cloth_constrain_bodies
        cloth_apply_body_corrections)
    add_mirror(geometry_constraints integrate_rigid_bodies_kernel
        compute_rigid_world_bounds_kernel broad_phase_rigid_pairs_kernel
        generate_rigid_leaf_pairs_kernel evaluate_rigid_leaf_pairs_kernel
        reduce_rigid_leaf_manifolds_kernel
        finalize_rigid_contact_manifolds_kernel clear_rigid_inputs_kernel
        capture_rigid_inputs_kernel prepare_parallel_contact_events_kernel
        load_rigid_contact_cache_kernel save_rigid_contact_cache_kernel
        gather_rigid_debug_samples_kernel clamp_rigid_speeds_kernel
        build_fixed_collision_groups_kernel)
    add_mirror(geometry_fluid fluid_static_contacts fluid_body_bounds_kernel
        fluid_index_body_cells fluid_detect_moving_contacts
        fluid_resolve_moving_contacts reduce_point_body_impulses)
    add_mirror(geometry_rope rope_advance)
    add_mirror(geometry_smoke smoke_emit_cells smoke_density_pressure
        smoke_pair_forces smoke_compute_vorticity smoke_advect smoke_emit
        smoke_deformable_bounds smoke_rigid_contact
        smoke_apply_rigid_impulses)
    add_mirror(geometry_soft_body deformable_collide
        soft_body_surface_contacts soft_body_apply_surface_contacts)
    add_mirror(rope_smoke smoke_rope_wind smoke_rope_contact)
    add_mirror(smoke_grid smoke_grid_initialize_cells
        smoke_grid_initialize_faces smoke_grid_clear_obstacles
        smoke_grid_raster_rigid smoke_grid_raster_deformable
        smoke_grid_raster_soft smoke_grid_resolve_particle_fields
        smoke_grid_splat_particles smoke_grid_advect_face
        smoke_grid_correct_face smoke_grid_cell_diagnostics
        smoke_grid_subgrid_force smoke_grid_apply_face_forces
        smoke_grid_apply_face_boundaries smoke_grid_divergence
        smoke_grid_pressure_smooth smoke_grid_pressure_residual
        smoke_grid_restrict_residual smoke_grid_clear_scalar
        smoke_grid_restrict_open smoke_grid_mark_domain_boundaries
        smoke_grid_prolong_add smoke_grid_compare_residual
        smoke_grid_reset_residual smoke_grid_project_face smoke_grid_trace
        smoke_grid_soft_body_force smoke_grid_cloth_force
        smoke_grid_apply_cloth_force smoke_grid_rigid_force)
    add_mirror(soft_body deformable_predict soft_body_measure_momentum
        soft_body_restore_momentum soft_body_project_rest_shape
        deformable_project_links deformable_project_links_warp
        soft_body_finalize_velocities soft_body_apply_contact_friction
        soft_body_damp_springs soft_body_damp_springs_warp
        soft_body_update_surface)
    add_mirror(soft_body_cloth soft_cloth_detect soft_cloth_apply_soft
        soft_cloth_apply_cloth)
    add_mirror(soft_body_rope rope_soft_sample_anchor
        rope_soft_scatter_anchor rope_soft_apply)
    add_mirror(soft_body_smoke smoke_soft_body_wind smoke_soft_body_contact)
    add_mirror(world query_rigid_hit_box_kernel query_particle_hit_box_kernel)
    add_mirror(avbd_cuda prepare_avbd_kernel solve_avbd_kernel)

    add_custom_target(parallel-mater-slang-kernel-mirrors ALL
        DEPENDS ${outputs})
endfunction()
