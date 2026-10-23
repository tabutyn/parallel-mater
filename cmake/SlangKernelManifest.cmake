# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# One inventory drives the production CUDA adapter and every compile mirror.
# Module-specific CUDA entries exist only where the generated CUDA ABI uses a
# thin adapter entry point instead of the backend-neutral compute entry point.
set(PARALLEL_MATER_SLANG_KERNEL_MODULES
    fluid
    geometry_constraints
    queries
    geometry
    primitives
    cloth
    soft_body
    soft_body_cloth
    cloth_rope
    soft_body_rope
    fluid_cloth
    fluid_rope
    fluid_smoke
    cloth_smoke
    soft_body_smoke
    rope_smoke
    fluid_soft_body
    geometry_cloth
    geometry_fluid
    geometry_smoke
    geometry_soft_body
    smoke_grid
    rope)

set(PARALLEL_MATER_SLANG_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/slang")

set(PARALLEL_MATER_SLANG_fluid_ENTRIES
    fluid_emit_cells fluid_compute_forces fluid_integrate
    fluid_reserve_contact_events fluid_gather_contact_events
    fluid_source_vacancies fluid_source_emit fluid_destroy_flags
    fluid_gather fluid_copy_initial fluid_emit_pair_cells
    fluid_count_pair_neighbors fluid_apply_pair_pressure_contact
    fluid_apply_pair_viscosity fluid_apply_miscible_pair_pressure_contact
    fluid_apply_miscible_pair_viscosity
    fluid_reconstruct_immiscible_interface_normals
    fluid_reconstruct_immiscible_interface_curvature
    fluid_apply_immiscible_pair_pressure_capillarity
    fluid_emit_spatial_particles fluid_emit_spatial_source_proposals
    fluid_apply_spatial_source_admission
    fluid_filter_spatial_candidates fluid_count_spatial_neighbors
    fluid_realtime_predict fluid_realtime_density_lambda
    fluid_realtime_position_delta fluid_realtime_apply_position_delta
    fluid_realtime_viscosity_delta fluid_realtime_commit)

set(PARALLEL_MATER_SLANG_geometry_constraints_ENTRIES
    prepare_avbd_kernel solve_avbd_kernel solve_avbd_stage_kernel)
set(PARALLEL_MATER_SLANG_geometry_constraints_DEFINITIONS
    PM_GEOMETRY_CONSTRAINTS_ADAPTER=1)

set(PARALLEL_MATER_SLANG_queries_ENTRIES
    export_fluid_particles export_cloth_vertices export_surface_positions
    export_rope_nodes export_soft_body_nodes export_smoke_particles
    query_rigid_hit_box_kernel query_particle_hit_box_kernel)

set(PARALLEL_MATER_SLANG_geometry_ENTRIES
    pm_integrate_rigid_bodies
    pm_compute_rigid_world_bounds
    pm_broad_phase_rigid_pairs
    pm_emit_rigid_spatial_proxies
    pm_filter_rigid_spatial_candidates
    pm_gather_rigid_spatial_pairs
    pm_generate_rigid_leaf_pairs
    pm_evaluate_rigid_leaf_pairs
    pm_reduce_rigid_leaf_manifolds
    pm_prepare_rigid_face_manifolds
    pm_finalize_rigid_contact_manifolds
    pm_clear_rigid_inputs
    pm_capture_rigid_inputs
    pm_prepare_parallel_contact_events
    pm_load_rigid_contact_cache
    pm_save_rigid_contact_cache
    pm_gather_rigid_debug_samples
    pm_clamp_rigid_speeds
    pm_build_fixed_collision_groups)
set(PARALLEL_MATER_SLANG_geometry_OPTIMIZATION 3)

set(PARALLEL_MATER_SLANG_primitives_ENTRIES
    pm_exclusive_scan_u32
    pm_compact_count_blocks
    pm_compact_scatter_indices
    pm_radix_histogram_u64
    pm_radix_prefix_u32
    pm_radix_scatter_u64_u32
    pm_spatial_reduce_bounds_blocks
    pm_spatial_reduce_bounds_final
    pm_spatial_emit_keys
    pm_spatial_sort_small_u64_u32
    pm_spatial_radix_histogram_u64
    pm_spatial_radix_prefix_u32
    pm_spatial_radix_scatter_u64_u32
    pm_spatial_build_topology
    pm_spatial_union_bounds
    pm_spatial_refit_leaves
    pm_spatial_initialize_epoch
    pm_spatial_stage_leaves
    pm_spatial_resolve_admission
    pm_spatial_finalize_transfers
    pm_spatial_publish_epoch
    pm_spatial_query_aabbs
    pm_spatial_validate_envelopes)

set(PARALLEL_MATER_SLANG_cloth_ENTRIES
    cloth_project_volume
    cloth_realtime_predict cloth_realtime_project_links
    cloth_realtime_apply_position_delta cloth_realtime_project_volume
    cloth_realtime_project_contacts cloth_realtime_commit
    cloth_break_bonds cloth_limit_strain cloth_update_surface)
set(PARALLEL_MATER_SLANG_soft_body_ENTRIES
    deformable_predict soft_body_measure_momentum soft_body_restore_momentum
    soft_body_project_rest_shape deformable_project_links
    deformable_project_links_warp soft_body_finalize_velocities
    soft_body_apply_contact_friction soft_body_damp_springs
    soft_body_damp_springs_warp soft_body_update_surface
    soft_body_realtime_predict soft_body_realtime_project_links
    soft_body_realtime_apply_position_delta soft_body_realtime_project_shape
    soft_body_realtime_project_contacts soft_body_realtime_commit)
set(PARALLEL_MATER_SLANG_soft_body_cloth_ENTRIES
    soft_cloth_detect soft_cloth_apply_soft soft_cloth_apply_cloth)
set(PARALLEL_MATER_SLANG_cloth_rope_ENTRIES
    rope_cloth_sample_anchor rope_cloth_apply_anchor)
set(PARALLEL_MATER_SLANG_soft_body_rope_ENTRIES
    rope_soft_sample_anchor rope_soft_scatter_anchor rope_soft_apply)
set(PARALLEL_MATER_SLANG_fluid_cloth_ENTRIES
    fluid_cloth_containment_forces cloth_apply_fluid_forces
    fluid_project_inside_cloth)
set(PARALLEL_MATER_SLANG_fluid_rope_ENTRIES
    fluid_rope_contacts fluid_rope_apply)
set(PARALLEL_MATER_SLANG_fluid_smoke_ENTRIES
    fluid_smoke_drag fluid_smoke_grid_drag fluid_smoke_exchange
    fluid_clear_inactive_keep)
set(PARALLEL_MATER_SLANG_cloth_smoke_ENTRIES
    smoke_cloth_wind smoke_cloth_contact)
set(PARALLEL_MATER_SLANG_soft_body_smoke_ENTRIES
    smoke_soft_body_wind smoke_soft_body_contact)
set(PARALLEL_MATER_SLANG_rope_smoke_ENTRIES
    smoke_rope_wind smoke_rope_contact)
set(PARALLEL_MATER_SLANG_fluid_soft_body_ENTRIES
    fluid_soft_refit fluid_soft_detect fluid_soft_solve fluid_soft_apply
    fluid_soft_recover)
set(PARALLEL_MATER_SLANG_geometry_cloth_ENTRIES
    cloth_constrain_bodies cloth_apply_body_corrections)
set(PARALLEL_MATER_SLANG_geometry_fluid_ENTRIES
    fluid_static_contacts fluid_body_bounds_kernel fluid_index_body_cells
    fluid_detect_moving_contacts fluid_resolve_moving_contacts
    reduce_point_body_impulses)
set(PARALLEL_MATER_SLANG_geometry_smoke_ENTRIES
    smoke_emit_cells smoke_density_pressure smoke_pair_forces
    smoke_compute_vorticity smoke_advect smoke_emit smoke_deformable_bounds
    smoke_rigid_contact smoke_apply_rigid_impulses
    smoke_realtime_particle_emit smoke_realtime_particle_predict
    smoke_realtime_particle_density_pressure
    smoke_realtime_particle_position_delta
    smoke_realtime_particle_apply_position_delta
    smoke_realtime_particle_commit)
set(PARALLEL_MATER_SLANG_geometry_soft_body_ENTRIES
    deformable_collide soft_body_surface_contacts
    soft_body_apply_surface_contacts
    pm_emit_deformable_spatial_primitives
    pm_filter_deformable_spatial_candidates
    pm_emit_smoke_spatial_queries
    pm_filter_smoke_spatial_obstacles)
set(PARALLEL_MATER_SLANG_smoke_grid_ENTRIES
    smoke_grid_initialize_cells
    smoke_grid_initialize_faces
    smoke_grid_clear_obstacles
    smoke_grid_raster_rigid
    smoke_grid_raster_deformable
    smoke_grid_raster_soft
    smoke_grid_resolve_particle_fields
    smoke_grid_splat_particles
    smoke_grid_advect_face
    smoke_grid_correct_face
    smoke_grid_cell_diagnostics
    smoke_grid_subgrid_force
    smoke_grid_apply_face_forces
    smoke_grid_apply_face_boundaries
    smoke_grid_divergence
    smoke_grid_pressure_smooth
    smoke_grid_pressure_residual
    smoke_grid_restrict_residual
    smoke_grid_clear_scalar
    smoke_grid_restrict_open
    smoke_grid_mark_domain_boundaries
    smoke_grid_prolong_add
    smoke_grid_compare_residual
    smoke_grid_reset_residual
    smoke_grid_project_face
    smoke_grid_trace
    smoke_grid_soft_body_force
    smoke_grid_cloth_force
    smoke_grid_apply_cloth_force
    smoke_grid_rigid_force
    smoke_realtime_grid_begin
    smoke_realtime_grid_source
    smoke_realtime_grid_advect_scalar
    smoke_realtime_grid_advect_face
    smoke_realtime_grid_face_forces
    smoke_realtime_grid_boundaries
    smoke_realtime_grid_divergence
    smoke_realtime_grid_pressure_cycle
    smoke_realtime_grid_project_face
    smoke_realtime_grid_trace)
set(PARALLEL_MATER_SLANG_rope_ENTRIES
    rope_realtime_predict rope_realtime_project_constraints
    rope_realtime_apply_position_delta rope_realtime_project_contacts
    rope_realtime_commit
    rope_advance rope_advance_coupled)
set(PARALLEL_MATER_SLANG_rope_OPTIMIZATION 3)

function(parallel_mater_slang_entry_arguments module variant output)
    set(entries "${PARALLEL_MATER_SLANG_${module}_${variant}_ENTRIES}")
    if(NOT entries)
        set(entries "${PARALLEL_MATER_SLANG_${module}_ENTRIES}")
    endif()
    set(arguments)
    foreach(entry IN LISTS entries)
        list(APPEND arguments -entry "${entry}")
        if(NOT variant STREQUAL "CUDA")
            list(APPEND arguments -stage compute)
        endif()
    endforeach()
    set(${output} "${arguments}" PARENT_SCOPE)
endfunction()

function(parallel_mater_slang_definition_arguments module variant output)
    set(definitions ${PARALLEL_MATER_SLANG_${module}_DEFINITIONS}
                    ${PARALLEL_MATER_SLANG_${module}_${variant}_DEFINITIONS})
    set(arguments)
    foreach(definition IN LISTS definitions)
        list(APPEND arguments "-D${definition}")
    endforeach()
    set(${output} "${arguments}" PARENT_SCOPE)
endfunction()

function(parallel_mater_compile_slang_cuda output_variable)
    set(generated_dir "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
    file(GLOB_RECURSE slang_dependencies CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.inc"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slangh")
    set(outputs)
    foreach(module IN LISTS PARALLEL_MATER_SLANG_KERNEL_MODULES)
        parallel_mater_slang_entry_arguments("${module}" CUDA entry_arguments)
        parallel_mater_slang_definition_arguments(
            "${module}" CUDA definition_arguments)
        set(optimization "${PARALLEL_MATER_SLANG_${module}_OPTIMIZATION}")
        if(NOT optimization)
            set(optimization 2)
        endif()
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang")
        set(generated
            "${generated_dir}/parallel_mater_${module}_runtime.cu")
        add_custom_command(
            OUTPUT "${generated}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}" "${source}"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                ${definition_arguments} ${entry_arguments}
                -target cuda -fp-mode precise "-O${optimization}"
                -line-directive-mode none -o "${generated}"
            # slangc preserves the timestamp when generated text is unchanged.
            # Refresh it so a changed shared import does not rebuild forever.
            COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${generated}"
            DEPENDS ${slang_dependencies}
                    "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            COMMENT "Compiling ${module}.slang for the CUDA device adapter"
            COMMAND_EXPAND_LISTS VERBATIM)
        list(APPEND outputs "${generated}")
    endforeach()
    set_source_files_properties(${outputs}
        PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
    set(${output_variable} "${outputs}" PARENT_SCOPE)
endfunction()

function(parallel_mater_compile_slang_dxil output_variable)
    if(NOT WIN32)
        set(${output_variable} "" PARENT_SCOPE)
        set(${output_variable}_DXIL "" PARENT_SCOPE)
        set(${output_variable}_RECORDS "" PARENT_SCOPE)
        return()
    endif()
    set(generated_dir "${PARALLEL_MATER_SLANG_GENERATED_DIR}/dxil")
    file(GLOB_RECURSE slang_dependencies CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.inc"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slangh")
    set(outputs)
    set(dxil_outputs)
    set(records)
    foreach(module IN LISTS PARALLEL_MATER_SLANG_KERNEL_MODULES)
        set(entries "${PARALLEL_MATER_SLANG_${module}_ENTRIES}")
        parallel_mater_slang_definition_arguments(
            "${module}" DXIL definition_arguments)
        set(optimization "${PARALLEL_MATER_SLANG_${module}_OPTIMIZATION}")
        if(NOT optimization)
            set(optimization 2)
        endif()
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang")
        foreach(entry IN LISTS entries)
            set(generated "${generated_dir}/${module}/${entry}.dxil")
            set(reflection "${generated_dir}/${module}/${entry}.json")
            add_custom_command(
                OUTPUT "${generated}" "${reflection}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${generated_dir}/${module}"
                COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}" "${source}"
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                    ${definition_arguments}
                    -DPM_TARGET_D3D12=1 -DPM_TARGET_CUDA=0
                    "-DPM_DXIL_ENTRY_${entry}=1"
                    -entry "${entry}" -stage compute
                    -target dxil -profile sm_6_6 -fp-mode precise
                    "-O${optimization}" -warnings-as-errors all
                    -o "${generated}" -reflection-json "${reflection}"
                DEPENDS ${slang_dependencies}
                        "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
                COMMENT "Compiling ${module}.${entry} shared schedule to DXIL"
                COMMAND_EXPAND_LISTS VERBATIM)
            list(APPEND outputs "${generated}" "${reflection}")
            list(APPEND dxil_outputs "${generated}")
            list(APPEND records "${module}|${entry}|${generated}")
        endforeach()
    endforeach()
    set(${output_variable} "${outputs}" PARENT_SCOPE)
    set(${output_variable}_DXIL "${dxil_outputs}" PARENT_SCOPE)
    set(${output_variable}_RECORDS "${records}" PARENT_SCOPE)
endfunction()
