# A recomp module's Brick build (aarch64): included by its CMakeLists.txt after psxrecomp_add_game_runtime.
# The Vulkan presenter shows the runtime's frames (host.cpp, patch 0007); no desktop OpenGL there
# (gl_absent.c). DEADSTATION_VULKAN_INCLUDE: the folder with the Vulkan headers (tools/build-module.sh).
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    return()
endif()
if(NOT DEADSTATION_VULKAN_INCLUDE OR NOT EXISTS "${DEADSTATION_VULKAN_INCLUDE}/vulkan/vulkan.hpp")
    message(FATAL_ERROR "DEADSTATION_VULKAN_INCLUDE: the folder with the Vulkan headers (vulkan/vulkan.hpp)")
endif()
set(_ds_brick "${CMAKE_CURRENT_LIST_DIR}")
find_program(DEADSTATION_GLSLANG glslangValidator NO_CMAKE_FIND_ROOT_PATH REQUIRED)
set(_ds_spirv)
foreach(_shader frame.vert frame.frag frame_exact.frag)
    string(REPLACE "." "_" _name ${_shader})
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/deadstation_shaders/${_name}.h)
    add_custom_command(OUTPUT ${_out}
            COMMAND ${DEADSTATION_GLSLANG} -V --quiet --vn ${_name} -o ${_out} ${_ds_brick}/shaders/${_shader}
            DEPENDS ${_ds_brick}/shaders/${_shader})
    list(APPEND _ds_spirv ${_out})
endforeach()
# the 3D layer's (scene_light.cpp): its parameters' include beside them
foreach(_shader scene_shadow.vert scene_flash.vert scene_cast.vert scene_light.vert scene_light.frag scene_flash_floor.frag psx_gpu.vert psx_gpu_flat.frag psx_gpu_tex.frag
        psx_gpu_cached.frag psx_gpu_cached_opaque.frag psx_gpu_spot.vert psx_gpu_spot.frag psx_gpu_world.vert psx_gpu_world.frag psx_gpu_world_opaque.frag psx_gpu_world_lit.vert psx_gpu_world_lit.frag psx_gpu_world_edge_lit.frag psx_gpu_world_edge.frag psx_gpu_world_sky.vert psx_gpu_world_sky.frag psx_gpu_lean.vert psx_gpu_lean.frag psx_gpu_lean_opaque.frag psx_gpu_decode.frag psx_gpu_rect.vert psx_gpu_blit.frag psx_gpu_pack.frag)
    string(REPLACE "." "_" _name ${_shader})
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/deadstation_shaders/${_name}.h)
    add_custom_command(OUTPUT ${_out}
            COMMAND ${DEADSTATION_GLSLANG} -V --quiet -I${_ds_brick}/shaders --vn ${_name} -o ${_out} ${_ds_brick}/shaders/${_shader}
            DEPENDS ${_ds_brick}/shaders/${_shader} ${_ds_brick}/shaders/scene_params.glsl ${_ds_brick}/shaders/psx_gpu.glsl
                    ${_ds_brick}/shaders/psx_gpu_cached.glsl ${_ds_brick}/shaders/psx_gpu_lean.glsl)
    list(APPEND _ds_spirv ${_out})
endforeach()
# PSXRecomp's lean CPU timing (patch 0011): whole-instruction cycles and an inline branch check, for the Brick's CPU;
# its GPU drawn by Deadstation's Vulkan (psx_gpu.cpp, patch 0012: PSXRecomp's own Vulkan backend left out)
target_compile_definitions(psx-runtime PRIVATE PSX_LEAN_TIMING=1 PSX_HOST_VK_RENDERER=1)
set(_ds_sources ${_ds_brick}/host.cpp ${_ds_brick}/host_link.cpp ${_ds_brick}/psx_gpu.cpp ${_ds_brick}/scene_light.cpp ${_ds_brick}/video.cpp ${_ds_brick}/panel.cpp)
target_sources(psx-runtime PRIVATE ${_ds_sources} ${_ds_brick}/gl_absent.c ${_ds_spirv})
set_source_files_properties(${_ds_sources} PROPERTIES
        INCLUDE_DIRECTORIES "${_ds_brick};${DEADSTATION_VULKAN_INCLUDE};${CMAKE_CURRENT_BINARY_DIR}/deadstation_shaders;${_ds_brick}/../psxrecomp/runtime/include")
# Deadstation starts a module as <exec> <disc>: host_link.cpp makes PSXRecomp's arguments
target_link_options(psx-runtime PRIVATE -Wl,--wrap=main)
