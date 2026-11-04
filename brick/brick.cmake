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
foreach(_shader psx_gpu.vert psx_gpu_flat.frag psx_gpu_tex.frag
        psx_gpu_cached.frag psx_gpu_cached_opaque.frag psx_gpu_world.vert psx_gpu_world.frag psx_gpu_world_opaque.frag psx_gpu_world_edge.frag psx_gpu_world_sky.vert psx_gpu_world_sky.frag psx_gpu_lean.vert psx_gpu_lean.frag psx_gpu_lean_opaque.frag psx_gpu_decode.frag psx_gpu_rect.vert psx_gpu_blit.frag psx_gpu_pack.frag)
    string(REPLACE "." "_" _name ${_shader})
    set(_out ${CMAKE_CURRENT_BINARY_DIR}/deadstation_shaders/${_name}.h)
    add_custom_command(OUTPUT ${_out}
            COMMAND ${DEADSTATION_GLSLANG} -V --quiet -I${_ds_brick}/shaders --vn ${_name} -o ${_out} ${_ds_brick}/shaders/${_shader}
            DEPENDS ${_ds_brick}/shaders/${_shader} ${_ds_brick}/shaders/psx_gpu.glsl
                    ${_ds_brick}/shaders/psx_gpu_cached.glsl ${_ds_brick}/shaders/psx_gpu_lean.glsl)
    list(APPEND _ds_spirv ${_out})
endforeach()
# PSXRecomp's lean CPU timing (patch 0011): whole-instruction cycles and an inline branch check, for the Brick's CPU;
# its GPU drawn by Deadstation's Vulkan (psx_gpu.cpp, patch 0012: PSXRecomp's own Vulkan backend left out)
target_compile_definitions(psx-runtime PRIVATE PSX_LEAN_TIMING=1 PSX_HOST_VK_RENDERER=1)
set(_ds_sources ${_ds_brick}/host.cpp ${_ds_brick}/host_link.cpp ${_ds_brick}/psx_gpu.cpp ${_ds_brick}/video.cpp ${_ds_brick}/panel.cpp)
target_sources(psx-runtime PRIVATE ${_ds_sources} ${_ds_brick}/gl_absent.c ${_ds_spirv})
set_source_files_properties(${_ds_sources} PROPERTIES
        INCLUDE_DIRECTORIES "${_ds_brick};${DEADSTATION_VULKAN_INCLUDE};${CMAKE_CURRENT_BINARY_DIR}/deadstation_shaders;${_ds_brick}/../psxrecomp/runtime/include")
# Deadstation starts a module as <exec> <disc>: host_link.cpp makes PSXRecomp's arguments
target_link_options(psx-runtime PRIVATE -Wl,--wrap=main)

# A module's own Vulkan code and shaders (deadstation_module_layer, deadstation_world_shaders), in its CMakeLists.txt after this file. On another machine than the handheld's this file
# ends above, so `if(COMMAND ...)` round them.
#   deadstation_module_shaders(DIR FILE...): each file in DIR into <file_name_with_underscores>.h, a uint32_t array of that name, for any source of the module to include; DIR on its include path
#   deadstation_module_sources(FILE...): compiled with the Vulkan headers and this folder on their include paths
function(deadstation_module_shaders _dir)
    set(_outs)
    foreach(_shader ${ARGN})
        string(REPLACE "." "_" _name ${_shader})
        set(_out ${CMAKE_BINARY_DIR}/deadstation_shaders/${_name}.h)
        file(GLOB _includes ${_dir}/*.glsl)
        add_custom_command(OUTPUT ${_out}
                COMMAND ${DEADSTATION_GLSLANG} -V --quiet -I${_dir} -I${_ds_brick}/shaders --vn ${_name} -o ${_out} ${_dir}/${_shader}
                DEPENDS ${_dir}/${_shader} ${_includes})
        list(APPEND _outs ${_out})
    endforeach()
    target_sources(psx-runtime PRIVATE ${_outs})
    target_include_directories(psx-runtime PRIVATE ${CMAKE_BINARY_DIR}/deadstation_shaders)
endfunction()
function(deadstation_module_sources)
    target_sources(psx-runtime PRIVATE ${ARGN})
    set_source_files_properties(${ARGN} PROPERTIES
            INCLUDE_DIRECTORIES "${_ds_brick};${DEADSTATION_VULKAN_INCLUDE};${CMAKE_BINARY_DIR}/deadstation_shaders;${_ds_brick}/../psxrecomp/runtime/include")
endfunction()
