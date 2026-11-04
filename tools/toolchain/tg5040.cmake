# Cross-build for NextUI's tg5040 handhelds (TrimUI Brick: aarch64 Cortex-A53) with the host's clang and lld,
# against the sysroot and GCC 8.3 runtime of LoveRetro's tg5040 toolchain.
#   cmake -S . -B <build> -DCMAKE_TOOLCHAIN_FILE=<this file> -DTG5040_ROOT=<root>
# TG5040_ROOT: the folder holding aarch64-nextui-linux-gnu/ (toolchain.sh).
# TG5040_GC_SECTIONS=ON: the link drops what nothing calls.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
if(NOT TG5040_ROOT AND DEFINED ENV{TG5040_ROOT})
	set(TG5040_ROOT $ENV{TG5040_ROOT})
endif()
if(NOT EXISTS "${TG5040_ROOT}/aarch64-nextui-linux-gnu/libc")
	message(FATAL_ERROR "tg5040: TG5040_ROOT (${TG5040_ROOT}) is not the toolchain's root")
endif()
# try_compile projects get the toolchain file again; pass the root on to them.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES TG5040_ROOT TG5040_GC_SECTIONS)
set(CMAKE_SYSROOT "${TG5040_ROOT}/aarch64-nextui-linux-gnu/libc")
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-nextui-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-nextui-linux-gnu)
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_ASM_COMPILER_TARGET aarch64-nextui-linux-gnu)
set(_tg5040_flags "--gcc-toolchain=${TG5040_ROOT} -mcpu=cortex-a53")
set(_tg5040_compile "${_tg5040_flags}")
set(_tg5040_link "${_tg5040_flags} -fuse-ld=lld -static-libstdc++ -static-libgcc")
if(TG5040_GC_SECTIONS)
	set(_tg5040_compile "${_tg5040_compile} -ffunction-sections -fdata-sections")
	set(_tg5040_link "${_tg5040_link} -Wl,--gc-sections")
endif()
set(CMAKE_C_FLAGS_INIT "${_tg5040_compile}")
set(CMAKE_CXX_FLAGS_INIT "${_tg5040_compile}")
# The device's libstdc++ is older than the toolchain's headers may assume: link the C++ runtime in.
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_tg5040_link}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_tg5040_flags} -fuse-ld=lld -static-libstdc++ -static-libgcc")
# GCC 8's std::filesystem lives in a library of its own, and glibc 2.28's threads too (before 2.34).
set(CMAKE_CXX_STANDARD_LIBRARIES "-lstdc++fs -lpthread")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
