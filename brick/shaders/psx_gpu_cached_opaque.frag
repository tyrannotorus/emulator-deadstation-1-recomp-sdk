#version 450
// A textured primitive from the texture cache whose page has no transparent texel, in one pass (psx_gpu_cached.glsl)
#extension GL_GOOGLE_include_directive : require
#define PSX_GPU_OPAQUE 1
#include "psx_gpu.glsl"
#include "psx_gpu_cached.glsl"
