#version 450
// A cached-texture primitive, lean, no transparent texel (psx_gpu_lean.glsl)
#extension GL_GOOGLE_include_directive : require
#define PSX_GPU_OPAQUE 1
#include "psx_gpu.glsl"
#include "psx_gpu_lean.glsl"
