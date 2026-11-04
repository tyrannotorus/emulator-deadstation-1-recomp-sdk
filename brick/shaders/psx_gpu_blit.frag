#version 450
// The raw mirror into the scaled VRAM image (a CPU upload): each target pixel takes its VRAM word; alpha is its mask bit.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(set = 0, binding = 0) uniform usampler2D raw;
layout(location = 0) out vec4 outColor;

void main()
{
	const uint w = texelFetch(raw, ivec2(gl_FragCoord.xy) / pc.flags.z, 0).r;
	outColor = vec4(vec3(uvec3(w & 31u, (w >> 5) & 31u, (w >> 10) & 31u)) / 31.0, float((w >> 15) | uint(pc.flags.x)));
}
