#version 450
// A texture page decoded into a slot of the texture cache (psx_gpu.cpp's atlas, 16 x 8 slots of 256 x 256): each
// texel's 15-bit colour from the raw VRAM mirror (4- and 8-bit through the CLUT, 15-bit direct), the slot's origin at
// pc.flags.zw. Alpha: the STP bit; 0x0000 (transparent) all zero (the atlas may be A1R5G5B5: 15-bit colour and the bit,
// exactly). pc.window: the page's GPU bits
// (x, y, depth), the CLUT's x and y.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(set = 0, binding = 0) uniform usampler2D raw;
layout(location = 0) out vec4 outColor;

uint word(int x, int y)
{
	return texelFetch(raw, ivec2(x & 1023, y & 511), 0).r;
}

void main()
{
	const ivec2 at = ivec2(gl_FragCoord.xy) - pc.flags.zw;
	const int u = at.x & 255, v = at.y & 255;
	const uint page = uint(pc.window.x);
	const int px = int(page & 15u) * 64, py = int((page >> 4) & 1u) * 256;
	const uint depth = (page >> 7) & 3u;
	uint texel;
	if (depth == 0u)
	{
		const uint w = word(px + (u >> 2), py + v);
		texel = word(pc.window.y + int((w >> ((u & 3) * 4)) & 15u), pc.window.z);
	}
	else if (depth == 1u)
	{
		const uint w = word(px + (u >> 1), py + v);
		texel = word(pc.window.y + int((w >> ((u & 1) * 8)) & 255u), pc.window.z);
	}
	else
		texel = word(px + u, py + v);
	const vec3 c = vec3(uvec3(texel & 31u, (texel >> 5) & 31u, (texel >> 10) & 31u)) / 31.0;
	outColor = texel == 0u ? vec4(0.0) : vec4(c, (texel >> 15) != 0u ? 1.0 : 0.0);
}
