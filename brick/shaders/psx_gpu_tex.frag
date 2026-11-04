#version 450
// A textured primitive: the texel from the raw VRAM mirror (4- and 8-bit through the CLUT, 15-bit direct, the texture
// window applied), 0x0000 transparent, modulated by the colour (texel * colour >> 7) unless raw; alpha is the mask bit
// (the texel's STP bit or the set-mask state). STP pass 1 keeps the opaque texels, 2 the semi-transparent ones.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(set = 0, binding = 0) uniform usampler2D raw;
layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;
layout(location = 2) flat in uvec4 vTex;
layout(location = 0) out vec4 outColor;

uint word(int x, int y)
{
	return texelFetch(raw, ivec2(x & 1023, y & 511), 0).r;
}

void main()
{
	int u = int(floor(vUv.x)) & 255;
	int v = int(floor(vUv.y)) & 255;
	u = (u & ~(pc.window.x * 8)) | ((pc.window.z & pc.window.x) * 8);
	v = (v & ~(pc.window.y * 8)) | ((pc.window.w & pc.window.y) * 8);
	const uint page = vTex.x;
	const int px = int(page & 15u) * 64, py = int((page >> 4) & 1u) * 256;
	const uint depth = (page >> 7) & 3u;
	uint texel;
	if (depth == 0u)
	{
		const uint w = word(px + (u >> 2), py + v);
		texel = word(int(vTex.y) + int((w >> ((u & 3) * 4)) & 15u), int(vTex.z));
	}
	else if (depth == 1u)
	{
		const uint w = word(px + (u >> 1), py + v);
		texel = word(int(vTex.y) + int((w >> ((u & 1) * 8)) & 255u), int(vTex.z));
	}
	else
		texel = word(px + u, py + v);
	if (texel == 0u)
		discard;
	const uint stp = texel >> 15;
	if ((pc.flags.y == 1 && stp != 0u) || (pc.flags.y == 2 && stp == 0u))
		discard;
	uvec3 c = uvec3(texel & 31u, (texel >> 5) & 31u, (texel >> 10) & 31u);
	if ((vTex.w & 1u) == 0u)
		c = min((c * uvec3(round(vColor.rgb * 255.0))) >> 7, uvec3(31u));
	outColor = vec4(vec3(c) / 31.0, float(stp | uint(pc.flags.x)));
}
