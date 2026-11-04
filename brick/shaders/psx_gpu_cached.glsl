// A textured primitive from the texture cache (psx_gpu.cpp's atlas, filled by psx_gpu_decode.frag): one fetch of its
// texel (the texture window applied) from its slot (vTex.y; 16 slots a row), all zero transparent, alpha the STP bit; then as
// psx_gpu_tex.frag does: modulated by the colour (texel * colour >> 7) unless raw, alpha the mask bit, STP pass 1 the
// opaque texels, 2 the semi-transparent ones. PSX_GPU_OPAQUE (psx_gpu_cached_opaque.frag): a page with no transparent
// texel drawn in one pass, without discard (the GPU keeps hidden-surface removal).
layout(set = 0, binding = 2) uniform sampler2D atlas;
layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;
layout(location = 2) flat in uvec4 vTex;
layout(location = 0) out vec4 outColor;

void main()
{
	int u = int(floor(vUv.x)) & 255;
	int v = int(floor(vUv.y)) & 255;
	u = (u & ~(pc.window.x * 8)) | ((pc.window.z & pc.window.x) * 8);
	v = (v & ~(pc.window.y * 8)) | ((pc.window.w & pc.window.y) * 8);
	const uint slot = vTex.y;
	const vec4 t = texelFetch(atlas, ivec2(int(slot & 15u) * 256 + u, int(slot >> 4) * 256 + v), 0);
	const bool stp = t.a > 0.5;
#ifndef PSX_GPU_OPAQUE
	if (!stp && t.r + t.g + t.b == 0.0)
		discard;
	if ((pc.flags.y == 1 && stp) || (pc.flags.y == 2 && !stp))
		discard;
#endif
	vec3 c = floor(t.rgb * 31.0 + 0.5);
	if ((vTex.w & 1u) == 0u)
		c = min(floor(c * floor(vColor.rgb * 255.0 + 0.5) / 128.0), vec3(31.0));
	outColor = vec4(c / 31.0, (stp || pc.flags.x != 0) ? 1.0 : 0.0);
}
