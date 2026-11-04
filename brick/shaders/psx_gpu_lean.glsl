// The lean cached-texture fragment (psx_gpu_lean.vert): the texel by one normalized nearest fetch (all zero transparent,
// alpha the STP bit) times the vertex shader's modulation (the PlayStation's texel * colour >> 7 without its rounding to
// 5 bits), alpha the mask bit; STP pass 1 the opaque texels, 2 the semi-transparent ones. PSX_GPU_OPAQUE: a page with no
// transparent texel, one pass, no discard.
layout(set = 0, binding = 3) uniform sampler2D atlasNearest;
layout(location = 0) in vec2 vAtlas;
layout(location = 1) in vec3 vMod;
layout(location = 2) in float vFog;
layout(location = 0) out vec4 outColor;

void main()
{
	const vec4 t = texture(atlasNearest, vAtlas);
	const bool stp = t.a > 0.5;
#ifndef PSX_GPU_OPAQUE
	if (!stp && t.r + t.g + t.b == 0.0)
		discard;
	if ((pc.flags.y == 1 && stp) || (pc.flags.y == 2 && !stp))
		discard;
#endif
	outColor = vec4(mix(min(t.rgb * vMod, vec3(1.0)), pc.rect.rgb, vFog), (stp || pc.flags.x != 0) ? 1.0 : 0.0);
}
