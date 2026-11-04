#version 450
// The scaled VRAM image back into the raw mirror (render-to-texture, readbacks): each VRAM word from its top-left
// sample, 5 bits a colour and the mask bit.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(set = 0, binding = 1) uniform sampler2D scaled;
layout(location = 0) out uint outWord;

void main()
{
	const vec4 c = texelFetch(scaled, ivec2(gl_FragCoord.xy) * pc.flags.z, 0);
	const uvec3 c5 = uvec3(round(c.rgb * 31.0));
	outWord = c5.r | (c5.g << 5) | (c5.b << 10) | (c.a > 0.5 ? 0x8000u : 0u);
}
