#version 450
// A rect (push constant, NDC) as a strip of 4: blits into VRAM and packs out of it.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"

void main()
{
	const vec2 t = vec2(gl_VertexIndex & 1, (gl_VertexIndex >> 1) & 1);
	gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, t), 0.0, 1.0);
}
