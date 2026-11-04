#version 450
// An untextured primitive: its colour; alpha is the mask bit.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(location = 1) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main()
{
	outColor = vec4(vColor.rgb, float(pc.flags.x));
}
