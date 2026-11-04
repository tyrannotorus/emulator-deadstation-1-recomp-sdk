#version 450
#extension GL_GOOGLE_include_directive : require
// The models (or the floor round the flashes) where the game drew them on its screen (which the frame fills), at
// their view depth, so the varyings interpolate in perspective.
#include "scene_params.glsl"
layout(location = 0) in vec2 screenPos;
layout(location = 1) in vec3 pos;
layout(location = 2) in vec3 normal;
layout(location = 0) out vec3 sun;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vPos;

void main()
{
	vec2 ndc = screenPos / params.screen.zw * 2.0 - 1.0;
	float z = max(pos.z, 1.0);
	gl_Position = vec4(ndc * z, z / params.light.z * z, z);
	sun = sunSpace(pos);
	vNormal = normal;
	vPos = pos;
}
