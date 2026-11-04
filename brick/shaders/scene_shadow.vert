#version 450
#extension GL_GOOGLE_include_directive : require
// The models from the sun: their depth along its light, into the shadow map.
#include "scene_params.glsl"
layout(location = 1) in vec3 pos;

void main()
{
	vec3 s = sunSpace(pos);
	gl_Position = vec4(s.xy * 2.0 - 1.0, s.z, 1.0);
}
