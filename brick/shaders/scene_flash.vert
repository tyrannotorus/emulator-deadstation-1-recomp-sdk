#version 450
#extension GL_GOOGLE_include_directive : require
// The models from the brightest flash: their depth in its view, into its shadow map.
#include "scene_params.glsl"
layout(location = 1) in vec3 pos;

void main()
{
	gl_Position = flashClip(pos);
}
