#version 450
#extension GL_GOOGLE_include_directive : require
// One flash's light on the floor, added, except where the stencil says the models block it (scene_cast.vert): each flash its own
// shadows, as many as there are flashes. As the models' light in scene_light.frag: falling off with the square of the
// distance to its reach, taken against the light the floor had (as if half lit).
#include "scene_params.glsl"
layout(set = 0, binding = 0) uniform sampler2D frame;
layout(push_constant) uniform Pass { int index; } pass;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vPos;
layout(location = 0) out vec4 color;

const float Metre = 1000.0;

void main()
{
	vec2 uv = params.frameAt.xy + gl_FragCoord.xy / params.screen.xy * params.frameUv.xy;
	vec3 c = texture(frame, clamp(uv, params.frameAt.zw, params.frameUv.zw)).rgb;
	vec3 l = params.lightAt[pass.index].xyz - vPos;
	float d = length(l);
	float nl = max(dot(normalize(vNormal), l / max(d, 1.0)), 0.0);
	float r = max(d, 1.0) / Metre;
	float reach = params.lightAt[pass.index].w;
	float window = max(1.0 - (d / reach) * (d / reach), 0.0);
	color = vec4(c * params.lightColour[pass.index].rgb * nl * min(3.0, 1.0 / (r * r)) * window * window / (params.light.x + 0.5), 1.0);
}
