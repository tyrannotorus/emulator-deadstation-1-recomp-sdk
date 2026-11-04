#version 450
#extension GL_GOOGLE_include_directive : require
// A pixel of the frame relit. A model's, and the floor's: where the
// sun's light is blocked (its shadow map, 2 x 2 samples), the share of the pixel's light that came from the sun
// (its normal against the sun, against what the shade keeps) taken away. A model's (the floor's: scene_flash_floor.frag): the flashes' light added (in their colours, by the normal, falling off
// with the square of the distance to 3 m), the brightest blocked by the models (its shadow map).
#include "scene_params.glsl"
layout(set = 0, binding = 0) uniform sampler2D frame;
layout(set = 1, binding = 0) uniform sampler2DShadow shadowMap;	// compared: lit where the pixel is at most the map's depth
layout(set = 1, binding = 2) uniform sampler2DShadow flashMap;
layout(push_constant) uniform Pass { int floorPass; } pass;
layout(location = 0) in vec3 sun;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vPos;
layout(location = 0) out vec4 color;

const float Metre = 1000.0, FlashBias = 30.0;

// lit by the flash unless a model is nearer it (the map's perspective depth; the pixel's distance less the bias taken
// into that depth, compared in the sampler, 2 x 2 filtered)
float flashLit(vec3 p)
{
	vec4 c = flashClip(p);
	float n = params.flashUp.w, f = params.flashAhead.w;
	float x = c.w - FlashBias;
	if (x <= n)
		return 1.0;
	vec2 uv = c.xy / c.w * 0.5 + 0.5;
	float ref = f * (1.0 - n / x) / (f - n);
	return texture(flashMap, vec3(uv, min(ref, 1.0)));
}

void main()
{
	vec2 uv = params.frameAt.xy + gl_FragCoord.xy / params.screen.xy * params.frameUv.xy;
	vec3 c = texture(frame, clamp(uv, params.frameAt.zw, params.frameUv.zw)).rgb;
	vec3 n = normalize(vNormal);
	float ndl = max(dot(n, params.axisS.xyz), 0.0);
	float sunShade = 1.0;
	if ((pass.floorPass != 0 || params.options.x > 0.0) && ndl > 0.0)
	{
		float blocked = 1.0 - texture(shadowMap, vec3(sun.xy, min(sun.z - params.light.w, 1.0)));
		sunShade = mix(1.0, params.light.x / (params.light.x + ndl), blocked * params.light.y);
	}
	// the light the pixel had, in the sun's units (the floor: as if half lit)
	float had = params.light.x + (pass.floorPass == 0 ? ndl : 0.5);
	vec3 add = vec3(0.0);
	int count = pass.floorPass == 0 ? int(params.flashAt.w) : 0;
	for (int i = 0; i < 6; i++)
	{
		if (i >= count)
			break;
		float reach = params.lightAt[i].w;
		vec3 l = params.lightAt[i].xyz - vPos;
		float d = length(l);
		float nl = max(dot(n, l / max(d, 1.0)), 0.0);
		if (nl <= 0.0 || d >= reach)
			continue;
		float r = max(d, 1.0) / Metre;
		float window = 1.0 - (d / reach) * (d / reach);
		float lit = i == 0 ? flashLit(vPos) : 1.0;
		add += params.lightColour[i].rgb * nl * min(3.0, 1.0 / (r * r)) * window * window * lit;
	}
	color = vec4(min(c * sunShade + c * add / had, vec3(1.0)), 1.0);
}
