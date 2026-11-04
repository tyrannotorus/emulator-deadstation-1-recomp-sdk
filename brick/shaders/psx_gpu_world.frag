#version 450
// A world's polygons with cut-outs. Up close, one nearest fetch, its transparent texels discarded. Far off, where a pixel covers several texels and one fetch would pick a different one
// each frame (leaves that flicker as the camera moves): the texels it covers that aren't transparent, averaged; gone only if all of them are.
precision mediump float;
layout(set = 0, binding = 3) uniform mediump sampler2D atlasNearest;
layout(location = 0) in highp vec2 vAtlas;
layout(location = 1) in vec4 vMod;
layout(push_constant) uniform World
{
	layout(offset = 80) vec4 fog;
	vec4 fogRange;
} world;
layout(location = 0) out vec4 outColor;

void main()
{
	const highp vec2 dx = dFdx(vAtlas), dy = dFdy(vAtlas);
	const highp float texels = max(dot(dx, dx), dot(dy, dy)) * 4096.0 * 4096.0;
	vec3 rgb;
	bool stp;
	if (texels <= 1.5)
	{
		const vec4 t = texture(atlasNearest, vAtlas);
		stp = t.a > 0.5;
		if (!stp && t.r + t.g + t.b == 0.0)
			discard;
		rgb = t.rgb;
	}
	else if (texels <= 6.25)
	{
		// (two of what it covers, across the longer way: most of a jungle is here, and a fetch is what it costs)
		const highp vec2 a = (dot(dx, dx) > dot(dy, dy) ? dx : dy) * 0.25;
		const vec4 s0 = texture(atlasNearest, vAtlas + a), s1 = texture(atlasNearest, vAtlas - a);
		const bool on0 = s0.a > 0.5 || s0.r + s0.g + s0.b > 0.0, on1 = s1.a > 0.5 || s1.r + s1.g + s1.b > 0.0;
		if (!on0 && !on1)
			discard;
		rgb = on0 && on1 ? (s0.rgb + s1.rgb) * 0.5 : on0 ? s0.rgb : s1.rgb;
		stp = (on0 && s0.a > 0.5) || (on1 && s1.a > 0.5);
	}
	else
	{
		const highp vec2 a = (dx + dy) * 0.25, b = (dx - dy) * 0.25;
		const vec4 s[4] = vec4[4](texture(atlasNearest, vAtlas + a), texture(atlasNearest, vAtlas - a), texture(atlasNearest, vAtlas + b), texture(atlasNearest, vAtlas - b));
		vec3 sum = vec3(0.0);
		float there = 0.0;
		stp = false;
		for (int i = 0; i < 4; i++)
			if (s[i].a > 0.5 || s[i].r + s[i].g + s[i].b > 0.0)
			{
				sum += s[i].rgb;
				there += 1.0;
				stp = stp || s[i].a > 0.5;
			}
		if (there == 0.0)
			discard;
		rgb = sum / there;
	}
	outColor = vec4(mix(min(rgb * vMod.rgb, vec3(1.0)), world.fog.rgb, vMod.a), stp ? 1.0 : 0.0);
}
