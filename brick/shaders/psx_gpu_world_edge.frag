#version 450
// A world's polygons with nothing cut out of what they show, but transparent texels beside it or a speck or two in it: no discard, so hidden surface removal keeps them, and far off
// the texels a pixel covers that are not transparent, averaged, so what is beside them never darkens their edges. A speck in one is black, up close.
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
		rgb = t.rgb;
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
		rgb = sum / max(there, 1.0);
	}
	outColor = vec4(mix(min(rgb * vMod.rgb, vec3(1.0)), world.fog.rgb, vMod.a), stp ? 1.0 : 0.0);
}
