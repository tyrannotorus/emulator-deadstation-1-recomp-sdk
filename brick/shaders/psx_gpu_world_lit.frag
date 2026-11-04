#version 450
// A world's solid polygons: the PlayStation's own sharp texels up close. Farther off, where a pixel covers more than a texel and one nearest fetch would pick a different one each frame
// (a shimmer, whole polygons seeming to flash as the camera moves), one filtered fetch, the four nearest in one; past that, four of those. There's no mip chain: the atlas's textures sit
// edge to edge in their pages.
// Colour in mediump: this GPU's 16-bit units do it in fewer cycles, and there are few enough of those for a frame at 1024 x 768.
precision mediump float;
layout(set = 0, binding = 3) uniform mediump sampler2D atlasNearest;
layout(set = 0, binding = 4) uniform mediump sampler2D atlasFiltered;
layout(location = 0) in highp vec2 vAtlas;
layout(location = 1) in vec4 vMod;
layout(location = 2) in vec4 vBeam;
layout(push_constant) uniform World
{
	layout(offset = 80) vec4 fog;
	vec4 fogRange;
} world;
layout(location = 0) out vec4 outColor;

// The near beam on this pixel, in the one drawing of it: what it doesn't reach costs a compare.
vec3 beamHere()
{
	const float across = abs(vBeam.x), width = vBeam.y;
	if (vBeam.w <= 0.004 || across >= width * 1.5)
		return vec3(0.0);
	const float shape = 1.0 - smoothstep(width * 0.4, width, abs(across - vBeam.z));
	const float spill = 0.24 * (1.0 - smoothstep(0.0, width * 1.5, across));
	return vec3(1.0, 0.94, 0.78) * (shape + spill) * vBeam.w;
}

void main()
{
	const highp vec2 dx = dFdx(vAtlas), dy = dFdy(vAtlas);
	const highp float texels = max(dot(dx, dx), dot(dy, dy)) * 4096.0 * 4096.0;
	vec4 t;
	if (texels <= 1.5)
		t = texture(atlasNearest, vAtlas);
	else if (texels <= 6.25)
		t = texture(atlasFiltered, vAtlas);
	else
	{
		const highp vec2 a = (dx + dy) * 0.25, b = (dx - dy) * 0.25;
		t = (texture(atlasFiltered, vAtlas + a) + texture(atlasFiltered, vAtlas - a) + texture(atlasFiltered, vAtlas + b) + texture(atlasFiltered, vAtlas - b)) * 0.25;
	}
	outColor = vec4(mix(min(t.rgb * (vMod.rgb + beamHere()), vec3(1.0)), world.fog.rgb, vMod.a), t.a > 0.5 ? 1.0 : 0.0);
}
