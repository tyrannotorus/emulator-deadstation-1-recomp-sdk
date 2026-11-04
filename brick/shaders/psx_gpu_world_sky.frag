#version 450
// A world's sky: a gradient, or its panels' texels (the far hills), their cut-outs leaving the gradient behind them
precision mediump float;
layout(set = 0, binding = 3) uniform mediump sampler2D atlasNearest;
layout(location = 0) in highp vec2 vAtlas;
layout(location = 1) in vec3 vColor;
layout(location = 2) flat in float vFlat;
layout(push_constant) uniform World
{
	layout(offset = 80) vec4 fog;	// its colour and the most of it there is: all of that, in a sky as far as can be
} world;
layout(location = 0) out vec4 outColor;

void main()
{
	if (vFlat > 0.5)
	{
		outColor = vec4(mix(vColor, world.fog.rgb, world.fog.a), 0.0);
		return;
	}
	const vec4 t = texture(atlasNearest, vAtlas);
	if (t.a < 0.5 && t.r + t.g + t.b == 0.0)
		discard;
	outColor = vec4(mix(min(t.rgb * vColor, vec3(1.0)), world.fog.rgb, world.fog.a), t.a > 0.5 ? 1.0 : 0.0);
}
