#version 450
// A pair of headlights on the world, added: the surface's own texel times the light, per pixel, so the ground shows through it. vBeam: red
// how far along the beam (-400 to 2800 world units), green how far across it (-1600 to 1600), blue its strength here. The two pools start at
// the lamps and run together as they spread, with a low spill round them so the track's edges can be read. Its shape is the plugin's too
// (the module's plugin), for what it lights there.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(set = 0, binding = 3) uniform sampler2D atlasNearest;
layout(location = 0) in vec2 vAtlas;
layout(location = 1) in vec3 vBeam;
layout(location = 0) out vec4 outColor;

void main()
{
	const vec4 t = texture(atlasNearest, vAtlas);
	if (t.a < 0.5 && t.r + t.g + t.b == 0.0)
		discard;
	const float along = vBeam.r * 3200.0 - 400.0, across = abs(vBeam.g * 3200.0 - 1600.0);
	const float start = smoothstep(40.0, 160.0, along);
	const float left = max(1.0 - along / 2400.0, 0.0);
	const float reach = left * sqrt(left);
	const float width = 70.0 + max(along, 0.0) * 0.42;
	const float lamp = 50.0 * max(1.0 - along / 500.0, 0.0);
	const float shape = 1.0 - smoothstep(width * 0.4, width, abs(across - lamp));
	const float spill = 0.24 * (1.0 - smoothstep(0.0, width * 2.1, across));
	const vec3 light = vec3(1.0, 0.94, 0.78) * (shape + spill) * start * reach * vBeam.b * 1.9;
	outColor = vec4(min(t.rgb * light, vec3(1.0)), pc.flags.x != 0 ? 1.0 : 0.0);
}
