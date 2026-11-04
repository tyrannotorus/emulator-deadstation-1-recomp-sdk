#version 450
// A world as a mesh (replay.h, deadstation_world_*): its polygons stay in a buffer of their own, in the world's own units, and the camera moves them here. The GPU clips what
// crosses the near plane. Depth is reversed: near / the view's z (with a polygon's layer, as a farther z, for depth only).
layout(location = 0) in ivec4 position;	// x, y, z, how much farther to test it
layout(location = 1) in uvec4 tex;		// u, v, bit 0 raw, its slot in the atlas
layout(location = 2) in vec4 color;
layout(push_constant) uniform World
{
	vec4 row[3];	// world to view
	vec4 project;	// clip x = view x * x + view z * z, clip y = view y * y + view z * w
	vec4 tint;		// the race's light
	vec4 fog;		// its colour, and the most of it there is
	vec4 fogRange;	// x where it starts, y one over how far it goes
} world;
layout(location = 0) out vec2 vAtlas;
layout(location = 1) out mediump vec4 vMod;		// the corner's colour, and how much of the fog is in front of it
invariant gl_Position;

void main()
{
	const vec4 p = vec4(vec3(position.xyz), 1.0);
	const vec3 v = vec3(dot(world.row[0], p), dot(world.row[1], p), dot(world.row[2], p));
	const float far = v.z + float(position.w);
	gl_Position = vec4(v.x * world.project.x + v.z * world.project.z, v.y * world.project.y + v.z * world.project.w, far > 1.0 ? 16.0 * v.z / far : 16.0, v.z);
	vAtlas = (vec2(float(tex.w & 15u), float(tex.w >> 4)) * 256.0 + vec2(tex.xy)) / 4096.0;
	const vec3 light = ((tex.z & 1u) != 0u ? vec3(1.0) : color.rgb * (255.0 / 128.0)) * world.tint.rgb;
	const float fogged = world.fog.a * sqrt(clamp((v.z - world.fogRange.x) * world.fogRange.y, 0.0, 1.0));
	// (what's added to the frame goes dim in fog; given the fog's colour it would be added white)
	vMod = world.fogRange.z != 0.0 ? vec4(light * (1.0 - fogged), 0.0) : vec4(light, fogged);
}
