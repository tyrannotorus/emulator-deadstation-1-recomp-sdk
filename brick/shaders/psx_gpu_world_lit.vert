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
	vec4 lights;	// x, y: how strong each of two beams at the corners, 0 for none; z the near beam's, at the pixels (here its axes are lightA)
	vec4 lightA[3], lightB[3];	// world to each beam's axes: along, across, height
} world;
layout(location = 0) out vec2 vAtlas;
layout(location = 2) out mediump vec4 vBeam;	// the near beam, for each pixel: across, its width there, the lamp's offset there, how bright
layout(location = 1) out mediump vec4 vMod;		// the corner's colour, and how much of the fog is in front of it
invariant gl_Position;

// A pair of headlights at this corner, as psx_gpu_world_spot.frag has them for each pixel: for what's small and many (fronds, spectators) or far from the camera's interest (the other car's)
// there's no second drawing of the polygons for it. What the pixel's version costs a frame in a jungle, this costs next to nothing. A big polygon has few corners for a narrow beam to fall on, one
// lit and the others not making a hard bright triangle of it: so the beam's blurred by the polygon's own size (blur, half its longest edge) and dimmed to keep its light the same.
float beamAt(const vec4 along, const vec4 across, const vec4 height, const vec4 p, const float blur)
{
	const float a = dot(along, p), c = abs(dot(across, p));
	const float left = max(1.0 - a / (2000.0 + blur), 0.0);
	const float sharp = 70.0 + max(a, 0.0) * 0.42, width = sharp + blur;
	if (left <= 0.0 || a <= -40.0 - blur || c >= width * 1.5)
		return 0.0;
	const float start = smoothstep(-40.0 - blur, 300.0 + blur, a);
	const float reach = left * sqrt(left);
	const float lamp = 50.0 * max(1.0 - a / 500.0, 0.0);
	const float shape = 1.0 - smoothstep(width * 0.4, width, abs(c - lamp));
	const float spill = 0.24 * (1.0 - smoothstep(0.0, width * 1.5, c));
	return (shape + spill) * start * reach * clamp(1.0 - abs(dot(height, p)) / (450.0 + blur), 0.0, 1.0) * 1.9 * sharp / width;
}

void main()
{
	const vec4 p = vec4(vec3(position.xyz), 1.0);
	const vec3 v = vec3(dot(world.row[0], p), dot(world.row[1], p), dot(world.row[2], p));
	const float far = v.z + float(position.w);
	gl_Position = vec4(v.x * world.project.x + v.z * world.project.z, v.y * world.project.y + v.z * world.project.w, far > 1.0 ? 16.0 * v.z / far : 16.0, v.z);
	vAtlas = (vec2(float(tex.w & 15u), float(tex.w >> 4)) * 256.0 + vec2(tex.xy)) / 4096.0;
	vec3 light = ((tex.z & 1u) != 0u ? vec3(1.0) : color.rgb * (255.0 / 128.0)) * world.tint.rgb;
	{
		const float a = dot(world.lightA[0], p), left = max(1.0 - a / 2000.0, 0.0);
		vBeam = vec4(dot(world.lightA[1], p), 70.0 + max(a, 0.0) * 0.42, 50.0 * max(1.0 - a / 500.0, 0.0),
				smoothstep(-40.0, 300.0, a) * left * sqrt(left) * clamp(1.0 - abs(dot(world.lightA[2], p)) / 450.0, 0.0, 1.0) * world.lights.z * 1.9);
	}
	const float blur = color.a * 255.0 * 8.0;
	if (world.lights.x > 0.0)
		light += vec3(1.0, 0.94, 0.78) * world.lights.x * beamAt(world.lightA[0], world.lightA[1], world.lightA[2], p, blur);
	if (world.lights.y > 0.0)
		light += vec3(1.0, 0.94, 0.78) * world.lights.y * beamAt(world.lightB[0], world.lightB[1], world.lightB[2], p, blur);
	const float fogged = world.fog.a * sqrt(clamp((v.z - world.fogRange.x) * world.fogRange.y, 0.0, 1.0));
	// (what's added to the frame goes dim in fog; given the fog's colour it would be added white)
	vMod = world.fogRange.z != 0.0 ? vec4(light * (1.0 - fogged), 0.0) : vec4(light, fogged);
}
