#version 450
// A world's sky as a mesh: the camera's turn and not its place, so it's infinitely far. Clipped at the near plane by the GPU. Its depth next to nothing: drawn after what's solid, it costs
// only the pixels nothing's in front of.
layout(location = 0) in ivec4 position;
layout(location = 1) in uvec4 tex;		// u, v, bit 0 raw, bit 1 no texture; its slot in the atlas
layout(location = 2) in vec4 color;
layout(push_constant) uniform World
{
	vec4 row[3];
	vec4 project;
} world;
layout(location = 0) out vec2 vAtlas;
layout(location = 1) out mediump vec3 vColor;
layout(location = 2) flat out mediump float vFlat;

void main()
{
	const vec3 p = vec3(position.xyz);
	const vec3 v = vec3(dot(world.row[0].xyz, p), dot(world.row[1].xyz, p), dot(world.row[2].xyz, p));
	gl_Position = vec4(v.x * world.project.x + v.z * world.project.z, v.y * world.project.y + v.z * world.project.w, (v.z - 16.0) * 1.0e-7, v.z);
	vAtlas = (vec2(float(tex.w & 15u), float(tex.w >> 4)) * 256.0 + vec2(tex.xy)) / 4096.0;
	vFlat = (tex.z & 2u) != 0u ? 1.0 : 0.0;
	vColor = (tex.z & 2u) != 0u ? color.rgb : (tex.z & 1u) != 0u ? vec3(1.0) : color.rgb * (255.0 / 128.0);
}
