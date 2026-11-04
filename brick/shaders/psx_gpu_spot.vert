#version 450
// A plugin's beam over polygons already drawn (replay.h, deadstation_triangles with DeadstationBeam): the same place and texel as the
// lean path's (psx_gpu_lean.vert), its colour being where the corner lies in the beam, for the fragment shader to shape.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 3) in uvec4 tex;
layout(location = 4) in float depth;
layout(location = 5) in float w;
layout(location = 0) out vec2 vAtlas;
layout(location = 1) out vec3 vBeam;

void main()
{
	gl_Position = vec4((pos.x / 512.0 - 1.0) * w, (pos.y / 256.0 - 1.0) * w, depth * w, w);
	const vec2 origin = vec2(float(tex.y & 15u), float(tex.y >> 4)) * 256.0;
	vAtlas = (origin + uv) / vec2(4096.0, 4096.0);
	vBeam = color.rgb;
}
