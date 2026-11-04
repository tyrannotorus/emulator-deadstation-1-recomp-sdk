#version 450
// A cached-texture primitive with no texture window and its texels within its page (psx_gpu.cpp's lean path): the
// atlas coordinate made here once a corner (its slot, vTex.y, 16 a row, plus the page's u, v), and the modulation as a
// factor (texel * colour / 128; 1 when raw), so the fragment shader takes one normalized fetch and one multiply.
#extension GL_GOOGLE_include_directive : require
#include "psx_gpu.glsl"
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 3) in uvec4 tex;
layout(location = 4) in float depth;
layout(location = 5) in float w;		// 3D in a scene: its view depth (textures and colours interpolate in perspective); else 1
layout(location = 0) out vec2 vAtlas;
layout(location = 1) out vec3 vMod;
layout(location = 2) out float vFog;	// how much of a plugin's fog (colour in pc.rect, when its a is 1): the corner's alpha

void main()
{
	gl_Position = vec4((pos.x / 512.0 - 1.0) * w, (pos.y / 256.0 - 1.0) * w, depth * w, w);
	const vec2 origin = vec2(float(tex.y & 15u), float(tex.y >> 4)) * 256.0;
	vAtlas = (origin + uv) / vec2(4096.0, 4096.0);
	vMod = (tex.w & 1u) != 0u ? vec3(1.0) : color.rgb * (255.0 / 128.0);
	vFog = color.a * pc.rect.a;
}
