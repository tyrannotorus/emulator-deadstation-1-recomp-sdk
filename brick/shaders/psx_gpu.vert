#version 450
// A primitive's corner: VRAM pixels (1024 x 512 over the whole target), its texture coordinate, colour, texture page
// (x: the page bits, y, z: the CLUT, w: bit 0 the raw texture flag).
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 3) in uvec4 tex;
layout(location = 4) in float depth;	// the primitive's place in its pass (psx_gpu.cpp's order depth): later is nearer
layout(location = 5) in float w;		// 3D in a scene: its view depth (textures and colours interpolate in perspective); else 1
layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vColor;
layout(location = 2) flat out uvec4 vTex;

void main()
{
	gl_Position = vec4((pos.x / 512.0 - 1.0) * w, (pos.y / 256.0 - 1.0) * w, depth * w, w);
	vUv = uv;
	vColor = color;
	vTex = tex;
}
