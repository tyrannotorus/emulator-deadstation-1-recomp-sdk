#version 450
// One triangle over the whole page; its uv covers the frame's part of the texture.
layout(push_constant) uniform Frame { vec2 uvOffset; vec2 uvScale; vec2 uvMin; vec2 uvMax; } frame;
layout(location = 0) out vec2 uv;

void main()
{
	vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	uv = frame.uvOffset + pos * frame.uvScale;
	gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
