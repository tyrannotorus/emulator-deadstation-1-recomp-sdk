#version 450
// The frame, bilinear, never past its first or last texel (the texture is larger than the frame).
layout(push_constant) uniform Frame { vec2 uvOffset; vec2 uvScale; vec2 uvMin; vec2 uvMax; } frame;
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
	color = vec4(texture(tex, clamp(uv, frame.uvMin, frame.uvMax)).rgb, 1.0);
}
