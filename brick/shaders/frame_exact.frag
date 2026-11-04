#version 450
// A frame the size of the page, pixel for pixel: no filter's worth of arithmetic and no clamp, on every pixel of every frame.
precision mediump float;
layout(binding = 0) uniform mediump sampler2D tex;
layout(location = 0) in highp vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
	color = vec4(texture(tex, uv).rgb, 1.0);
}
