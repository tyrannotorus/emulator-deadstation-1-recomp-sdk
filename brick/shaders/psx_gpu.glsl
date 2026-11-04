// The PlayStation GPU's drawing in Vulkan (psx_gpu.cpp): the push constants every pipeline shares, and the 16-bit VRAM
// mirror textures read (raw 1555 words at the PlayStation's own 1024 x 512).
layout(push_constant) uniform Push
{
	ivec4 window;	// texture window: x, y mask; z, w offset (8-texel steps)
	ivec4 flags;	// x: set the mask bit; y: STP pass (0 every texel, 1 the opaque ones, 2 the semi-transparent ones); z: scale
	vec4 rect;		// a rect's corners in NDC (x0, y0, x1, y1): blits and packs
} pc;
