// The layer's parameters (scene_light.cpp's Params), std140.
layout(std140, set = 1, binding = 1) uniform Params
{
	vec4 axisU;		// sun space: u = (dot(p, axisU.xyz) + axisU.w) * scale.x, in 0..1
	vec4 axisV;		// v likewise with scale.y
	vec4 axisS;		// xyz: toward the sun; depth = (axisS.w - dot(p, axisS.xyz)) * scale.z, 0 nearest the sun
	vec4 scale;
	vec4 screen;	// xy: the target's pixels; zw: the game's screen (pixels) the corners are in
	vec4 frameUv;	// xy: the frame's uv scale; zw: its largest uv
	vec4 light;		// x: ambient (share of the sun's light kept in shade), y: darkness, z: view depth's far end, w: bias
	vec4 flashRight;	// the brightest flash's view (its shadow map): across, w: tan of half its angle
	vec4 flashUp;		// up, w: its near end
	vec4 flashAhead;	// ahead, w: its far end
	vec4 flashAt;		// xyz: where it is; w: how many lights
	vec4 lightAt[6];	// the lights (view space), w: where each ends
	vec4 lightColour[6];	// their colours x strengths
	vec4 options;		// x: the models' sun shadows on themselves
	vec4 frameAt;		// xy: the frame's uv offset; zw: its smallest uv
	vec4 floorPlane;	// the floor: dot(p, xyz) = w, xyz its unit normal, up
	vec4 project;		// a view space point's place on the game's screen: ofx, ofy, h, the camera's vertical scale
} params;

vec3 sunSpace(vec3 p)
{
	return vec3((dot(p, params.axisU.xyz) + params.axisU.w) * params.scale.x,
			(dot(p, params.axisV.xyz) + params.axisV.w) * params.scale.y,
			(params.axisS.w - dot(p, params.axisS.xyz)) * params.scale.z);
}

// p in the brightest flash's view, as a clip position (w: its distance ahead of the flash)
vec4 flashClip(vec3 p)
{
	vec3 q = p - params.flashAt.xyz;
	float w = dot(q, params.flashAhead.xyz);
	float n = params.flashUp.w, f = params.flashAhead.w;
	return vec4(dot(q, params.flashRight.xyz) / params.flashRight.w, dot(q, params.flashUp.xyz) / params.flashRight.w,
			(w - n) * f / (f - n), w);
}
