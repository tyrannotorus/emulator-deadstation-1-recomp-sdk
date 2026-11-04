#version 450
#extension GL_GOOGLE_include_directive : require
// A model flattened onto the floor along the rays from one of the flashes (Blinn's planar projection, for a point light): where
// it lands, the stencil is marked, and that light's share of the floor is not added there (scene_flash_floor.frag). A corner at
// or above the light has no such ray to the floor: it goes as far as the light reaches, away from the light, as a body
// reaching the light's height shadows the floor all the way out.
#include "scene_params.glsl"
layout(location = 1) in vec3 pos;
layout(push_constant) uniform Pass { int index; } pass;

void main()
{
	vec3 light = params.lightAt[pass.index].xyz;
	float longest = params.lightAt[pass.index].w + 1000.0;	// past where its light ends (scene_flash_floor.frag)
	vec3 up = params.floorPlane.xyz;
	float lightHeight = dot(light, up) - params.floorPlane.w;
	float height = dot(pos, up) - params.floorPlane.w;
	vec3 away = pos - light;
	vec3 across = away - up * dot(away, up);
	float len = length(across);
	float gap = lightHeight - height;
	float reach = gap > 1.0 ? min(len * lightHeight / gap, longest) : longest;
	vec3 flat_ = light - up * lightHeight + (len > 0.001 ? across / len : vec3(0.0)) * reach;
	// (below the floor, the light casts nothing: all at a point)
	if (lightHeight <= 1.0)
		flat_ = vec3(0.0, 0.0, 1.0);
	// its place on the game's screen, as a clip position (w: the view depth), so a shadow reaching behind the eye is clipped
	float z = flat_.z;
	vec2 screenZ = params.project.xy * z + vec2(flat_.x, flat_.y * params.project.w) * params.project.z;
	gl_Position = vec4(screenZ / params.screen.zw * 2.0 - vec2(z), z * clamp(z / params.light.z, 0.0, 1.0), z);
}
