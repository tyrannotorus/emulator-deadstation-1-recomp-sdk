// A recomp module's 3D layer, from the game's plugin to Deadstation's presenter (scene_light.h): the models the
// game drew this frame as triangles, where the game projected them (its screen pixels) and where they are (view
// space, the camera's own axes and units, x right, y down, z ahead), with the sun's direction and the flashes, so
// the presenter can light the frame's models per pixel (self-shadowing, the flashes' light and shadows). The plugin
// builds one per game frame and publishes it once complete; the presenter takes the latest with each frame.
#pragma once
#include <cstdint>
#include <mutex>
#include <vector>

namespace deadstation::scene
{

struct Corner
{
	float sx, sy;		// the game's screen pixels
	float x, y, z;		// view space
	float nx, ny, nz;	// the surface's normal there (view space, unit)
};

// a flash: a point light in view space; its colour times its strength (1: as bright as the sun at a metre)
struct Light
{
	float x, y, z;
	float r, g, b;
	float reach = 3000;		// where its light ends
};

struct Frame
{
	std::vector<Corner> corners;		// 3 a triangle
	std::vector<Corner> floor;			// the floor (3 a triangle): round the models, where the sun's shadows fall, then round the lights
	uint32_t floorBase = 0;				// how many of its corners are the first: only they draw the sun's shadows
	std::vector<Light> lights;			// the brightest first (its shadow map, on the models); at most MaxLights, each casting
										// its own planar shadows of the models on the floor
	float width = 0, height = 0;		// the game's screen (pixels) the corners' sx, sy are in
	float sun[3] = { 0, -1, 0 };		// toward the sun (view space, unit)
	float ambient = 0.5f;				// the light a surface keeps in the shade, as a share of the sun's (>= 0)
	float darkness = 1.0f;				// how much of the sun's share a shadow takes (0: none, 1: all)
	bool selfShadow = true;				// the models' sun shadows on themselves (Lighting: Original: none)
	bool flashMap = false;				// lights[0], a passing flash, shadows the models from its map (a stage's lamps do not: a pass every frame)
	float floorPlane[4] = { 0, -1, 0, 0 };	// the floor in view space: dot(p, xyz) = w, xyz its unit normal, up
	float project[4] = { 0, 0, 1, 1 };	// a view space point's place on the screen: ofx, ofy, h, the camera's vertical scale
	uint64_t serial = 0;				// which game frame
};

constexpr unsigned MaxLights = 6;

// the plugin: a frame's layer, complete
inline std::mutex& lock()
{
	static std::mutex m;
	return m;
}
inline Frame& latestFrame()
{
	static Frame f;
	return f;
}
inline void publish(Frame& frame)
{
	std::lock_guard<std::mutex> hold(lock());
	std::swap(latestFrame(), frame);
}
// the presenter: the latest layer (empty corners: none)
inline void take(Frame& out)
{
	std::lock_guard<std::mutex> hold(lock());
	out.corners.assign(latestFrame().corners.begin(), latestFrame().corners.end());
	out.floor.assign(latestFrame().floor.begin(), latestFrame().floor.end());
	out.lights.assign(latestFrame().lights.begin(), latestFrame().lights.end());
	out.width = latestFrame().width;
	out.height = latestFrame().height;
	for (int i = 0; i < 3; i++)
		out.sun[i] = latestFrame().sun[i];
	out.ambient = latestFrame().ambient;
	out.darkness = latestFrame().darkness;
	out.floorBase = latestFrame().floorBase;
	out.selfShadow = latestFrame().selfShadow;
	out.flashMap = latestFrame().flashMap;
	for (int i = 0; i < 4; i++)
		out.floorPlane[i] = latestFrame().floorPlane[i], out.project[i] = latestFrame().project[i];
	out.serial = latestFrame().serial;
}

}
