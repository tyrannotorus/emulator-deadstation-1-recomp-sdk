// A module's in-between frames: a game that draws every second blank (30 frames a second) gets the blank between
// from its last frame's own polygons, moved half a step on: on the render thread, no work for the game's thread, one more
// scene pass for the GPU. The plugin says, per frame the game draws, where the camera and each moving object are; polygons
// are tagged with their object as the display list reaches them (object 0: the world, moved by the camera's motion).
#pragma once
#include <cstddef>
#include <cstdint>

namespace deadstation::replay
{
constexpr unsigned MaxObjects = 8;	// the world (0) and up to 7 moving objects
struct Transform
{
	float r[9];		// object (or world) -> view: rotation (with the camera's scale)
	float t[3];		// and translation
	bool present;
};
struct Step	// view -> view
{
	float r[9], t[3];
};
struct Steps
{
	Step objects[MaxObjects];
};
struct Frame
{
	bool valid;
	float ofx, ofy, h;		// the GTE's projection
	float drawX, drawY;		// the draw offset: VRAM pixels = the game's screen + these
	Transform objects[MaxObjects];
};
}

// A plugin's polygons in bulk: one block to the render thread rather than three queued calls each. Corners in VRAM pixels
// (the draw offset included) with their true depths; drawn as the GPU's shaded textured triangles with the current draw area.
struct DeadstationTriangle
{
	float x[3], y[3], z[3];
	uint32_t rgb[3];
	uint16_t uv[3];
	uint16_t page, clut;
	uint8_t raw, semi;		// semi: 0 none, 1 + the blend mode
	// How far behind what lies over it, in 32s of depth (negative: in front), for a game that layers polygons by the order it draws them rather than where they are: tested and
	// written that much farther off, seen where it is. So what was laid over a surface stays over it though that surface comes up through it here and there.
	int8_t layer;
};
extern "C" void deadstation_triangles(const DeadstationTriangle *triangles, size_t count);

// semi for a solid polygon in the order it's given among others (a model whose decals lie on its panels as the game ordered them): tested
// against the depth, none written; then deadstation_depth_only writes theirs, for what's drawn after.
constexpr uint8_t DeadstationPainted = 0x81;

// Light a plugin adds to the frame so far: untextured, in full colour, each corner its own 0xBBGGRR, added to what is
// there. z: its true depth, behind what's nearer and moved with its object in an in-between frame (replay.h); 0 (or left out): none, over
// everything, where it is. One block to the render thread and one draw there. Corners as above; a frame not in a scene draws none.
struct DeadstationGlow
{
	float x[3], y[3];
	uint32_t rgb[3];
	float z[3];
};
extern "C" void deadstation_glows(const DeadstationGlow *glows, size_t count);
// A veil of fog: the same, not added but laid over at about a seventh of its colour, only over what is farther than its z; a veil stays put in
// an in-between frame. A few, near to far, are fog with distance over everything that has a depth.
extern "C" void deadstation_veil(const DeadstationGlow *glows, size_t count);
// A shadow on what's drawn: each spot once (the stencil), tested against the depth. darken: blended half way to its colour, as the PlayStation's
// shadows; else no colour at all, it only marks the stencil.
extern "C" void deadstation_shadows(const DeadstationGlow *polygons, size_t count, int darken);
extern "C" void deadstation_depth_only(const DeadstationGlow *polygons, size_t count);

// A colour over everything drawn from now on, red, green and blue in 256ths (a raw texture is treated as modulated by it): a race's light
// (a module's plugin says where it starts and stops, so what's drawn over the world keeps its own colours). In order with what's drawn.
extern "C" void deadstation_tint(int r, int g, int b);

// The next vertical blank shows nothing new and isn't to be shown at all: a module catching up, a step of the game with no frame to give (its time still passes)
extern "C" { extern int deadstation_no_present; }

// the next display list's frame (nullptr: the next frames aren't to be replayed)
extern "C" void deadstation_replay_frame(const deadstation::replay::Frame *frame);
// Where each object really is at the blank between, for a game whose logic runs at every blank and draws every second: for the newest frame,
// in place of half way to the next.
extern "C" void deadstation_replay_steps(const deadstation::replay::Steps *steps);
// the object the next polygons belong to
extern "C" void deadstation_replay_object(int object);
// frames in the last second made this way
extern "C" unsigned deadstation_replay_take_count(void);

// A world as a mesh: a game's static polygons, given once and kept in a buffer on the GPU in the world's own units, the camera moving them in a vertex shader. No polygon is projected
// on the CPU, none dropped for crossing the camera (the GPU clips it), none sent again each frame. Each cell of the world's a run of it, for what's in view to be drawn.
struct DeadstationWorldTriangle
{
	int16_t p[3][3];		// x, y, z
	uint8_t uv[3][2];
	uint8_t rgb[3][3];
	uint16_t page, clut;
	uint8_t raw, twoSided;
	int8_t layer;			// (DeadstationTriangle's)
	uint16_t cell;
	int16_t nudge;			// and this much more, in units of depth: one of two polygons lying in the same place
};
struct DeadstationWorldCamera
{
	float r[9], t[3];		// world to view
	float ofx, ofy, h;		// the view on the game's screen: x = ofx + h * view x / view z
	float drawX, drawY;		// and the screen in VRAM
};
extern "C" void deadstation_world_load(const DeadstationWorldTriangle *triangles, size_t count, unsigned cells);
// The cells (ascending), the world's light the tint (deadstation_tint), in order with what's drawn
// parts: 1 what's solid, 2 what's cut out, near to far. The cells near to far. (Solid first and the sky next, for what's behind it to cost nothing; then what's cut out.)
extern "C" void deadstation_world_draw(const DeadstationWorldCamera *camera, const uint16_t *cells, size_t count, int parts);
// Where in VRAM (x, y, w, h, each) the game will load other texels during a race. Before the load: what's there now says nothing about what's cut out of them later.
extern "C" void deadstation_world_animated(const int16_t *rects, size_t count);
// A module's own shaders for its world, for lighting it its own way: SPIR-V, given before the first frame (deadstation_module_start). They take what the SDK's take, and the same push constants
// (shaders/psx_gpu_world.vert), with float numbers[DeadstationWorldNumbers] after them, whatever the module wants them to mean. vert: for all of the world's polygons, in place of the SDK's. solidVert,
// solidFrag, solidEdgeFrag (all or none; the fragment ones as psx_gpu_world_opaque.frag and psx_gpu_world_edge.frag): for what's solid, while deadstation_world_numbers says so.
constexpr int DeadstationWorldNumbers = 28;
struct DeadstationWorldShaders
{
	const uint32_t *vert;
	size_t vertBytes;
	const uint32_t *solidVert;
	size_t solidVertBytes;
	const uint32_t *solidFrag;
	size_t solidFragBytes;
	const uint32_t *solidEdgeFrag;
	size_t solidEdgeFragBytes;
};
extern "C" void deadstation_world_shaders(const DeadstationWorldShaders *shaders);
// The numbers for what's drawn from now on: solid for the mesh's solid polygons, others for what's cut out and solid quads (nullptr: zeros; what's added or blended has zeros). ownSolid: what's solid
// drawn with the module's solid shaders.
extern "C" void deadstation_world_numbers(const float *solid, const float *others, int ownSolid);
// A sky: a mesh of its own, seen from where the camera is whatever its place. page 0xFFFF: no texture, its colour as it is.
extern "C" void deadstation_world_sky_load(const DeadstationWorldTriangle *triangles, size_t count);
extern "C" void deadstation_world_sky(const DeadstationWorldCamera *camera);
// What moves or changes each frame, as quads in the world (two triangles: 0 1 2, 1 3 2), drawn with the camera the mesh is. semi: 0 solid, 1 + a blend mode, over what's drawn, none of its
// depth written. Solid ones are lit by beams with the mesh.
struct DeadstationWorldQuad
{
	int16_t p[4][3];
	uint8_t uv[4][2];
	uint8_t rgb[4][3];
	uint16_t page, clut;
	uint8_t raw, semi;
	int8_t layer;
};
extern "C" void deadstation_world_quads(const DeadstationWorldCamera *camera, const DeadstationWorldQuad *quads, size_t count);
// Fog on what's drawn from now on through the mesh and bulk polygons: amount = most * sqrt((z - near) / (far - near)). most 0: none.
extern "C" void deadstation_fog(float nearZ, float farZ, float most, float r, float g, float b);
extern int deadstation_world_skip;
// What's about to be drawn covers the whole frame, so the last one in its place needn't be read in first: this colour, for whatever it leaves
extern "C" void deadstation_scene_whole(float r, float g, float b);
// Flat polygons in their rgb, over what's there, no depth
extern "C" void deadstation_flat(const DeadstationGlow *polygons, size_t count);
