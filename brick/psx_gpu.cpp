// The PlayStation GPU drawn by Vulkan (psx_gpu.h). VRAM is an RGBA8 image at the module's scale (the mask bit in
// alpha, and in a stencil buffer for mask tests); textures read a raw 16-bit mirror of VRAM at the PlayStation's own
// size, which a pack pass refreshes from the scaled image when a texture page or CLUT overlaps what was drawn since.
// Primitives are batched into one render pass; uploads, copies, packs and readbacks end it. Frames are shown straight
// from the scaled image. The CPU's copy of VRAM (PSXRecomp's) is written by uploads and refreshed on readbacks.
#include "psx_gpu.h"
#include "replay.h"

extern "C" uint32_t psx_gpu_source_addr(void);	// patch 0028
// told each triangle's packet address, for a module tagging polygons by what drew them
extern "C" { void (*psx_gpu_triangle_source_hook)(uint32_t addr) = nullptr; }
#include "log.h"

#include "psx_gpu_vert.h"
#include "psx_gpu_flat_frag.h"
#include "psx_gpu_tex_frag.h"
#include "psx_gpu_cached_frag.h"
#include "psx_gpu_cached_opaque_frag.h"
#include "psx_gpu_lean_vert.h"
#include "psx_gpu_world_vert.h"
#include "psx_gpu_world_frag.h"
#include "psx_gpu_world_opaque_frag.h"
#include "psx_gpu_world_sky_vert.h"
#include "psx_gpu_world_edge_frag.h"
#include "psx_gpu_world_sky_frag.h"
#include "psx_gpu_lean_frag.h"
#include "psx_gpu_lean_opaque_frag.h"
#include "psx_gpu_decode_frag.h"
#include "psx_gpu_rect_vert.h"
#include "psx_gpu_blit_frag.h"
#include "psx_gpu_pack_frag.h"

#include "gpu_render.h"
#include "gpu_sw_renderer.h"
#include "gpu_vk_renderer.h"

#include <algorithm>
#include <ctime>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int VramW = 1024, VramH = 512;
constexpr uint32_t Slots = 3;
constexpr vk::DeviceSize VertexBytes = 4u << 20, StagingBytes = 2u << 20;

struct Vertex
{
	float x, y, u, v;
	uint8_t r, g, b, a;
	uint16_t page, clutX, clutY, flags;
	float depth = 0;	// the order depth (set as the primitive is recorded)
	float w = 1;		// clip w: 3D in a scene, its view depth (perspective-correct); else 1
};

struct Push
{
	int32_t window[4];
	int32_t flags[4];	// x: set mask, y: STP pass, z: scale
	float rect[4];
};

// (their numbers are in modules' lists of pipelines, pipelines.keys: one that's gone keeps its place)
enum Kind : uint32_t { Flat, Tex, Blit, Pack, Cached, Decode, CachedOpaque, Lean, LeanOpaque, KindRetired, World, WorldOpaque, WorldOwnSolid, WorldSky, WorldOwnEdge, WorldEdge };
// a pipeline key's bit: drawn into a native scene (its render pass, with its depth buffer)
constexpr uint32_t SceneBit = 1u << 22;
constexpr uint32_t StencilMark = 1u << 24;	// a shadow's polygons: no colour, a spare bit of the stencil where not already darkened
constexpr uint32_t StencilMarked = 1u << 25;	// and the one draw that darkens where that bit is, moving it to the bit that says it's done
constexpr uint32_t PageAlpha = 1u << 26;	// drawn straight into a page (flushScene): alpha left as it's cleared, opaque. The display blends by each pixel's alpha, over black
constexpr uint32_t CullBit = 1u << 23;	// a world's one-sided polygons: the backs not drawn
// A world as a mesh (replay.h)
struct WorldVertex
{
	int16_t x, y, z, layer;
	uint8_t u, v, flags, slot;
	uint8_t r, g, b, a;
};
DeadstationWorldShaders worldShadersGiven {};	// (a module's, deadstation_world_shaders)
struct WorldNumbersOp { float numbers[2][DeadstationWorldNumbers]; int ownSolid; };
struct WorldPush
{
	float rows[12], project[4], a[4], b[4], c[4];	// (a tint, b fog's colour and most, c its range)
	float numbers[DeadstationWorldNumbers];	// a module's own world shaders' (deadstation_world_numbers)
};
constexpr vk::Format SceneFormat = vk::Format::eB8G8R8A8Unorm;	// a page's (Panel::format()): a whole frame is drawn straight into one
constexpr int VeilBlend = 5;	// past the PlayStation's four: a plugin's fog (deadstation_veil)
constexpr int BlockBlend = 6;	// no colour: it only marks the stencil (deadstation_shadows)
constexpr int DarkenBlend = 7;	// what's there times the colour: a shadow, which can't lighten what's darker than it
// the texture cache: texture pages decoded (through their CLUTs) into 256 x 256 slots of an RGBA atlas, so a textured
// fragment takes one fetch, not two dependent ones and the decoding (psx_gpu_cached.frag)
// 256 slots: a busy 3D scene uses more than 128 pages through their CLUTs, and with 128 the least recently used were decoded again every frame
constexpr int AtlasW = 4096, AtlasH = 4096, TexSlots = (AtlasW / 256) * (AtlasH / 256);
constexpr int SlotWords = TexSlots / 64;
static_assert(TexSlots == 256, "slot masks are four words");
// decodes a frame at most (the rest stay queued, their pages drawn from VRAM meanwhile): no frame takes a burst
constexpr int DecodesPerFrame = 3;
constexpr int DecodeBands = 8;		// pages one pass can decode: a world's mesh, its pages all wanted now

// DEADSTATION_PERF: a slot's timestamps: its commands' begin and end, then pairs around its operations
constexpr uint32_t StampPairs = 32, StampsPerSlot = 2 + StampPairs * 2;
enum StampKind : uint8_t { StampPass, StampPack, StampCopy, StampUpload, StampDecode, StampScene, StampKinds };

struct Slot
{
	bool timed = false;		// its timestamps written (DEADSTATION_PERF)
	bool timing = false;	// being written (its recording began in a timed frame)
	uint32_t stamps = 0;	// pairs' timestamps written
	uint8_t stampKind[StampPairs] = {};
	vk::UniqueCommandBuffer cmd;
	vk::UniqueFence fence;
	vk::UniqueBuffer vertices, staging;
	vk::UniqueDeviceMemory verticesMemory, stagingMemory;
	uint8_t *vertexMap = nullptr, *stagingMap = nullptr;
	vk::DeviceSize vertexUsed = 0, stagingUsed = 0;
	bool recording = false;
	int borrowed = -1;		// submitted with a frame (Video::frameImage): that page's, whose fence is the panel's to wait on (Panel::waitDrawn), not ours
};

struct Image
{
	vk::UniqueImage image;
	vk::UniqueDeviceMemory memory;
	vk::UniqueImageView view;
};

struct State
{
	bool ready = false;
	int scale = 1;
	int filter = 0;
	uint16_t *cpu = nullptr;	// PSXRecomp's VRAM
	// the renderer's: PSXRecomp's itself, or with the render thread its own copy (PSXRecomp's written only while the
	// module's thread waits for the render thread)
	uint16_t *mirror = nullptr;
	Video::Device dev;
	vk::Format stencilFormat = vk::Format::eUndefined;
	Image scaled, stencil, raw, copy;
	vk::UniqueRenderPass drawPass, colorPass, packPass;	// drawing with the stencil, without, packing
	vk::UniqueFramebuffer drawFramebuffer, colorFramebuffer, packFramebuffer;
	bool stencilOn = false;		// from the game's first use of the mask bit: drawing passes keep the stencil
	// The scene's depth (DEADSTATION_GPU_DEPTH=0: none): each 3D triangle's corners carry the depth of the GTE projection
	// that made them (patch 0022), and a drawing pass is drawn in three steps: its opaque 3D primitives nearest first,
	// tested and written (what they hide is never shaded), its 2D primitives from before any 3D one (backgrounds) behind
	// them, then the rest in the game's order (semi-transparent 3D tested, sprites and the HUD over everything). The depth
	// lives in tile memory only (cleared at each pass, never stored). With the mask bit in play: the game's order only.
	bool sceneDepth = false;
	float triDepth[3] = {};
	bool triDepthValid = false;		// the next triangle's (set_triangle_depth)
	struct Deferred
	{
		uint32_t first, count, key;
		vk::Rect2D scissor;
		Push push;
		bool twoPasses;
		uint8_t step;		// 0 opaque 3D, 1 background 2D, 2 the rest
		float depth;
	};
	std::vector<Deferred> passPrims;
	bool deferredAny3D = false;
	float lastDepth = 1.f;
	vk::UniqueSampler samplerFiltered;
	vk::UniqueSampler sampler;
	vk::UniqueDescriptorSetLayout setLayout;
	vk::UniqueDescriptorPool pool;
	vk::DescriptorSet set;
	vk::UniquePipelineLayout layout;
	vk::UniqueShaderModule vert, flat, tex, cached, cachedOpaque, leanVert, lean, leanOpaque, decode, rectVert, blit, pack;
	vk::UniqueShaderModule worldVert, world, worldOpaque, worldOwnVert, worldOwn, worldSkyVert, worldSky, worldOwnEdge, worldEdge;
	std::vector<int16_t> animatedRects;
	float worldNumbers[2][DeadstationWorldNumbers] = {};	// (what's solid in the mesh; what's cut out and solid quads)
	bool worldOwnSolid = false;
	uint32_t worldPushSize = sizeof(WorldPush);
	float fogNear = 0, fogScale = 0, fogMost = 0, fogRgb[3] = {};
	vk::UniqueBuffer skyBuffer;
	vk::UniqueDeviceMemory skyMemory;
	uint32_t skyVertices = 0;
	// a world as a mesh: its buffer, each kind of polygon's run for each of its cells (bit 0 of the kind: cut-outs, bit 1: two-sided), the atlas slots it keeps
	vk::UniquePipelineLayout worldLayout;
	vk::UniqueBuffer worldBuffer;
	vk::UniqueDeviceMemory worldMemory;
	struct WorldRun { uint32_t first = 0, count = 0; };
	std::vector<WorldRun> worldRuns[6];
	std::vector<int> worldSlots;
	bool takingForWorld = false;
	std::vector<WorldPush> worldPushes;
	// the texture cache (DEADSTATION_TEXCACHE=0: none). A page through a CLUT not in it is drawn from VRAM (psx_gpu_tex)
	// and its decode queued; the queue is decoded at the frame's end in one pass over the atlas (never between the
	// frame's drawing passes). A slot is taken back only when unused for two frames (no thrashing), and is out of date
	// once VRAM under its page or CLUT changes (each VRAM tile's slots, 128 bits).
	bool texCache = false;
	Image atlas;
	vk::Format atlasFormat = vk::Format::eR8G8B8A8Unorm;	// A1R5G5B5 where it can be drawn into: the texels exactly, half the reads
	vk::UniqueRenderPass atlasPass, atlasDecodePass;	// (decodes: a slot's texels not loaded, every one is written)
	// decodes drawn here, a 256 x 256 band each, then copied into their slots: a pass on the atlas itself stored all
	// 32 MB of it on the GE8300 (its render area notwithstanding), ~several ms a page
	Image decodeImage;
	vk::UniqueFramebuffer decodeFramebuffer;
	vk::UniqueFramebuffer atlasFramebuffer;
	struct TexSlot
	{
		uint32_t key = ~0u;
		uint32_t usedFrame = 0;
		bool valid = false, pending = false;
		uint16_t page = 0, clutX = 0, clutY = 0;
		uint32_t depth = 0;
		uint8_t texels = 7;		// what its page's texels are: bit 0 transparent, 1 opaque, 2 STP (7: not known)
		uint32_t changedFrame = 0;	// when VRAM under it last changed
		bool smallChange = false;	// by a small upload (not drawn over, not a large texture loaded)
		bool kept = false;			// a world's mesh names it: never taken back
		int16_t changed[4] = { 0, 0, -1, -1 };	// of its 256 x 256 by small uploads since its decode: u0, v0, u1, v1 (u1 < 0: none known, all of it)
		uint16_t clearCells[16] = {};	// 16 rows of 16 bits: the 16 x 16 texel cells with a transparent texel
		uint16_t usedCells[16] = {};	// a world's page: those its polygons show, the rest of the page's changes nobody sees
		bool usedElsewhere = false;		// a world's page something else draws from too
	} texSlots[TexSlots];
	std::unordered_map<uint32_t, int> texSlotOf;
	uint64_t tileSlots[1024][SlotWords] = {};
	std::vector<int> texPending;
	uint32_t texFrame = 2;
	unsigned decodes = 0;
	// copies kept in the raw mirror too (MoveImage over undrawn VRAM: no pack needed after them)
	Image rawCopy;
	std::unordered_map<uint32_t, vk::UniquePipeline> pipelines;
	// A pipeline compiled when first drawn stalls the render thread for 100 ms or more, in the middle of a fight. So they are
	// built ahead of use: on a thread of their own from the start (warmPipelines), from the list of what the module draws
	// (pipelines.keys beside it, shipped with it and added to at each end), through the driver's cache kept on the card.
	vk::UniquePipelineCache pipelineCache;
	std::mutex pipelineLock;
	std::thread pipelineWarmer;
	std::atomic<bool> pipelineWarmerStop { false };
	bool pipelineKeysNew = false;
	vk::UniqueCommandPool commandPool;
	Slot slots[Slots];
	uint32_t current = 0;
	vk::UniqueBuffer readback;
	vk::UniqueDeviceMemory readbackMemory;
	uint8_t *readbackMap = nullptr;
	bool inPass = false;
	int passArea[4] = { 0, 0, 0, 0 };	// the pass's VRAM rect (x0, y0, x1, y1 exclusive)
	// DEADSTATION_PERF: what the GPU did, a line a second, and its time on the slots' commands (timestamps: begin, end)
	bool perf = false;
	// timestamps written in one frame in TimedEvery (the sums scaled to all): the PowerVR driver keeps kernel memory for
	// every timestamp written until the process ends (about 2 MB a second when every frame is timed)
	vk::UniqueQueryPool timestamps;
	static constexpr unsigned TimedEvery = 8;
	double gpuSeconds = 0, gpuKind[StampKinds] = {};
	unsigned draws = 0;		// draw calls made
	bool scenePerspective = true;	// DEADSTATION_SCENE_W=0 off
	// this frame is 3D whose traced depths mean what they say (a module's plugin says so, deadstation_scene_perspective):
	// elsewhere (2D layouts drawn through the GTE: a select screen's portraits) a corner's depth need not match where the
	// corner is, and perspective would warp its texture
	bool frameIs3D = false;
	int quickDecodes = 0;			// pages decoded at once this frame (decodeNow)
	unsigned drawnSinceShown = 0;	// the game's primitives since the last frame shown (none: the game repeated its frame)
	// the pixels primitives cover at the drawing's scale (their triangles' areas, twice for the two STP passes): all of
	// them, the blended ones, the ones textured from VRAM (not the cache), the untextured ones
	double pixels = 0, pixelsBlended = 0, pixelsVram = 0, pixelsFlat = 0;
	uint32_t passStamp = ~0u;
	std::chrono::steady_clock::time_point perfStart;
	unsigned frames = 0, passes = 0, packs = 0, uploads = 0, copies = 0, fills = 0, readbacks = 0;
	uint64_t vertices = 0;
	// the drawing state PSXRecomp sets
	bool semi = false;
	int semiMode = 0;
	bool maskSet = false, maskCheck = false;
	uint32_t window = 0;
	uint8_t mod[3] = { 128, 128, 128 };
	bool modRaw = false;
	int area[4] = { 0, 0, VramW - 1, VramH - 1 };
	// the batch: primitives with one pipeline, scissor and push constants
	bool runOpen = false;
	uint32_t runKey = 0, runFirst = 0, runCount = 0;
	bool runTextured = false, runSemi = false;
	vk::Rect2D runScissor;
	Push runPush {};
	// what was drawn since the raw mirror was last packed: 32 x 16 VRAM tiles (32 x 32)
	uint64_t dirty[16] = {};
	// the same tiles: drawn by the GPU since PSXRecomp's VRAM was last read back (the CPU's copy is stale there), and
	// uploaded by the CPU but not yet given to the GPU (deferred uploads: the CPU's copy is newer there)
	uint64_t gpuNewer[16] = {}, cpuNewer[16] = {};
	// the same tiles: uploaded into the raw mirror and not yet into the scaled image (it takes them only when something
	// reads it or draws over them there: a render pass on the scaled image loads and stores all of it on this GPU, its
	// render area notwithstanding, so an upload is a transfer into the mirror only)
	uint64_t rawNewer[16] = {};
	unsigned deferred = 0;
	// drawn since PSXRecomp's VRAM was last read back
	bool cpuStale = false;
	// The native scene (DEADSTATION_SCENE=0: none). A frame buffer (a draw area the size of the frame last shown) is drawn
	// into an image of the panel's size with a depth buffer, not into VRAM: triangles at their GTE positions (sub-pixel,
	// PGXP), 3D ones tested at their projection's depth (opaque ones writing it), 2D ones in the game's order (backgrounds
	// before any 3D, sprites and the HUD over it). Its primitives are recorded as they come and drawn in one render pass
	// at the frame's end (sooner only before VRAM they read changes), and the frame is shown straight from it. VRAM's copy
	// of the area is brought up to date only when something reads it (a resolve); what writes the area by other means
	// (uploads, copies) goes into the scene from VRAM.
	bool sceneOn = false;
	struct Scene
	{
		bool used = false;
		int ax = 0, ay = 0, aw = 0, ah = 0;	// its VRAM area
		uint32_t lastUsed = 0;		// the frame (texFrame) it was last drawn into or shown
		bool depthStarted = false;	// its depth cleared since it was last shown
		int pageHolding = -1;		// its last frame went straight into this page, not into its image (restoreScenes)
		vk::UniqueFramebuffer onPage[8];	// each page with its depth, for that
		bool any3D = false;			// a 3D primitive drawn since it was last shown (2D ones after it are no background)
		float lastDepth = 1.f;		// the last 3D primitive's (a polygon without a depth among 3D ones takes it)
		bool newer = false;			// drawn since VRAM last took it (any of drawn)
		// the VRAM tiles (32 x 16) it drew into that VRAM has not taken: the only pixels a resolve writes back (VRAM keeps
		// what was written there by other means: a write over drawn tiles has them resolved first)
		uint64_t drawn[16] = {};
		// written in VRAM by other means since it last took VRAM: all of its area (VRAM held everything), or these rects
		// (x0, y0, x1, y1; it holds drawing VRAM lacks around them)
		bool vramAll = false;
		std::vector<std::array<int, 4>> vramRects;
		Image color, depth;
		vk::UniqueFramebuffer framebuffer;
	};
	static constexpr int SceneCount = 4;
	Scene scenes[SceneCount];
	uint32_t sceneW = 0, sceneH = 0;
	vk::Format sceneDepthFormat = vk::Format::eUndefined;
	bool sceneStencil = false;		// its depth format has a stencil: the mask bit there as in VRAM
	vk::UniqueRenderPass scenePasses[8];	// [colour cleared << 2 | depth loaded << 1 | depth stored]
	bool sceneWhole = false;				// a module's whole frame is on its way: what it's drawn over needn't be read in
	float sceneClear[3] = {};				// and the colour for what it leaves
	int framePage = -1;						// the page it's to go straight into, as it's drawn
	int lastOnPage = -1;					// the scene the last frame went straight from
	bool frameOnPage = false;				// and it did
	unsigned framesOnPages = 0;
	double lastRecordMs = 0, lastPageMs = 0;
	int displayW = 0, displayH = 0;		// the frame last shown (frame buffers are draw areas of its size)
	// the rects of VRAM shown lately (a draw area inside one draws a frame buffer: the game's viewport, its HUD)
	struct ShownRect { int x = 0, y = 0, w = 0, h = 0; uint32_t frame = 0; bool used = false; };
	ShownRect shownRects[4];
	struct SceneDraw
	{
		int scene;
		uint32_t first, count, key;		// count 0: a clear of the scissor (colour clearColor, 15-bit, and depth)
		vk::Rect2D scissor;
		Push push;
		bool twoPasses;
		uint16_t clearColor;
		int world = -1;				// of a world's mesh: its push constants
		int source = 0;				// 0 the mesh, 1 the sky's, 2 the frame's own vertices
	};
	std::vector<SceneDraw> sceneDraws;
	// Each frame kept, per scene, to be drawn again for its second blank (replay.h). By then the game has drawn the next
	// frame (it flips, then draws, between blanks), so the in-between frame lies between two real frames.
	struct Replay
	{
		deadstation::replay::Frame frame {};
		std::vector<Vertex> vertices;
		std::vector<float> z;			// the corners' view depths (0: no depth: stays where it is)
		std::vector<uint8_t> object;	// bit 7: drawn with the depth bias of what's blended
		std::vector<SceneDraw> draws;
	} replays[SceneCount];
	deadstation::replay::Frame replayLatest {};
	bool replayPending = false;			// a frame was announced: its first primitive starts its recording
	int replayRecording = -1;			// the scene being recorded into
	int replayNewest = -1;				// the scene holding the newest frame
	deadstation::replay::Steps replaySteps {};
	int replayStepsFor = -1;			// the scene they take a step from (deadstation_replay_steps)
	int replayObject = 0;
	int lastShownScene = -1, lastShownX = -1, lastShownY = -1;
	std::chrono::steady_clock::time_point lastPresentAt;
	unsigned replayFrames = 0;
	int tint[3] = { 256, 256, 256 };	// deadstation_tint
	bool tinted = false;
	Scene replayScene;					// where they're drawn (its own images: the frame shown stays as it was drawn)
	uint64_t sceneReads[16] = {};		// the VRAM tiles recorded scene draws read (texels, CLUTs)
	// the next triangle's precise corners (PGXP, set_precise_triangle), VRAM pixels
	bool preciseValid = false;
	float precise[6] = {};
	unsigned scenePassCount = 0, sceneResolves = 0, sceneBlits = 0, sceneDrawCount = 0;
	// DEADSTATION_PERF: the render thread's seconds running batches, and waiting for a slot's earlier work there
	double rtBusy = 0, rtFenceWait = 0;
	// DEADSTATION_PERF: primitives drawn in VRAM since the last line, the last one's draw area and bounds
	unsigned vramPrims = 0;
	int lastVramArea[4] = {}, lastVramBounds[4] = {};
	double uploadPixels = 0;		// uploaded (texels), the last upload's rect
	int lastUpload[4] = {};
};
// never destroyed: at exit the driver may be gone before static destructors run
State& g = *new State;
double takeBlockedSeconds();	// (the render thread's, below)
vk::QueryPool stampPool(const Slot&) { return *g.timestamps; }

uint32_t memoryType(uint32_t bits, vk::MemoryPropertyFlags flags)
{
	const vk::PhysicalDeviceMemoryProperties props = g.dev.physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < props.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
			return i;
	throw std::runtime_error("no memory type");
}

void makeImage(Image& out, vk::Format format, uint32_t w, uint32_t h, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect)
{
	vk::Device d = g.dev.device;
	out.image = d.createImageUnique(vk::ImageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, format, vk::Extent3D(w, h, 1), 1, 1,
			vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage));
	const vk::MemoryRequirements req = d.getImageMemoryRequirements(*out.image);
	out.memory = d.allocateMemoryUnique(vk::MemoryAllocateInfo(req.size, memoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)));
	d.bindImageMemory(*out.image, *out.memory, 0);
	out.view = d.createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), *out.image, vk::ImageViewType::e2D, format,
			vk::ComponentMapping(), vk::ImageSubresourceRange(aspect, 0, 1, 0, 1)));
}

void makeBuffer(vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& memory, uint8_t *&map, vk::DeviceSize size, vk::BufferUsageFlags usage)
{
	vk::Device d = g.dev.device;
	buffer = d.createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage));
	const vk::MemoryRequirements req = d.getBufferMemoryRequirements(*buffer);
	memory = d.allocateMemoryUnique(vk::MemoryAllocateInfo(req.size,
			memoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)));
	d.bindBufferMemory(*buffer, *memory, 0);
	map = (uint8_t *)d.mapMemory(*memory, 0, size);
}

vk::UniqueShaderModule module(const uint32_t *code, size_t size)
{
	return g.dev.device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), size, code));
}

// every earlier command's memory work done and visible before any later command (the GPU's work is small and in
// few batches; one broad barrier per transition keeps the ordering obvious)
void barrier(vk::CommandBuffer cmd)
{
	vk::MemoryBarrier b(vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags(), b,
			nullptr, nullptr);
}

Slot& slot()
{
	Slot& s = g.slots[g.current];
	if (!s.recording)
	{
		const auto waitFrom = std::chrono::steady_clock::now();
		if (s.borrowed >= 0)
			hostVideo().panel().waitDrawn((uint32_t)s.borrowed);
		else
			(void)g.dev.device.waitForFences(*s.fence, true, UINT64_MAX);
		if (g.perf)
			g.rtFenceWait += std::chrono::duration<double>(std::chrono::steady_clock::now() - waitFrom).count();
		s.borrowed = -1;
		if (s.timed)
		{
			uint64_t t[StampsPerSlot];
			const double tick = hostVideo().timestampNs() * 1e-9;
			if (g.dev.device.getQueryPoolResults(stampPool(s), g.current * StampsPerSlot, 2 + s.stamps, (2 + s.stamps) * 8, t,
					sizeof(uint64_t), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess)
			{
				g.gpuSeconds += (double)(t[1] - t[0]) * tick * State::TimedEvery;
				for (uint32_t k = 0; k < s.stamps / 2; k++)
					g.gpuKind[s.stampKind[k]] += (double)(t[2 + 2 * k + 1] - t[2 + 2 * k]) * tick * State::TimedEvery;
			}
			s.timed = false;
		}
		s.stamps = 0;
		s.cmd->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		s.timing = g.timestamps && g.texFrame % State::TimedEvery == 0;
		if (s.timing)
		{
			s.cmd->resetQueryPool(stampPool(s), g.current * StampsPerSlot, StampsPerSlot);
			s.cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, stampPool(s), g.current * StampsPerSlot);
		}
		s.vertexUsed = s.stagingUsed = 0;
		s.recording = true;
		barrier(*s.cmd);	// after the frames shown from VRAM (their reads), and the last slot's work
	}
	return s;
}

// a pair of timestamps round an operation of the slot's (DEADSTATION_PERF): open returns its first, ~0u for none
uint32_t stampOpen(Slot& s, StampKind kind)
{
	if (!s.timing || s.stamps + 2 > StampPairs * 2)
		return ~0u;
	const uint32_t at = s.stamps;
	s.stamps += 2;
	s.stampKind[at / 2] = kind;
	s.cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, stampPool(s), g.current * StampsPerSlot + 2 + at);
	return at;
}
void stampClose(Slot& s, uint32_t at)
{
	if (at != ~0u)
		s.cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, stampPool(s), g.current * StampsPerSlot + 2 + at + 1);
}

vk::PipelineColorBlendAttachmentState blendState(int blend)
{
	using F = vk::BlendFactor;
	vk::PipelineColorBlendAttachmentState b(false);
	b.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;
	if (blend == 0)
		return b;
	if (blend == BlockBlend)
	{
		b.colorWriteMask = vk::ColorComponentFlags();
		return b;
	}
	if (blend == DarkenBlend)
	{
		b.blendEnable = true;
		b.srcColorBlendFactor = F::eZero, b.dstColorBlendFactor = F::eSrcColor, b.colorBlendOp = vk::BlendOp::eAdd;
		b.srcAlphaBlendFactor = F::eZero, b.dstAlphaBlendFactor = F::eOne, b.alphaBlendOp = vk::BlendOp::eAdd;
		return b;
	}
	// the colour blended, the mask bit (alpha) always the primitive's
	b.blendEnable = true;
	b.srcAlphaBlendFactor = F::eOne;
	b.dstAlphaBlendFactor = F::eZero;
	b.alphaBlendOp = vk::BlendOp::eAdd;
	switch (blend - 1)
	{
	case 0: b.srcColorBlendFactor = F::eConstantColor; b.dstColorBlendFactor = F::eConstantColor; b.colorBlendOp = vk::BlendOp::eAdd; break;	// B/2 + F/2
	case 1: b.srcColorBlendFactor = F::eOne; b.dstColorBlendFactor = F::eOne; b.colorBlendOp = vk::BlendOp::eAdd; break;	// B + F
	case 2: b.srcColorBlendFactor = F::eOne; b.dstColorBlendFactor = F::eOne; b.colorBlendOp = vk::BlendOp::eReverseSubtract; break;	// B - F
	case 4: b.srcColorBlendFactor = F::eConstantColor; b.dstColorBlendFactor = F::eOneMinusConstantColor; b.colorBlendOp = vk::BlendOp::eAdd; break;	// a veil
	default: b.srcColorBlendFactor = F::eConstantColor; b.dstColorBlendFactor = F::eOne; b.colorBlendOp = vk::BlendOp::eAdd; break;	// B + F/4
	}
	return b;
}

// key: kind | blend (0 none, 1 + the PlayStation's mode) << 4 | stencil (check << 1 | set) << 8 | the stencil pass << 12
vk::UniquePipeline buildPipeline(uint32_t key)
{
	const Kind kind = (Kind)(key & 15);
	const int blend = (key >> 4) & 15;
	const int stencil = (key >> 8) & 3;
	const bool check = stencil & 2, set = stencil & 1;
	const bool stencilPass = (key >> 12) & 1;

	const bool worldKind = kind == World || kind == WorldOpaque || kind == WorldOwnSolid || kind == WorldSky || kind == WorldOwnEdge || kind == WorldEdge;
	vk::PipelineShaderStageCreateInfo stages[2] = {
		{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
				kind == WorldSky ? *g.worldSkyVert : kind == WorldOwnSolid || kind == WorldOwnEdge ? *g.worldOwnVert : worldKind ? *g.worldVert :
				kind == Lean || kind == LeanOpaque ? *g.leanVert :
				kind == Flat || kind == Tex || kind == Cached || kind == CachedOpaque ? *g.vert : *g.rectVert, "main" },
		{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment,
				kind == WorldEdge ? *g.worldEdge : kind == WorldSky ? *g.worldSky : kind == WorldOwnEdge ? *g.worldOwnEdge : kind == WorldOwnSolid ? *g.worldOwn : kind == World ? *g.world : kind == WorldOpaque ? *g.worldOpaque :
				kind == Lean ? *g.lean : kind == LeanOpaque ? *g.leanOpaque :
				kind == Flat ? *g.flat : kind == Tex ? *g.tex : kind == Cached ? *g.cached : kind == CachedOpaque ? *g.cachedOpaque :
				kind == Decode ? *g.decode :
				kind == Blit ? *g.blit : *g.pack, "main" },
	};
	vk::VertexInputBindingDescription binding(0, sizeof(Vertex), vk::VertexInputRate::eVertex);
	vk::VertexInputAttributeDescription attributes[6] = {
		{ 0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, x) },
		{ 1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, u) },
		{ 2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(Vertex, r) },
		{ 3, 0, vk::Format::eR16G16B16A16Uint, offsetof(Vertex, page) },
		{ 4, 0, vk::Format::eR32Sfloat, offsetof(Vertex, depth) },
		{ 5, 0, vk::Format::eR32Sfloat, offsetof(Vertex, w) },
	};
	const bool geometry = kind == Flat || kind == Tex || kind == Cached || kind == CachedOpaque || kind == Lean || kind == LeanOpaque || worldKind;
	vk::PipelineVertexInputStateCreateInfo input = geometry ?
			vk::PipelineVertexInputStateCreateInfo(vk::PipelineVertexInputStateCreateFlags(), binding, attributes) :
			vk::PipelineVertexInputStateCreateInfo();
	vk::VertexInputBindingDescription worldBinding(0, sizeof(WorldVertex), vk::VertexInputRate::eVertex);
	vk::VertexInputAttributeDescription worldAttributes[3] = {
		{ 0, 0, vk::Format::eR16G16B16A16Sint, offsetof(WorldVertex, x) },
		{ 1, 0, vk::Format::eR8G8B8A8Uint, offsetof(WorldVertex, u) },
		{ 2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(WorldVertex, r) },
	};
	if (worldKind)
		input = vk::PipelineVertexInputStateCreateInfo(vk::PipelineVertexInputStateCreateFlags(), worldBinding, worldAttributes);
	vk::PipelineInputAssemblyStateCreateInfo assembly(vk::PipelineInputAssemblyStateCreateFlags(),
			geometry ? vk::PrimitiveTopology::eTriangleList : vk::PrimitiveTopology::eTriangleStrip);
	vk::PipelineViewportStateCreateInfo viewport(vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);
	// (a world's polygons are wound as the game has them: their fronts clockwise on the screen)
	vk::PipelineRasterizationStateCreateInfo raster(vk::PipelineRasterizationStateCreateFlags(), false, false, vk::PolygonMode::eFill,
			(key & CullBit) != 0 ? vk::CullModeFlagBits::eBack : vk::CullModeFlagBits::eNone, worldKind ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise, false, 0, 0, 0, 1.f);
	vk::PipelineMultisampleStateCreateInfo multisample;
	// the mask bit: check = draw only where the stencil is 0; the stencil written is the set-mask state
	vk::StencilOpState st(vk::StencilOp::eKeep, set && check ? vk::StencilOp::eInvert : vk::StencilOp::eReplace, vk::StencilOp::eKeep,
			check ? vk::CompareOp::eEqual : vk::CompareOp::eAlways, 1, 1, check ? 0u : (set ? 1u : 0u));
	// (mark: bit 1 set where bit 0 isn't. Marked: where bit 1 is, inverting both: 2 becomes 1, so darkened once however many shadows cross it)
	if (key & StencilMark)
		st = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eReplace, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 1, 2, 2);
	else if (key & StencilMarked)
		st = vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eInvert, vk::StencilOp::eKeep, vk::CompareOp::eEqual, 2, 3, 2);
	const uint32_t depthMode = geometry ? (key >> 20) & 3 : 0;
	const bool scene = (key & SceneBit) != 0;
	vk::PipelineDepthStencilStateCreateInfo depthStencil(vk::PipelineDepthStencilStateCreateFlags(), depthMode != 0, depthMode == 2,
			scene ? vk::CompareOp::eGreaterOrEqual : vk::CompareOp::eLessOrEqual, false, stencilPass && kind != Pack, st, st);
	const bool depthState = stencilPass || scene || (g.sceneDepth && kind != Pack && kind != Decode);
	vk::PipelineColorBlendAttachmentState attachment = blendState(kind == Pack ? 0 : blend);
	if (kind == Pack)
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR;
	if (key & PageAlpha)
		attachment.colorWriteMask &= ~vk::ColorComponentFlags(vk::ColorComponentFlagBits::eA);
	const float constants = blend == 1 ? 0.5f : blend == VeilBlend ? 0.15f : 0.25f;
	vk::PipelineColorBlendStateCreateInfo blendInfo(vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eCopy, attachment,
			{ constants, constants, constants, constants });
	vk::DynamicState dynamics[2] = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
	vk::PipelineDynamicStateCreateInfo dynamic(vk::PipelineDynamicStateCreateFlags(), dynamics);
	vk::GraphicsPipelineCreateInfo info(vk::PipelineCreateFlags(), stages, &input, &assembly, nullptr, &viewport, &raster, &multisample,
			depthState ? &depthStencil : nullptr, &blendInfo, &dynamic, worldKind ? *g.worldLayout : *g.layout,
			scene ? *g.scenePasses[0] : kind == Pack ? *g.packPass : kind == Decode ? *g.atlasPass : stencilPass ? *g.drawPass :
			*g.colorPass, 0);
	return std::move(g.dev.device.createGraphicsPipelineUnique(*g.pipelineCache, info).value);
}

// the pipeline for a key: built ahead of use if it was on the list; if not, here, and on the list from now on
vk::Pipeline pipeline(uint32_t key)
{
	{
		std::lock_guard<std::mutex> hold(g.pipelineLock);
		auto found = g.pipelines.find(key);
		if (found != g.pipelines.end())
			return *found->second;
	}
	vk::UniquePipeline made = buildPipeline(key);
	std::lock_guard<std::mutex> hold(g.pipelineLock);
	g.pipelineKeysNew = true;
	vk::UniquePipeline& kept = g.pipelines[key];
	if (!kept)
		kept = std::move(made);
	return *kept;
}

constexpr const char *PipelineKeysFile = "pipelines.keys", *PipelineCacheFile = "pipelines.cache";

// the driver's cache from the card (anything it can't use it ignores), and the list built on a thread of its own
void startPipelines()
{
	std::vector<char> cached;
	if (FILE *f = std::fopen(PipelineCacheFile, "rb"))
	{
		char chunk[65536];
		size_t n;
		while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
			cached.insert(cached.end(), chunk, chunk + n);
		std::fclose(f);
	}
	g.pipelineCache = g.dev.device.createPipelineCacheUnique(vk::PipelineCacheCreateInfo(vk::PipelineCacheCreateFlags(), cached.size(), cached.data()));
	std::vector<uint32_t> keys;
	if (FILE *f = std::fopen(PipelineKeysFile, "r"))
	{
		unsigned key;
		while (std::fscanf(f, "%x", &key) == 1)
			keys.push_back(key);
		std::fclose(f);
	}
	DS_LOG("recomp gpu: %zu pipelines to build ahead of use; the driver's cache %zu bytes", keys.size(), cached.size());
	g.pipelineWarmerStop = false;
	g.pipelineWarmer = std::thread([keys] {
		const auto from = std::chrono::steady_clock::now();
		for (uint32_t key : keys)
		{
			if (g.pipelineWarmerStop)
				return;
			{
				std::lock_guard<std::mutex> hold(g.pipelineLock);
				if (g.pipelines.count(key) != 0)
					continue;
			}
			// (one no module draws any more; one of shaders this module didn't give)
			const uint32_t keyKind = key & 15;
			if (keyKind == KindRetired || ((keyKind == WorldOwnSolid || keyKind == WorldOwnEdge) && !g.worldOwnVert))
				continue;
			vk::UniquePipeline made = buildPipeline(key);
			std::lock_guard<std::mutex> hold(g.pipelineLock);
			vk::UniquePipeline& kept = g.pipelines[key];
			if (!kept)
				kept = std::move(made);
		}
		DS_LOG("recomp gpu: pipelines built ahead of use in %.0f ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count());
	});
}

// the thread stopped; the driver's cache to the card and, if new ones were drawn, the list (each through a new file: never half
// written). Before the GPU's objects go.
void finishPipelines()
{
	g.pipelineWarmerStop = true;
	if (g.pipelineWarmer.joinable())
		g.pipelineWarmer.join();
	if (!g.pipelineCache)
		return;
	const std::vector<uint8_t> data = g.dev.device.getPipelineCacheData(*g.pipelineCache);
	if (FILE *f = std::fopen("pipelines.cache.new", "wb"))
	{
		const bool written = std::fwrite(data.data(), 1, data.size(), f) == data.size();
		std::fclose(f);
		if (written)
			std::rename("pipelines.cache.new", PipelineCacheFile);
	}
	if (g.pipelineKeysNew)
	{
		std::vector<uint32_t> keys;
		for (const auto& kept : g.pipelines)
			keys.push_back(kept.first);
		std::sort(keys.begin(), keys.end());
		if (FILE *f = std::fopen("pipelines.keys.new", "w"))
		{
			for (uint32_t key : keys)
				std::fprintf(f, "%08x\n", key);
			std::fclose(f);
			std::rename("pipelines.keys.new", PipelineKeysFile);
		}
		DS_LOG("recomp gpu: %zu pipelines on the list now", keys.size());
	}
	g.pipelineCache.reset();
}

void flushRun()
{
	if (!g.runOpen)
		return;
	g.runOpen = false;
	if (g.runCount == 0)
		return;
	vk::CommandBuffer cmd = *g.slots[g.current].cmd;
	cmd.setScissor(0, g.runScissor);
	if (g.runTextured && g.runSemi)
	{
		// the opaque texels, then the semi-transparent ones blended
		Push p = g.runPush;
		p.flags[1] = 1;
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(g.runKey & ~(15u << 4)));
		cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
		cmd.draw(g.runCount, 1, g.runFirst, 0);
		g.draws += 2;
		p.flags[1] = 2;
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(g.runKey));
		cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
		cmd.draw(g.runCount, 1, g.runFirst, 0);
		return;
	}
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(g.runKey));
	cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(g.runPush), &g.runPush);
	cmd.draw(g.runCount, 1, g.runFirst, 0);
	g.draws++;
}

// the pass's deferred primitives drawn (scene depth): opaque 3D nearest first, background 2D, the rest in order
void flushDeferred()
{
	if (g.passPrims.empty())
		return;
	vk::CommandBuffer cmd = *g.slots[g.current].cmd;
	std::vector<uint32_t> order;
	order.reserve(g.passPrims.size());
	for (uint32_t i = 0; i < g.passPrims.size(); i++)
		if (g.passPrims[i].step == 0)
			order.push_back(i);
	std::stable_sort(order.begin(), order.end(), [](uint32_t a, uint32_t b) { return g.passPrims[a].depth < g.passPrims[b].depth; });
	for (uint8_t step : { (uint8_t)1, (uint8_t)2 })
		for (uint32_t i = 0; i < g.passPrims.size(); i++)
			if (g.passPrims[i].step == step)
				order.push_back(i);
	uint32_t key = ~0u;
	vk::Rect2D scissor;
	bool first = true;
	for (uint32_t i : order)
	{
		const auto& d = g.passPrims[i];
		if (first || d.scissor != scissor)
			cmd.setScissor(0, d.scissor), scissor = d.scissor;
		first = false;
		if (d.twoPasses)
		{
			// the opaque texels, then the semi-transparent ones blended
			Push p = d.push;
			p.flags[1] = 1;
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key & ~(15u << 4)));
			cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
			cmd.draw(d.count, 1, d.first, 0);
			p.flags[1] = 2;
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key));
			cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
			cmd.draw(d.count, 1, d.first, 0);
			key = d.key;
			g.draws += 2;
			continue;
		}
		if (d.key != key)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key)), key = d.key;
		cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(d.push), &d.push);
		cmd.draw(d.count, 1, d.first, 0);
		g.draws++;
	}
	g.passPrims.clear();
	g.deferredAny3D = false;
	g.lastDepth = 1.f;
}

void endPass();

// a drawing pass over (at least) the VRAM rect x0, y0 - x1, y1 (exclusive): on a tiled GPU a pass loads and stores all of
// its render area. withArea: the draw area too (primitives after it share the pass); an upload's or a fill's pass is
// its own rect only (with the draw area all of VRAM, a few texels uploaded loaded and stored all of it, and the stencil)
void beginPass(int x0, int y0, int x1, int y1, bool withArea = true)
{
	if (g.inPass && x0 >= g.passArea[0] && y0 >= g.passArea[1] && x1 <= g.passArea[2] && y1 <= g.passArea[3])
		return;
	endPass();
	if (withArea)
	{
		x0 = std::min(x0, g.area[0]), y0 = std::min(y0, g.area[1]);
		x1 = std::max(x1, g.area[2] + 1), y1 = std::max(y1, g.area[3] + 1);
	}
	x0 = std::max(0, x0), y0 = std::max(0, y0), x1 = std::min(VramW, x1), y1 = std::min(VramH, y1);
	g.passArea[0] = x0, g.passArea[1] = y0, g.passArea[2] = x1, g.passArea[3] = y1;
	Slot& s = slot();
	const int S = g.scale;
	const vk::Extent2D extent(VramW * S, VramH * S);
	const vk::Rect2D area({ x0 * S, y0 * S }, vk::Extent2D((uint32_t)(x1 - x0) * S, (uint32_t)(y1 - y0) * S));
	g.passStamp = stampOpen(s, StampPass);
	const vk::ClearValue clears[2] = { vk::ClearColorValue(), vk::ClearDepthStencilValue(1.f, 0) };
	s.cmd->beginRenderPass(vk::RenderPassBeginInfo(g.stencilOn ? *g.drawPass : *g.colorPass,
			g.stencilOn ? *g.drawFramebuffer : *g.colorFramebuffer, area, g.sceneDepth ? 2u : 0u, clears), vk::SubpassContents::eInline);
	g.passes++;
	s.cmd->setViewport(0, vk::Viewport(0, 0, (float)extent.width, (float)extent.height, 0, 1));
	vk::DeviceSize offset = 0;
	s.cmd->bindVertexBuffers(0, *s.vertices, offset);
	s.cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
	g.inPass = true;
}

void endPass()
{
	flushRun();
	flushDeferred();
	if (!g.inPass)
		return;
	vk::CommandBuffer cmd = *g.slots[g.current].cmd;
	cmd.endRenderPass();
	barrier(cmd);
	stampClose(g.slots[g.current], g.passStamp);
	g.passStamp = ~0u;
	g.inPass = false;
}

void flushScene(int shown);

void submit()
{
	flushScene(-2);
	endPass();
	Slot& s = g.slots[g.current];
	if (!s.recording)
		return;
	barrier(*s.cmd);
	if (s.timing)
	{
		s.cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, stampPool(s), g.current * StampsPerSlot + 1);
		s.timed = true;
	}
	s.cmd->end();
	g.dev.device.resetFences(*s.fence);
	const vk::CommandBuffer cmd = *s.cmd;
	hostVideo().submit(&cmd, 1, *s.fence);
	s.recording = false;
	g.current = (g.current + 1) % Slots;
}

// the tiles of a VRAM rect (clamped), as a call per tile
template<typename F> void tiles(int x, int y, int w, int h, F f)
{
	const int x0 = std::max(x, 0), y0 = std::max(y, 0), x1 = std::min(x + w, VramW), y1 = std::min(y + h, VramH);
	if (x0 >= x1 || y0 >= y1)
		return;
	for (int ty = y0 >> 4; ty <= (y1 - 1) >> 4; ty++)
		for (int tx = x0 >> 5; tx <= (x1 - 1) >> 5; tx++)
			f(ty * 32 + tx);
}

bool anyDirty()
{
	uint64_t any = 0;
	for (uint64_t d : g.dirty)
		any |= d;
	return any != 0;
}

struct PageArea
{
	int px, py, pw, cx, cy, cw;
};
PageArea pageArea(uint32_t depth, uint16_t page, uint16_t clutX, uint16_t clutY);
// the texture cache's slots reading VRAM in a rect (not wrapping) out of date
void invalidateTextures(int x, int y, int w, int h, bool smallUpload = false)
{
	uint64_t hit[SlotWords] = {};
	tiles(x, y, w, h, [&](int t) {
		for (int k = 0; k < SlotWords; k++)
			hit[k] |= g.tileSlots[t][k];
	});
	for (int k = 0; k < SlotWords; k++)
		for (; hit[k] != 0; hit[k] &= hit[k] - 1)
		{
			auto& t = g.texSlots[64 * k + __builtin_ctzll(hit[k])];
			// (several of a world's pages can be one page through different CLUTs: an animated texture is shown through one, and each decode is a pass of this GPU's and a transfer of its own)
			if (t.kept && !t.usedElsewhere && smallUpload)
			{
				const PageArea area = pageArea(t.depth, t.page, t.clutX, t.clutY);
				const int perWord = t.depth == 0 ? 2 : t.depth == 1 ? 1 : 0;
				if (x >= area.px && x + w <= area.px + area.pw && y >= area.py && y + h <= area.py + 256 && !(area.cw > 0 && y <= area.cy && area.cy < y + h && x < area.cx + area.cw && area.cx < x + w))
				{
					const int u0 = (x - area.px) << perWord, u1 = ((x + w - area.px) << perWord) - 1, v0 = y - area.py, v1 = y + h - area.py - 1;
					const uint16_t columns = (uint16_t)(((2u << (std::min(u1, 255) >> 4)) - 1u) & ~((1u << (u0 >> 4)) - 1u));
					bool shown = false;
					for (int r = v0 >> 4; r <= (std::min(v1, 255) >> 4) && !shown; r++)
						shown = (t.usedCells[r] & columns) != 0;
					if (!shown)
						continue;
				}
			}
			// the part of it that changed, if it was only ever a small upload in its texels: a texture animated needn't decode the whole page each time
			const PageArea a = pageArea(t.depth, t.page, t.clutX, t.clutY);
			const int shift = t.depth == 0 ? 2 : t.depth == 1 ? 1 : 0;
			const bool inTexels = x >= a.px && x + w <= a.px + a.pw && y >= a.py && y + h <= a.py + 256;
			const bool inClut = a.cw > 0 && y <= a.cy && a.cy < y + h && x < a.cx + a.cw && a.cx < x + w;
			if (smallUpload && inTexels && !inClut && (t.valid || t.changed[2] >= 0))
			{
				const int u0 = (x - a.px) << shift, u1 = ((x + w - a.px) << shift) - 1, v0 = y - a.py, v1 = y + h - a.py - 1;
				const bool first = t.valid;
				t.changed[0] = (int16_t)(first ? u0 : std::min<int>(t.changed[0], u0)), t.changed[1] = (int16_t)(first ? v0 : std::min<int>(t.changed[1], v0));
				t.changed[2] = (int16_t)(first ? u1 : std::max<int>(t.changed[2], u1)), t.changed[3] = (int16_t)(first ? v1 : std::max<int>(t.changed[3], v1));
			}
			else
				t.changed[2] = t.changed[3] = -1;
			t.valid = false, t.changedFrame = g.texFrame;
			t.smallChange = smallUpload;
		}
}

void markDirty(int x, int y, int w, int h)
{
	invalidateTextures(x, y, w, h);
	tiles(x, y, w, h, [](int t) { g.dirty[t >> 6] |= 1ull << (t & 63); g.gpuNewer[t >> 6] |= 1ull << (t & 63); });
	g.cpuStale = true;
}

bool anyCpuNewer()
{
	uint64_t any = 0;
	for (uint64_t d : g.cpuNewer)
		any |= d;
	return any != 0;
}

void flushUploads();

bool overlapsDirty(int x, int y, int w, int h)
{
	bool hit = false;
	if (anyDirty())
		tiles(x, y, w, h, [&](int t) { hit = hit || (g.dirty[t >> 6] >> (t & 63)) & 1; });
	return hit;
}

// NDC of a VRAM rect (the targets' viewports cover all of VRAM)
void ndc(float out[4], int x, int y, int w, int h)
{
	out[0] = x / 512.f - 1.f, out[1] = y / 256.f - 1.f, out[2] = (x + w) / 512.f - 1.f, out[3] = (y + h) / 256.f - 1.f;
}

// the scaled image given what the raw mirror holds newer in a VRAM rect (not wrapping): the tiles uploaded since, blitted
// from the mirror in one pass
void refreshScaled(int x, int y, int w, int h)
{
	uint64_t any = 0;
	for (uint64_t r : g.rawNewer)
		any |= r;
	if (!any)
		return;
	uint64_t mask[16] = {};
	bool hit = false;
	tiles(x, y, w, h, [&](int t) {
		if ((g.rawNewer[t >> 6] >> (t & 63)) & 1)
			mask[t >> 6] |= 1ull << (t & 63), hit = true;
	});
	if (!hit)
		return;
	struct Run { int x, y, w, h; };
	std::vector<Run> runs;
	int x0 = VramW, y0 = VramH, x1 = 0, y1 = 0;
	for (int ty = 0; ty < 32; ty++)
		for (int tx = 0; tx < 32;)
		{
			if (!((mask[(ty * 32 + tx) >> 6] >> ((ty * 32 + tx) & 63)) & 1))
			{
				tx++;
				continue;
			}
			int e = tx;
			while (e < 32 && ((mask[(ty * 32 + e) >> 6] >> ((ty * 32 + e) & 63)) & 1))
				e++;
			runs.push_back({ tx * 32, ty * 16, (e - tx) * 32, 16 });
			x0 = std::min(x0, tx * 32), y0 = std::min(y0, ty * 16), x1 = std::max(x1, e * 32), y1 = std::max(y1, ty * 16 + 16);
			tx = e;
		}
	for (int i = 0; i < 16; i++)
		g.rawNewer[i] &= ~mask[i];
	beginPass(x0, y0, x1, y1, false);
	Slot *s = &g.slots[g.current];
	s->cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(Blit | (g.stencilOn ? (1u << 12) : 0u)));
	for (const Run& r : runs)
	{
		s->cmd->setScissor(0, vk::Rect2D({ r.x * g.scale, r.y * g.scale }, vk::Extent2D((uint32_t)r.w * g.scale, (uint32_t)r.h * g.scale)));
		Push p {};
		p.flags[2] = g.scale;
		ndc(p.rect, r.x, r.y, r.w, r.h);
		s->cmd->pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
		s->cmd->draw(4, 1, 0, 0);
	}
	endPass();
}


// the raw mirror brought up to date with what was drawn: all of it, or (area) only the drawn tiles in a VRAM rect (not
// wrapping), the rest left dirty (a primitive packs what it reads, not everything drawn)
void pack(const int *area = nullptr)
{
	if (anyCpuNewer())
		flushUploads();
	if (!anyDirty())
		return;
	// the dirty tiles' bounds (the rest packs to what it already is)
	int tx0 = 32, ty0 = 32, tx1 = -1, ty1 = -1;
	const int ax0 = area ? std::max(area[0], 0) / 32 : 0, ay0 = area ? std::max(area[1], 0) / 16 : 0;
	const int ax1 = area ? std::min(area[0] + area[2] - 1, VramW - 1) / 32 : 31, ay1 = area ? std::min(area[1] + area[3] - 1, VramH - 1) / 16 : 31;
	for (int ty = ay0; ty <= ay1; ty++)
		for (int tx = ax0; tx <= ax1; tx++)
		{
			const int t = ty * 32 + tx;
			if ((g.dirty[t >> 6] >> (t & 63)) & 1)
				tx0 = std::min(tx0, tx), ty0 = std::min(ty0, ty), tx1 = std::max(tx1, tx), ty1 = std::max(ty1, ty);
		}
	if (tx1 < 0)
		return;
	endPass();
	Slot& s = slot();
	const int x = tx0 * 32, y = ty0 * 16, w = (tx1 - tx0 + 1) * 32, h = (ty1 - ty0 + 1) * 16;
	g.packs++;
	const uint32_t stamp = stampOpen(s, StampPack);
	// the render area the rect packed: a tiled GPU loads and stores all of a pass's render area (the whole mirror before)
	s.cmd->beginRenderPass(vk::RenderPassBeginInfo(*g.packPass, *g.packFramebuffer, vk::Rect2D({ x, y }, vk::Extent2D((uint32_t)w, (uint32_t)h))),
			vk::SubpassContents::eInline);
	s.cmd->setViewport(0, vk::Viewport(0, 0, (float)VramW, (float)VramH, 0, 1));
	s.cmd->setScissor(0, vk::Rect2D({ x, y }, vk::Extent2D((uint32_t)w, (uint32_t)h)));
	s.cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(Pack));
	s.cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
	Push p {};
	p.flags[2] = g.scale;
	ndc(p.rect, x, y, w, h);
	s.cmd->pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
	s.cmd->draw(4, 1, 0, 0);
	s.cmd->endRenderPass();
	barrier(*s.cmd);
	stampClose(s, stamp);
	for (int ty = ty0; ty <= ty1; ty++)
		for (int tx = tx0; tx <= tx1; tx++)
			g.dirty[(ty * 32 + tx) >> 6] &= ~(1ull << ((ty * 32 + tx) & 63));
}

vk::Rect2D scissor()
{
	const int x0 = std::clamp(g.area[0], 0, VramW - 1), y0 = std::clamp(g.area[1], 0, VramH - 1);
	const int x1 = std::clamp(g.area[2], 0, VramW - 1), y1 = std::clamp(g.area[3], 0, VramH - 1);
	if (x1 < x0 || y1 < y0)
		return vk::Rect2D({ 0, 0 }, { 0, 0 });
	return vk::Rect2D({ x0 * g.scale, y0 * g.scale }, vk::Extent2D((uint32_t)(x1 - x0 + 1) * g.scale, (uint32_t)(y1 - y0 + 1) * g.scale));
}

// the stencil from the game's first use of the mask bit (cleared: the mask bits drawn before it are not kept there)
void enableStencil()
{
	if (g.stencilOn || !(g.maskSet || g.maskCheck))
		return;
	// (scenes keep the mask bit in their own stencil when their depth format has one)
	void sceneDisable();
	if (!g.sceneStencil)
		sceneDisable();
	endPass();
	g.stencilOn = true;
	beginPass(0, 0, VramW, VramH);
	vk::ClearAttachment clear(vk::ImageAspectFlagBits::eStencil, 0, vk::ClearDepthStencilValue(1.f, 0));
	g.slots[g.current].cmd->clearAttachments(clear, vk::ClearRect(vk::Rect2D({ 0, 0 },
			vk::Extent2D((uint32_t)(VramW * g.scale), (uint32_t)(VramH * g.scale))), 0, 1));
	DS_LOG("recomp module: the game uses the mask bit; VRAM's stencil on");
}

uint32_t stencilBits()
{
	if (!g.stencilOn)
		return 0;
	return (((g.maskCheck ? 2u : 0u) | (g.maskSet ? 1u : 0u)) << 8) | (1u << 12);
}

// ---- the native scene ----

// Its depth is reversed: 1 at the near limit, falling towards 0 far away, so the most compared as greater. Near / z in a 32-bit float keeps its precision at any distance; the
// usual 1 - near / z puts all that's more than a few hundred off in the last thousandth below 1, where a float has no room, and where the interpolation's error is
// wider than the gap between a road and the ground under it: distant surfaces taking turns in front, a flicker of the wrong one.
constexpr float SceneNear = 16.f;
inline float scenePerspectiveDepth(float z) { return z > SceneNear ? SceneNear / z : 1.f; }
inline float sceneFlatDepth(float z) { return std::clamp(1.f - z * (1.f / 65536.f), 0.f, 1.f); }

bool overlaps(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh)
{
	return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}
bool sceneReadsOverlap(int x, int y, int w, int h)
{
	bool hit = false;
	tiles(x, y, w, h, [&](int t) { hit = hit || ((g.sceneReads[t >> 6] >> (t & 63)) & 1); });
	return hit;
}
void sceneMarkReads(int x, int y, int w, int h)
{
	tiles(x, y, w, h, [](int t) { g.sceneReads[t >> 6] |= 1ull << (t & 63); });
}
// a VRAM x or y of a scene's area in its pixels
int sceneX(const State::Scene& sc, double x) { return (int)std::lround((x - sc.ax) * g.sceneW / sc.aw); }
int sceneY(const State::Scene& sc, double y) { return (int)std::lround((y - sc.ay) * g.sceneH / sc.ah); }

// the recorded scene draws, a render pass for each run of them into one scene. shown: the scene about to be shown (its
// depth is done with, not stored), -2 none
// a scene's recorded draws, in an open pass with its viewport, vertices and descriptors bound
void issueSceneDraws(vk::CommandBuffer cmd, const State::SceneDraw *draws, size_t count, uint32_t base, vk::Buffer vertices = nullptr)
{
		uint32_t key = ~0u;
		bool scissorSet = false;
		vk::Rect2D scissor;
		bool inWorld = false;
		int pushed = -1, source = 2;
		for (size_t k = 0; k < count; k++)
		{
			const State::SceneDraw& d = draws[k];
			if (d.world >= 0)
			{
				if (!inWorld)
				{
					inWorld = true;
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.worldLayout, 0, g.set, nullptr);
					key = ~0u;
					source = -1;
				}
				if (d.source != source)
				{
					source = d.source;
					const vk::DeviceSize zero = 0;
					cmd.bindVertexBuffers(0, source == 0 ? *g.worldBuffer : source == 1 ? *g.skyBuffer : vertices, zero);
				}
				if (!scissorSet || d.scissor != scissor)
					cmd.setScissor(0, d.scissor), scissor = d.scissor, scissorSet = true;
				if (d.key != key)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key)), key = d.key;
				if (d.world != pushed)
				{
					pushed = d.world;
					cmd.pushConstants(*g.worldLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, g.worldPushSize, &g.worldPushes[(size_t)d.world]);
				}
				cmd.draw(d.count, 1, d.first, 0);
				g.draws++;
				continue;
			}
			if (inWorld)
			{
				inWorld = false;
				const vk::DeviceSize zero = 0;
				cmd.bindVertexBuffers(0, vertices, zero);
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
				key = ~0u;
			}
			if (d.count == 0)
			{
				const uint16_t c = d.clearColor;
				const vk::ClearAttachment cl[2] = {
					{ vk::ImageAspectFlagBits::eColor, 0, vk::ClearColorValue(std::array<float, 4> { (c & 31) / 31.f, ((c >> 5) & 31) / 31.f,
							((c >> 10) & 31) / 31.f, 0.f }) },
					{ g.sceneStencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil :
							vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth), 0, vk::ClearDepthStencilValue(0.f, 0) },
				};
				if (d.scissor.extent.width > 0 && d.scissor.extent.height > 0)
					cmd.clearAttachments(cl, vk::ClearRect(d.scissor, 0, 1));
				continue;
			}
			if (!scissorSet || d.scissor != scissor)
				cmd.setScissor(0, d.scissor), scissor = d.scissor, scissorSet = true;
			if (d.twoPasses)
			{
				// the opaque texels, then the semi-transparent ones blended
				Push p = d.push;
				p.flags[1] = 1;
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key & ~(15u << 4)));
				cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
				cmd.draw(d.count, 1, d.first + base, 0);
				p.flags[1] = 2;
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key));
				cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
				cmd.draw(d.count, 1, d.first + base, 0);
				key = d.key;
				g.draws += 2;
				continue;
			}
			if (d.key != key)
				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(d.key)), key = d.key;
			cmd.pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(d.push), &d.push);
			cmd.draw(d.count, 1, d.first + base, 0);
			g.draws++;
		}
}

// A frame that went straight into its page isn't in its scene's image. For whatever comes to want it there (a frame that isn't whole, drawn over what's there, or shown
// from there; VRAM taking it): back from the page, which still holds it, every such scene being one of the last few frames.
void restoreScenes()
{
	bool any = false;
	for (const State::Scene& sc : g.scenes)
		any = any || sc.pageHolding >= 0;
	if (!any)
		return;
	endPass();
	vk::CommandBuffer cmd = *slot().cmd;
	const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	barrier(cmd);
	for (State::Scene& sc : g.scenes)
		if (sc.pageHolding >= 0)
		{
			cmd.copyImage(hostVideo().pageImage((uint32_t)sc.pageHolding), vk::ImageLayout::eGeneral, *sc.color.image, vk::ImageLayout::eGeneral,
					vk::ImageCopy(layers, { 0, 0, 0 }, layers, { 0, 0, 0 }, { g.sceneW, g.sceneH, 1 }));
			sc.pageHolding = -1;
		}
	barrier(cmd);
}

void flushScene(int shown)
{
	if (g.sceneDraws.empty())
		return;
	if (anyCpuNewer())
		flushUploads();		// (a submit on the way draws them)
	if (g.sceneDraws.empty())
		return;
	endPass();
	Slot& s = slot();
	vk::CommandBuffer cmd = *s.cmd;
	std::vector<State::SceneDraw> draws;
	draws.swap(g.sceneDraws);
	std::fill(std::begin(g.sceneReads), std::end(g.sceneReads), 0);
	const uint32_t stamp = stampOpen(s, StampScene);
	const vk::Rect2D all({ 0, 0 }, { g.sceneW, g.sceneH });
	const bool straight = g.framePage >= 0 && shown >= 0 && !g.scenes[shown].depthStarted &&
			std::all_of(draws.begin(), draws.end(), [&](const State::SceneDraw& d) { return d.scene == shown; });
	if (straight)
		for (State::SceneDraw& d : draws)
			d.key |= PageAlpha;
	for (size_t i = 0; i < draws.size();)
	{
		State::Scene& sc = g.scenes[draws[i].scene];
		size_t e = i;
		while (e < draws.size() && draws[e].scene == draws[i].scene)
			e++;
		// (this GPU reads every tile in to begin with otherwise: most of a millisecond and memory's bandwidth, for pixels about to be drawn over)
		// (and a whole frame, ended by its being shown, writes out no depth for the next one to throw away: it clears its own)
		const bool keepDepth = draws[i].scene != shown && !(g.sceneWhole && shown >= 0);
		const int variant = (g.sceneWhole && !sc.depthStarted ? 4 : 0) | (sc.depthStarted ? 2 : 0) | (keepDepth ? 1 : 0);
		// (the page: one pass over the frame instead of two, the second only to copy it)
		vk::Framebuffer target = *sc.framebuffer;
		if (straight)
		{
			vk::UniqueFramebuffer& there = sc.onPage[g.framePage];
			if (!there)
			{
				const vk::ImageView views[2] = { hostVideo().pageView((uint32_t)g.framePage), *sc.depth.view };
				there = g.dev.device.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.scenePasses[0], views, g.sceneW, g.sceneH, 1));
			}
			target = *there;
			const vk::ImageMemoryBarrier toGeneral(vk::AccessFlags(), vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, hostVideo().pageImage((uint32_t)g.framePage), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::DependencyFlags(), nullptr, nullptr, toGeneral);
			sc.pageHolding = g.framePage;
			g.frameOnPage = true;
			g.framesOnPages++;
		}
		else if (sc.pageHolding >= 0)
		{
			// (cleared, what was in it is wanted no more)
			if (variant & 4)
				sc.pageHolding = -1;
			else
				restoreScenes();
		}
		const vk::ClearValue clears[2] = { vk::ClearColorValue(std::array<float, 4> { g.sceneClear[0], g.sceneClear[1], g.sceneClear[2], straight ? 1.f : 0.f }), vk::ClearDepthStencilValue(0.f, 0) };
		cmd.beginRenderPass(vk::RenderPassBeginInfo(*g.scenePasses[variant], target, all, 2, clears), vk::SubpassContents::eInline);
		sc.depthStarted = true;
		g.scenePassCount++;
		cmd.setViewport(0, vk::Viewport(0, 0, (float)g.sceneW, (float)g.sceneH, 0, 1));
		vk::DeviceSize offset = 0;
		cmd.bindVertexBuffers(0, *s.vertices, offset);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
		issueSceneDraws(cmd, draws.data() + i, e - i, 0, *s.vertices);
		cmd.endRenderPass();
		i = e;
	}
	g.worldPushes.clear();
	g.sceneWhole = false;
	barrier(cmd);
	stampClose(s, stamp);
}

// what VRAM holds newer than a scene (written by other means since it last took VRAM) into it, one blit: before the scene
// draws again, gives its drawing to VRAM, or is shown with drawing of its own
void sceneSync(State::Scene& sc)
{
	if (!sc.used || (!sc.vramAll && sc.vramRects.empty()))
		return;
	flushScene(-2);
	if (anyCpuNewer())
		flushUploads();
	if (sc.vramAll)
		refreshScaled(sc.ax, sc.ay, sc.aw, sc.ah);
	else
		for (const auto& r : sc.vramRects)
			refreshScaled(r[0], r[1], r[2] - r[0], r[3] - r[1]);
	endPass();
	Slot& s = slot();
	const int S = g.scale;
	const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	std::vector<vk::ImageBlit> regions;
	const auto add = [&](int x0, int y0, int x1, int y1) {
		const std::array<vk::Offset3D, 2> from { vk::Offset3D(x0 * S, y0 * S, 0), vk::Offset3D(x1 * S, y1 * S, 1) };
		const std::array<vk::Offset3D, 2> to { vk::Offset3D(sceneX(sc, x0), sceneY(sc, y0), 0), vk::Offset3D(sceneX(sc, x1), sceneY(sc, y1), 1) };
		regions.emplace_back(layers, from, layers, to);
	};
	if (sc.vramAll)
		add(sc.ax, sc.ay, sc.ax + sc.aw, sc.ay + sc.ah);
	else
		for (const auto& r : sc.vramRects)
			add(r[0], r[1], r[2], r[3]);
	if (sc.pageHolding >= 0 && !g.sceneWhole)
		restoreScenes();
	s.cmd->blitImage(*g.scaled.image, vk::ImageLayout::eGeneral, *sc.color.image, vk::ImageLayout::eGeneral, regions, vk::Filter::eLinear);
	barrier(*s.cmd);
	sc.vramAll = false;
	sc.vramRects.clear();
	g.sceneBlits++;
}

// the tiles of a VRAM rect inside a scene's area it drew into
void sceneMarkDrawn(State::Scene& sc, int x0, int y0, int x1, int y1)
{
	x0 = std::max(x0, sc.ax), y0 = std::max(y0, sc.ay), x1 = std::min(x1, sc.ax + sc.aw), y1 = std::min(y1, sc.ay + sc.ah);
	if (x0 >= x1 || y0 >= y1)
		return;
	tiles(x0, y0, x1 - x0, y1 - y0, [&](int t) { sc.drawn[t >> 6] |= 1ull << (t & 63); });
	sc.newer = true;
}

// what a scene drew into VRAM (the scaled image; the raw mirror takes it as drawn tiles), for what reads or writes VRAM
// there (why): its drawn tiles in all of its area, or (rect: x, y, w, h) in that rect, each clipped to its area
void sceneResolve(State::Scene& sc, const char *why, const int *rect = nullptr)
{
	if (!sc.used || !sc.newer)
		return;
	uint64_t mask[16] = {};
	bool any = false;
	if (rect == nullptr)
		for (int i = 0; i < 16; i++)
			mask[i] = sc.drawn[i], any = any || mask[i] != 0;
	else
	{
		const int x0 = std::max(rect[0], sc.ax), y0 = std::max(rect[1], sc.ay);
		const int x1 = std::min(rect[0] + rect[2], sc.ax + sc.aw), y1 = std::min(rect[1] + rect[3], sc.ay + sc.ah);
		if (x0 < x1 && y0 < y1)
			tiles(x0, y0, x1 - x0, y1 - y0, [&](int t) {
				if ((sc.drawn[t >> 6] >> (t & 63)) & 1)
					mask[t >> 6] |= 1ull << (t & 63), any = true;
			});
	}
	if (!any)
		return;
	flushScene(-2);
	// the scaled image current under the tiles first (a resolve writes parts of them)
	{
		const int x0 = rect ? std::max(rect[0], sc.ax) : sc.ax, y0 = rect ? std::max(rect[1], sc.ay) : sc.ay;
		const int x1 = rect ? std::min(rect[0] + rect[2], sc.ax + sc.aw) : sc.ax + sc.aw;
		const int y1 = rect ? std::min(rect[1] + rect[3], sc.ay + sc.ah) : sc.ay + sc.ah;
		refreshScaled(x0 & ~31, y0 & ~15, ((x1 + 31) & ~31) - (x0 & ~31), ((y1 + 15) & ~15) - (y0 & ~15));
	}
	endPass();
	Slot& s = slot();
	const int S = g.scale;
	const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	std::vector<vk::ImageBlit> regions;
	std::vector<std::array<int, 4>> rects;
	for (int ty = 0; ty < 32; ty++)
		for (int tx = 0; tx < 32;)
		{
			const int t = ty * 32 + tx;
			if (!((mask[t >> 6] >> (t & 63)) & 1))
			{
				tx++;
				continue;
			}
			int e = tx;
			while (e < 32 && ((mask[(ty * 32 + e) >> 6] >> ((ty * 32 + e) & 63)) & 1))
				e++;
			const int x0 = std::max(tx * 32, sc.ax), y0 = std::max(ty * 16, sc.ay);
			const int x1 = std::min(e * 32, sc.ax + sc.aw), y1 = std::min(ty * 16 + 16, sc.ay + sc.ah);
			if (x0 < x1 && y0 < y1)
			{
				const std::array<vk::Offset3D, 2> from { vk::Offset3D(sceneX(sc, x0), sceneY(sc, y0), 0), vk::Offset3D(sceneX(sc, x1), sceneY(sc, y1), 1) };
				const std::array<vk::Offset3D, 2> to { vk::Offset3D(x0 * S, y0 * S, 0), vk::Offset3D(x1 * S, y1 * S, 1) };
				regions.emplace_back(layers, from, layers, to);
				rects.push_back({ x0, y0, x1 - x0, y1 - y0 });
			}
			tx = e;
		}
	if (!regions.empty())
	{
		restoreScenes();
		s.cmd->blitImage(*sc.color.image, vk::ImageLayout::eGeneral, *g.scaled.image, vk::ImageLayout::eGeneral, regions, vk::Filter::eLinear);
		barrier(*s.cmd);
	}
	bool left = false;
	for (int i = 0; i < 16; i++)
		sc.drawn[i] &= ~mask[i], left = left || sc.drawn[i] != 0;
	sc.newer = left;
	g.sceneResolves++;
	for (const auto& r : rects)
		markDirty(r[0], r[1], r[2], r[3]);
	static int logged = 0;
	if (g.perf && logged < 12 && !rects.empty())
	{
		logged++;
		DS_LOG("recomp gpu: frame buffer %d,%d given to VRAM for %s (%zu runs of tiles, from %d,%d)", sc.ax, sc.ay, why, rects.size(),
				rects[0][0], rects[0][1]);
	}
}
// before VRAM's x, y, w, h is written by other means than a scene's drawing: the scenes' drawing there given to VRAM
// first (the write then lands as it is)
void sceneBeforeWrite(int x, int y, int w, int h, const char *why)
{
	const int rect[4] = { x, y, w, h };
	for (State::Scene& sc : g.scenes)
		if (sc.used && sc.newer && overlaps(x, y, w, h, sc.ax, sc.ay, sc.aw, sc.ah))
			sceneResolve(sc, why, rect);
}
void sceneResolveAll(const char *why)
{
	for (State::Scene& sc : g.scenes)
		sceneResolve(sc, why);
}

// VRAM's x, y, w, h written by other means than a scene's drawing (why): the scenes over it take it when next needed
void sceneAfterWrite(int x, int y, int w, int h, const char *why)
{
	for (State::Scene& sc : g.scenes)
	{
		if (!sc.used || sc.vramAll)
			continue;
		const int x0 = std::max(x, sc.ax), y0 = std::max(y, sc.ay), x1 = std::min(x + w, sc.ax + sc.aw), y1 = std::min(y + h, sc.ay + sc.ah);
		if (x0 >= x1 || y0 >= y1)
			continue;
		static int logged = 0;
		if (g.perf && logged < 12)
		{
			logged++;
			DS_LOG("recomp gpu: VRAM %d,%d %d x %d (frame buffer %d,%d) written by %s", x0, y0, x1 - x0, y1 - y0, sc.ax, sc.ay, why);
		}
		if (!sc.newer)
		{
			// VRAM holds all of it: taken whole when next needed
			sc.vramAll = true;
			sc.vramRects.clear();
			continue;
		}
		sc.vramRects.push_back({ x0, y0, x1, y1 });
		if (sc.vramRects.size() >= 32)
			sceneSync(sc);
	}
}

// the mask bit in play: every scene given to VRAM, frames drawn there from now on
void sceneDisable()
{
	if (!g.sceneOn)
		return;
	sceneResolveAll("the mask bit");
	for (State::Scene& sc : g.scenes)
		sc.used = false;
	g.sceneOn = false;
	DS_LOG("recomp gpu: the game uses the mask bit: its frames drawn in VRAM from now on");
}

// a scene for the VRAM area x, y, w, h (its images made the first time a slot is used), started from what VRAM holds there
int sceneCreate(int x, int y, int w, int h)
{
	// scenes over the area given to VRAM and dropped (a change of mode: the new one starts from what they held)
	for (State::Scene& sc : g.scenes)
		if (sc.used && overlaps(x, y, w, h, sc.ax, sc.ay, sc.aw, sc.ah))
			sceneResolve(sc, "a new frame buffer over it"), sc.used = false;
	int pick = -1;
	for (int i = 0; i < State::SceneCount && pick < 0; i++)
		if (!g.scenes[i].used)
			pick = i;
	if (pick < 0)
	{
		pick = 0;
		for (int i = 1; i < State::SceneCount; i++)
			if (g.scenes[i].lastUsed < g.scenes[pick].lastUsed)
				pick = i;
		sceneResolve(g.scenes[pick], "another frame buffer");
		g.scenes[pick].used = false;
	}
	State::Scene& sc = g.scenes[pick];
	if (!sc.color.image)
	{
		using U = vk::ImageUsageFlagBits;
		makeImage(sc.color, SceneFormat, g.sceneW, g.sceneH, U::eColorAttachment | U::eSampled | U::eTransferSrc | U::eTransferDst,
				vk::ImageAspectFlagBits::eColor);
		const vk::ImageAspectFlags depthAspect = g.sceneStencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil :
				vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth);
		makeImage(sc.depth, g.sceneDepthFormat, g.sceneW, g.sceneH, U::eDepthStencilAttachment, depthAspect);
		const vk::ImageView views[2] = { *sc.color.view, *sc.depth.view };
		sc.framebuffer = g.dev.device.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.scenePasses[0],
				views, g.sceneW, g.sceneH, 1));
		endPass();
		Slot& s = slot();
		const auto to = [&](vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout layout) {
			vk::ImageMemoryBarrier b(vk::AccessFlags(), vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
					vk::ImageLayout::eUndefined, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
					vk::ImageSubresourceRange(aspect, 0, 1, 0, 1));
			s.cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags(),
					nullptr, nullptr, b);
		};
		to(*sc.color.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*sc.depth.image, depthAspect, vk::ImageLayout::eDepthStencilAttachmentOptimal);
	}
	sc.used = true;
	sc.ax = x, sc.ay = y, sc.aw = w, sc.ah = h;
	sc.lastUsed = g.texFrame;
	sc.depthStarted = false;
	sc.any3D = false;
	sc.lastDepth = 1.f;
	sc.newer = false;
	sc.vramAll = true;		// started from what VRAM holds (at its first draw)
	sc.vramRects.clear();
	static int logged = 0;
	if (logged < 16)
	{
		logged++;
		DS_LOG("recomp gpu: VRAM %d,%d %d x %d (a frame buffer) drawn natively at %u x %u", x, y, w, h, g.sceneW, g.sceneH);
	}
	return pick;
}

// the scene the draw area is in (a draw area the size of the frame last shown starts one), -1: drawn in VRAM
int sceneForArea(int bx0, int by0, int bx1, int by1)
{
	// DEADSTATION_PERF: each draw area (and why) drawn in VRAM, once
	const auto refuse = [](const char *why) {
		static uint64_t seen[32];
		static int count = 0;
		const uint64_t key = (uint64_t)(g.area[0] & 0x3FF) | (uint64_t)(g.area[1] & 0x1FF) << 10 | (uint64_t)(g.area[2] & 0x7FF) << 19 |
				(uint64_t)(g.area[3] & 0x3FF) << 30 | (uint64_t)(why[0]) << 40;
		if (!g.perf || count >= 32 || std::find(seen, seen + count, key) != seen + count)
			return -1;
		seen[count++] = key;
		DS_LOG("recomp gpu: draw area %d,%d..%d,%d drawn in VRAM: %s (the frame shown %d x %d)", g.area[0], g.area[1], g.area[2],
				g.area[3], why, g.displayW, g.displayH);
		return -1;
	};
	if (!g.sceneOn)
		return -1;
	if (g.stencilOn && !g.sceneStencil)
		return refuse("the mask bit in play");
	if (g.displayW <= 0)
		return -1;
	const int x0 = g.area[0], y0 = g.area[1], x1 = std::min(g.area[2] + 1, VramW), y1 = std::min(g.area[3] + 1, VramH);
	if (x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0)
		return refuse("outside VRAM");
	for (int i = 0; i < State::SceneCount; i++)
	{
		const State::Scene& sc = g.scenes[i];
		if (sc.used && x0 >= sc.ax && y0 >= sc.ay && x1 <= sc.ax + sc.aw && y1 <= sc.ay + sc.ah)
			return i;
	}
	// what it draws (inside the draw area) at least three quarters in a scene's area: drawn there, clipped to it (a draw
	// area over more than a frame; full-screen rects that run a pixel or two past their frame)
	if (bx1 > bx0 && by1 > by0)
		for (int i = 0; i < State::SceneCount; i++)
		{
			const State::Scene& sc = g.scenes[i];
			if (!sc.used)
				continue;
			const int ix0 = std::max(bx0, sc.ax), iy0 = std::max(by0, sc.ay), ix1 = std::min(bx1, sc.ax + sc.aw), iy1 = std::min(by1, sc.ay + sc.ah);
			if (ix0 < ix1 && iy0 < iy1 && (int64_t)(ix1 - ix0) * (iy1 - iy0) * 4 >= (int64_t)(bx1 - bx0) * (by1 - by0) * 3)
				return i;
		}
	// inside a rect shown in the last eight frames: a scene for that rect
	for (const State::ShownRect& r : g.shownRects)
		if (r.used && g.texFrame - r.frame <= 8 && x0 >= r.x && y0 >= r.y && x1 <= r.x + r.w && y1 <= r.y + r.h)
			return sceneCreate(r.x, r.y, r.w, r.h);
	if (std::abs(x1 - x0 - g.displayW) > 8 || std::abs(y1 - y0 - g.displayH) > 4)
		return refuse("in no frame shown, not its size");
	return sceneCreate(x0, y0, x1 - x0, y1 - y0);
}

// the VRAM a cached page reads: its texels' columns, and its CLUT's
PageArea pageArea(uint32_t depth, uint16_t page, uint16_t clutX, uint16_t clutY)
{
	PageArea a;
	a.px = (page & 15) * 64, a.py = ((page >> 4) & 1) * 256;
	a.pw = std::min(depth == 0 ? 64 : depth == 1 ? 128 : 256, VramW - a.px);
	a.cx = clutX & (VramW - 1), a.cy = clutY & (VramH - 1);
	a.cw = depth < 2 ? std::min(depth == 0 ? 16 : 256, VramW - a.cx) : 0;
	return a;
}

// the texture cache's slot holding a page through a CLUT, current; -1: none yet (its decode queued, when a slot is free)
bool decodeNow(int index);

int cachedSlot(uint32_t depth, uint16_t page, uint16_t clutX, uint16_t clutY)
{
	if (!g.texCache)
		return -1;
	if (depth > 2)
		depth = 2;
	const uint32_t key = (page & 0x1Fu) | depth << 5 | (depth < 2 ? ((uint32_t)(clutX >> 4) & 63u) << 7 | ((uint32_t)clutY & 511u) << 13 : 0u);
	auto found = g.texSlotOf.find(key);
	if (found != g.texSlotOf.end())
	{
		auto& t = g.texSlots[found->second];
		t.usedFrame = g.texFrame;
		t.usedElsewhere = t.usedElsewhere || !g.takingForWorld;
		if (t.valid)
			return found->second;
		if (t.smallChange && decodeNow(found->second))
			return found->second;
		if (!t.pending)
		{
			t.pending = true;
			g.texPending.push_back(found->second);
		}
		return -1;
	}
	// a slot unused for two frames, the least recently used
	int index = -1;
	uint32_t oldest = UINT32_MAX;
	for (int i = 0; i < TexSlots; i++)
		if (!g.texSlots[i].pending && !g.texSlots[i].kept && g.texSlots[i].usedFrame < oldest)
			oldest = g.texSlots[i].usedFrame, index = i;
	if (index < 0 || (oldest + 2 > g.texFrame && !g.takingForWorld))
		return -1;
	auto& t = g.texSlots[index];
	if (t.key != ~0u)
		g.texSlotOf.erase(t.key);
	t.key = key, t.valid = false, t.pending = true, t.usedFrame = g.texFrame;
	t.changed[2] = t.changed[3] = -1;
	t.page = page, t.clutX = clutX, t.clutY = clutY, t.depth = depth;
	g.texSlotOf[key] = index;
	const uint64_t bit = 1ull << (index & 63);
	const int word = index >> 6;
	for (auto& tile : g.tileSlots)
		tile[word] &= ~bit;
	const PageArea a = pageArea(depth, page, clutX, clutY);
	tiles(a.px, a.py, a.pw, 256, [&](int tile) { g.tileSlots[tile][word] |= bit; });
	if (a.cw > 0)
		tiles(a.cx, a.cy, a.cw, 1, [&](int tile) { g.tileSlots[tile][word] |= bit; });
	g.texPending.push_back(index);
	return -1;
}

// what a cached page's texels are, read from the renderer's VRAM copy through the CLUT: bit 0 transparent (0x0000),
// 1 opaque (STP 0), 2 semi-transparent (STP 1); 7 when not known (VRAM there drawn by the GPU since the copy was read
// back)
uint8_t texelClasses(uint32_t depth, uint16_t page, uint16_t clutX, uint16_t clutY, uint16_t *clearCells, bool uploadedOnly = false)
{
	// (clearCells: 16 rows of 16 bits: the page's 16 x 16 texel cells with a transparent texel in them)
	std::fill(clearCells, clearCells + 16, 0xFFFF);
	const PageArea a = pageArea(depth, page, clutX, clutY);
	bool drawn = false;
	const auto gpu = [&](int t) { drawn = drawn || ((g.gpuNewer[t >> 6] >> (t & 63)) & 1); };
	tiles(a.px, a.py, a.pw, 256, gpu);
	if (a.cw > 0)
		tiles(a.cx, a.cy, a.cw, 1, gpu);
	if ((drawn && !uploadedOnly) || g.mirror == nullptr)
		return 7;
	auto kind = [](uint16_t t) -> uint8_t { return t == 0 ? 1 : (t & 0x8000) ? 4 : 2; };
	uint8_t clut[256] = {};
	if (depth < 2)
		for (int i = 0; i < (depth == 0 ? 16 : 256); i++)
			clut[i] = kind(g.mirror[a.cy * VramW + ((a.cx + i) & (VramW - 1))]);
	std::fill(clearCells, clearCells + 16, 0);
	uint8_t cls = 0;
	for (int v = 0; v < 256; v++)
	{
		const uint16_t *row = &g.mirror[(a.py + v) * VramW];
		uint16_t &cells = clearCells[v >> 4];
		const auto note = [&](int u, uint8_t c) {
			cls |= c;
			if (c == 1)
				cells |= (uint16_t)(1u << (u >> 4));
		};
		if (depth == 0)
			for (int x = 0; x < 64; x++)
			{
				const uint16_t w = row[(a.px + x) & (VramW - 1)];
				note(4 * x, clut[w & 15]), note(4 * x + 1, clut[(w >> 4) & 15]), note(4 * x + 2, clut[(w >> 8) & 15]), note(4 * x + 3, clut[w >> 12]);
			}
		else if (depth == 1)
			for (int x = 0; x < 128; x++)
			{
				const uint16_t w = row[(a.px + x) & (VramW - 1)];
				note(2 * x, clut[w & 255]), note(2 * x + 1, clut[w >> 8]);
			}
		else
			for (int x = 0; x < 256; x++)
				note(x, kind(row[(a.px + x) & (VramW - 1)]));
	}
	return cls;
}

// the queued pages decoded into their slots, in one pass over the atlas (at the frame's end); a page whose VRAM was drawn
// and not packed waits (queued again when next used)
void decodeSlots(const std::vector<int>& ready);

// a page changed by a small upload (a texture animated, a few texels streamed in), used this frame: decoded again at once,
// so its polygons don't fall to the slow path, each frame, for what's a few texels of it
void decodeForced(int index);
bool decodeNow(int index)
{
	if (g.quickDecodes >= DecodesPerFrame)
		return false;
	g.quickDecodes++;
	decodeForced(index);
	return true;
}
void decodeForced(int index)
{
	auto& t = g.texSlots[index];
	if (anyCpuNewer())
		flushUploads();
	const PageArea a = pageArea(t.depth, t.page, t.clutX, t.clutY);
	const int texels[4] = { a.px, a.py, a.pw, 256 }, clut[4] = { a.cx, a.cy, a.cw, 1 };
	if (overlapsDirty(a.px, a.py, a.pw, 256))
		pack(texels);
	if (a.cw > 0 && overlapsDirty(a.cx, a.cy, a.cw, 1))
		pack(clut);
	decodeSlots({ index });
}

// A world's pages that changed (the game's animated textures), all in one pass: one to a page is a whole pass of this GPU's for each, and several pages use each texel changed, by their CLUTs
void decodeWorldPages()
{
	static std::vector<int> ready;
	ready.clear();
	for (int index : g.worldSlots)
		if (!g.texSlots[index].valid)
		{
			auto& t = g.texSlots[index];
			if (ready.empty() && anyCpuNewer())
				flushUploads();
			const PageArea a = pageArea(t.depth, t.page, t.clutX, t.clutY);
			const int texels[4] = { a.px, a.py, a.pw, 256 }, clut[4] = { a.cx, a.cy, a.cw, 1 };
			if (overlapsDirty(a.px, a.py, a.pw, 256))
				pack(texels);
			if (a.cw > 0 && overlapsDirty(a.cx, a.cy, a.cw, 1))
				pack(clut);
			ready.push_back(index);
			if ((int)ready.size() == DecodeBands)
			{
				decodeSlots(ready);
				ready.clear();
			}
		}
	if (!ready.empty())
		decodeSlots(ready);
}

void decodePending()
{
	if (g.texPending.empty())
		return;
	if (anyCpuNewer())
		flushUploads();
	int x0 = AtlasW, y0 = AtlasH, x1 = 0, y1 = 0;
	std::vector<int> ready, later;
	for (int index : g.texPending)
	{
		auto& t = g.texSlots[index];
		if ((int)ready.size() >= DecodesPerFrame)
		{
			later.push_back(index);	// still pending: the next frame's
			continue;
		}
		t.pending = false;
		if (t.valid)
			continue;	// (decoded at once meanwhile)
		// only a page unchanged for eight frames (one drawn into or animated often stays drawn from VRAM), or changed in a small way
		if (t.key == ~0u || (t.changedFrame + 8 > g.texFrame && !t.smallChange))
			continue;
		// its texels and CLUT current in the raw mirror (the drawn tiles there packed)
		const PageArea a = pageArea(t.depth, t.page, t.clutX, t.clutY);
		const int texels[4] = { a.px, a.py, a.pw, 256 }, clut[4] = { a.cx, a.cy, a.cw, 1 };
		if (overlapsDirty(a.px, a.py, a.pw, 256))
			pack(texels);
		if (a.cw > 0 && overlapsDirty(a.cx, a.cy, a.cw, 1))
			pack(clut);
		const int ox = (index % (AtlasW / 256)) * 256, oy = (index / (AtlasW / 256)) * 256;
		x0 = std::min(x0, ox), y0 = std::min(y0, oy), x1 = std::max(x1, ox + 256), y1 = std::max(y1, oy + 256);
		ready.push_back(index);
	}
	g.texPending.swap(later);
	if (ready.empty())
		return;
	(void)x0, (void)y0, (void)x1, (void)y1;
	decodeSlots(ready);
}

void decodeSlots(const std::vector<int>& ready)
{
	endPass();
	Slot& s = slot();
	const uint32_t stamp = stampOpen(s, StampDecode);
	// one pass over the decode image, a band each, then each band copied into its slot (transfers: only those texels)
	const uint32_t bands = (uint32_t)ready.size(), h = 256 * DecodeBands;
	s.cmd->beginRenderPass(vk::RenderPassBeginInfo(*g.atlasDecodePass, *g.decodeFramebuffer, vk::Rect2D({ 0, 0 }, { 256, 256 * bands })),
			vk::SubpassContents::eInline);
	s.cmd->setViewport(0, vk::Viewport(0, 0, 256.f, (float)h, 0, 1));
	s.cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline(Decode));
	const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	std::vector<vk::ImageCopy> copies;
	s.cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
	for (uint32_t b = 0; b < bands; b++)
	{
		auto& t = g.texSlots[ready[b]];
		const int by = 256 * (int)b;
		const bool part = t.changed[2] >= 0 && t.changed[3] >= 0;
		const int cu0 = part ? std::clamp<int>(t.changed[0], 0, 255) : 0, cv0 = part ? std::clamp<int>(t.changed[1], 0, 255) : 0;
		const int cu1 = part ? std::clamp<int>(t.changed[2], cu0, 255) : 255, cv1 = part ? std::clamp<int>(t.changed[3], cv0, 255) : 255;
		copies.emplace_back(layers, vk::Offset3D(cu0, by + cv0, 0), layers,
				vk::Offset3D((ready[b] % (AtlasW / 256)) * 256 + cu0, (ready[b] / (AtlasW / 256)) * 256 + cv0, 0), vk::Extent3D((uint32_t)(cu1 - cu0 + 1), (uint32_t)(cv1 - cv0 + 1), 1));
		s.cmd->setScissor(0, vk::Rect2D({ cu0, by + cv0 }, { (uint32_t)(cu1 - cu0 + 1), (uint32_t)(cv1 - cv0 + 1) }));
		Push p {};
		p.window[0] = (int)((t.page & 0x1Fu) | t.depth << 7), p.window[1] = t.clutX, p.window[2] = t.clutY;
		p.flags[2] = 0, p.flags[3] = by;
		p.rect[0] = -1.f, p.rect[1] = by * 2.f / h - 1.f, p.rect[2] = 1.f, p.rect[3] = (by + 256) * 2.f / h - 1.f;
		s.cmd->pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
		s.cmd->draw(4, 1, 0, 0);
		t.valid = true;
		// (a world's mesh was sorted by what it had when it was loaded: it stays as sorted; anyone else's, it's looked at again)
		if (!t.kept)
			t.texels = texelClasses(t.depth, t.page, t.clutX, t.clutY, t.clearCells);
		t.changed[2] = t.changed[3] = -1;
		g.decodes++;
	}
	s.cmd->endRenderPass();
	barrier(*s.cmd);
	s.cmd->copyImage(*g.decodeImage.image, vk::ImageLayout::eGeneral, *g.atlas.image, vk::ImageLayout::eGeneral, copies);
	barrier(*s.cmd);
	stampClose(s, stamp);
}

// (a frame announced: its first primitive starts its record)
void replayStart(int scene)
{
	if (!g.replayPending)
		return;
	g.replayPending = false;
	g.replayRecording = g.replayNewest = scene;
	State::Replay& fresh = g.replays[scene];
	fresh.frame = g.replayLatest;
	fresh.vertices.clear(), fresh.z.clear(), fresh.object.clear(), fresh.draws.clear();
}

// a primitive's triangles (3 corners each); textured ones sample the page and CLUT given
void primitive(bool textured, const Vertex *v, uint32_t n, uint16_t page, uint16_t clutX, uint16_t clutY)
{
	g.drawnSinceShown++;
	// the GTE depth and precise corners set for the next triangle: this one's (dropped either way). Precise corners more
	// than a pixel and a half from the drawn ones were set for a triangle never drawn: both dropped.
	bool is3D = g.triDepthValid && n == 3;
	bool precise = g.preciseValid && n == 3;
	g.triDepthValid = g.preciseValid = false;
	if (precise)
		for (uint32_t i = 0; i < 3; i++)
			if (std::fabs(g.precise[2 * i] - v[i].x) > 1.5f || std::fabs(g.precise[2 * i + 1] - v[i].y) > 1.5f)
				precise = is3D = false;
	if (!g.ready || n == 0)
		return;
	if (anyCpuNewer())
		flushUploads();
	const vk::Rect2D sc = scissor();
	if (sc.extent.width == 0 || sc.extent.height == 0)
		return;
	// the VRAM it can read: its UVs' bounds in its page (the whole page under a texture window), and its CLUT
	const uint32_t depth = (page >> 7) & 3;
	int texelArea[4] = {}, clutArea[4] = {};
	if (textured)
	{
		const int px = (page & 15) * 64, py = ((page >> 4) & 1) * 256;
		int u0 = 0, v0 = 0, u1 = 255, v1 = 255;
		if (g.window == 0)
		{
			float fu0 = v[0].u, fv0 = v[0].v, fu1 = fu0, fv1 = fv0;
			for (uint32_t i = 1; i < n; i++)
				fu0 = std::min(fu0, v[i].u), fv0 = std::min(fv0, v[i].v), fu1 = std::max(fu1, v[i].u), fv1 = std::max(fv1, v[i].v);
			u0 = std::max(0, (int)fu0), v0 = std::max(0, (int)fv0), u1 = std::min(255, (int)fu1), v1 = std::min(255, (int)fv1);
		}
		const int shift = depth == 0 ? 2 : depth == 1 ? 1 : 0;
		texelArea[0] = px + (u0 >> shift), texelArea[1] = py + v0;
		texelArea[2] = ((u1 >> shift) - (u0 >> shift)) + 1, texelArea[3] = v1 - v0 + 1;
		clutArea[0] = clutX & (VramW - 1), clutArea[1] = clutY & (VramH - 1);
		clutArea[2] = std::min(depth == 0 ? 16 : 256, VramW - (clutX & (VramW - 1))), clutArea[3] = 1;
		// texels or a CLUT in a scene's area (its frame read as a texture): VRAM given what the scene drew first
		for (State::Scene& scn : g.scenes)
			if (scn.used && scn.newer)
			{
				sceneResolve(scn, "a texture", texelArea);
				if (depth < 2)
					sceneResolve(scn, "a CLUT", clutArea);
			}
	}
	// from the texture cache: its slot in each corner (vTex.y) for the cached shader
	Vertex cachedCorners[6];
	bool cachedTexture = false;
	if (textured && n <= 6)
	{
		const int slotIndex = cachedSlot((page >> 7) & 3, page, clutX, clutY);
		if (slotIndex >= 0)
		{
			for (uint32_t i = 0; i < n; i++)
			{
				cachedCorners[i] = v[i];
				cachedCorners[i].clutX = (uint16_t)slotIndex;
			}
			v = cachedCorners;
			cachedTexture = true;
		}
	}
	if (textured && !cachedTexture)
	{
		// texels drawn since the last pack: bring the mirror up to date first
		const bool texels = overlapsDirty(texelArea[0], texelArea[1], texelArea[2], texelArea[3]);
		const bool clut = depth < 2 && overlapsDirty(clutArea[0], clutArea[1], clutArea[2], clutArea[3]);
		if (texels || clut)
		{
			static int logged = 0;
			if (g.perf && logged < 24)
			{
				logged++;
				DS_LOG("recomp gpu: pack for %s: depth %u, VRAM %d,%d %d x %d, CLUT %d,%d, drawing %d,%d..%d,%d", texels ? "texels" : "CLUT",
						depth, texelArea[0], texelArea[1], texelArea[2], texelArea[3], clutX, clutY, g.area[0], g.area[1], g.area[2], g.area[3]);
			}
			if (texels)
				pack(texelArea);
			if (clut)
				pack(clutArea);
		}
	}
	enableStencil();
	// what it may draw: its bounds, inside the draw area (the scissor)
	float bx0 = v[0].x, by0 = v[0].y, bx1 = bx0, by1 = by0;
	for (uint32_t i = 1; i < n; i++)
		bx0 = std::min(bx0, v[i].x), by0 = std::min(by0, v[i].y), bx1 = std::max(bx1, v[i].x), by1 = std::max(by1, v[i].y);
	const int ax0 = sc.offset.x / g.scale, ay0 = sc.offset.y / g.scale;
	const int ax1 = ax0 + (int)sc.extent.width / g.scale, ay1 = ay0 + (int)sc.extent.height / g.scale;
	const int dx0 = std::max((int)bx0, ax0), dy0 = std::max((int)by0, ay0);
	const int dx1 = std::min((int)bx1 + 1, ax1), dy1 = std::min((int)by1 + 1, ay1);
	const int si = sceneForArea(dx0, dy0, dx1, dy1);
	if (si >= 0)
		sceneSync(g.scenes[si]);
	Slot *s = &slot();
	if (s->vertexUsed + n * sizeof(Vertex) > VertexBytes)
	{
		submit();
		s = &slot();
	}
	// a cached page's texels of one kind only: a semi-transparent primitive in one pass (none blended when none is STP),
	// and no transparent texel: no discard (hidden-surface removal kept)
	uint32_t blend = g.semi ? 1 + g.semiMode : 0;
	uint32_t kind = cachedTexture ? Cached : textured ? Tex : Flat;
	bool twoPasses = textured && g.semi;
	if (cachedTexture)
	{
		const uint8_t cls = g.texSlots[v[0].clutX].texels;
		if (cls != 7)
		{
			if (g.semi && !(cls & 4))
				blend = 0, twoPasses = false;
			else if (g.semi && !(cls & 2))
				twoPasses = false;
			if (!(cls & 1) && !twoPasses)
				kind = CachedOpaque;
			// a page with transparent texels, none of them under this polygon: no discard for it either (a discard turns off
			// this GPU's hidden surface removal, and a page's ground and its trees' cut-outs would all pay for it)
			else if ((cls & 1) && !twoPasses && g.window == 0)
			{
				const auto& slotInfo = g.texSlots[v[0].clutX];
				float u0 = v[0].u, u1 = u0, v0 = v[0].v, v1 = v0;
				for (uint32_t i = 1; i < n; i++)
					u0 = std::min(u0, v[i].u), u1 = std::max(u1, v[i].u), v0 = std::min(v0, v[i].v), v1 = std::max(v1, v[i].v);
				const int cx0 = std::clamp((int)u0 >> 4, 0, 15), cx1 = std::clamp((int)u1 >> 4, 0, 15), cy0 = std::clamp((int)v0 >> 4, 0, 15), cy1 = std::clamp((int)v1 >> 4, 0, 15);
				const uint16_t columns = (uint16_t)(((2u << cx1) - 1u) & ~((1u << cx0) - 1u));
				bool clear = false;
				for (int r = cy0; r <= cy1 && !clear; r++)
					clear = (slotInfo.clearCells[r] & columns) != 0;
				if (!clear)
					kind = CachedOpaque;
			}
		}
	}
	if (kind == Cached || kind == CachedOpaque)
	{
		bool within = g.window == 0;
		for (uint32_t i = 0; i < n && within; i++)
			within = v[i].u >= 0.f && v[i].u <= 256.f && v[i].v >= 0.f && v[i].v <= 256.f;
		if (within)
			kind = kind == Cached ? Lean : LeanOpaque;
	}
	// the pixels it covers (its triangles' areas at the drawing's scale, clipped to the scissor's), twice in two passes
	if (g.perf)
	{
		double area = 0;
		for (uint32_t t = 0; t + 2 < n; t += 3)
			area += std::fabs((v[t + 1].x - v[t].x) * (v[t + 2].y - v[t].y) - (v[t + 2].x - v[t].x) * (v[t + 1].y - v[t].y)) * 0.5;
		const double k = si >= 0 ? (double)g.sceneW * g.sceneH / ((double)g.scenes[si].aw * g.scenes[si].ah) : (double)g.scale * g.scale;
		area = std::min(area * k, (double)sc.extent.width * sc.extent.height * k / (g.scale * g.scale)) * (twoPasses ? 2 : 1);
		g.pixels += area;
		if (blend != 0)
			g.pixelsBlended += area;
		if (kind == Tex)
			g.pixelsVram += area;
		if (!textured)
			g.pixelsFlat += area;
	}
	Push p {};
	p.window[0] = g.window & 31, p.window[1] = (g.window >> 5) & 31, p.window[2] = (g.window >> 10) & 31, p.window[3] = (g.window >> 15) & 31;
	p.flags[0] = g.maskSet, p.flags[2] = g.scale;
	if (si >= 0)
	{
		// into the scene: its corners there (the shaders map 1024 x 512 over the target), at their precise positions; a 3D
		// one at its corners' depths (tested; written when opaque; semi-transparent ones a little nearer: decals and
		// shadows over what they lie on); anything without depths (sprites, the HUD's polygons, a polygon whose depth was
		// not traced) untested in the game's order, as the PlayStation draws it
		State::Scene& S = g.scenes[si];
		sceneMarkDrawn(S, dx0, dy0, dx1, dy1);
		S.lastUsed = g.texFrame;
		uint32_t key = kind | (blend << 4) | SceneBit | stencilBits();
		float d[3] = { 0.f, 0.f, 0.f }, w[3] = { 1.f, 1.f, 1.f };
		if (is3D)
		{
			const bool opaque = blend == 0 && !twoPasses;
			// a true perspective depth (1 - near / z: linear on the screen, so triangles meet where they do in space), the
			// view depth as clip w. DEADSTATION_SCENE_W=0: the depth linear in z, no perspective (affine, as the PlayStation)
			for (int i = 0; i < 3; i++)
			{
				const float z = g.triDepth[i] - (opaque ? 0.f : 32.f);
				if (g.scenePerspective && g.frameIs3D)
					d[i] = scenePerspectiveDepth(z), w[i] = std::max(g.triDepth[i], 1.f);
				else
					d[i] = sceneFlatDepth(z);
			}
			key |= (opaque ? 2u : 1u) << 20;
			S.any3D = true;
		}
		const float kx = 1024.f / (float)S.aw, ky = 512.f / (float)S.ah;
		// (built here and written once: the vertex buffer is uncached GPU memory, slow to read back)
		Vertex built[6];
		for (uint32_t i = 0; i < n && i < 6; i++)
		{
			built[i] = v[i];
			const float x = precise ? g.precise[2 * i] : v[i].x, y = precise ? g.precise[2 * i + 1] : v[i].y;
			built[i].x = (x - (float)S.ax) * kx, built[i].y = (y - (float)S.ay) * ky;
			built[i].depth = d[i < 3 ? i : 0];
			built[i].w = w[i < 3 ? i : 0];
		}
		Vertex *at = (Vertex *)(s->vertexMap + s->vertexUsed);
		memcpy(at, built, n * sizeof(Vertex));
		const uint32_t first = (uint32_t)(s->vertexUsed / sizeof(Vertex));
		s->vertexUsed += n * sizeof(Vertex);
		g.vertices += n;
		g.sceneDrawCount++;
		// (the frame kept for its in-between frame)
		replayStart(si);
		if (g.replayRecording == si)
		{
			State::Replay& record = g.replays[si];
			const uint32_t base = (uint32_t)record.vertices.size();
			const uint8_t tag = (uint8_t)(g.replayObject & 0x7F) | (is3D && !(blend == 0 && !twoPasses) ? 0x80 : 0);
			record.vertices.insert(record.vertices.end(), built, built + n);
			for (uint32_t i = 0; i < n; i++)
			{
				record.z.push_back(is3D ? g.triDepth[i < 3 ? i : 0] : 0.f);
				record.object.push_back(tag);
			}
			// (its draw: the same joining as below, in the record's numbering)
			const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
			const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
			const vk::Rect2D area({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
			State::SceneDraw *last = record.draws.empty() ? nullptr : &record.draws.back();
			if (last != nullptr && last->count != 0 && last->key == key && last->twoPasses == twoPasses && last->scissor == area &&
					last->first + last->count == base && memcmp(&last->push, &p, sizeof(p)) == 0)
				last->count += n;
			else
				record.draws.push_back({ si, base, n, key, area, p, twoPasses, 0 });
		}
		if (kind == Tex)
		{
			sceneMarkReads(texelArea[0], texelArea[1], texelArea[2], texelArea[3]);
			if (depth < 2)
				sceneMarkReads(clutArea[0], clutArea[1], clutArea[2], clutArea[3]);
		}
		// the draw area in the scene's pixels
		const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
		const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
		const vk::Rect2D box({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
		// one draw with the last when it continues it
		if (!g.sceneDraws.empty())
		{
			State::SceneDraw& last = g.sceneDraws.back();
			if (last.scene == si && last.count != 0 && last.key == key && last.twoPasses == twoPasses && last.scissor == box &&
					last.first + last.count == first && memcmp(&last.push, &p, sizeof(p)) == 0)
			{
				last.count += n;
				return;
			}
		}
		g.sceneDraws.push_back({ si, first, n, key, box, p, twoPasses, 0 });
		return;
	}
	// into VRAM. VRAM recorded scene draws read: those drawn first; a scene's area: VRAM given what the scene drew first,
	// the scene given this after.
	if (g.perf)
	{
		g.vramPrims++;
		std::copy(g.area, g.area + 4, g.lastVramArea);
		g.lastVramBounds[0] = dx0, g.lastVramBounds[1] = dy0, g.lastVramBounds[2] = dx1, g.lastVramBounds[3] = dy1;
	}
	if (g.perf && dx0 < dx1 && dy0 < dy1)
	{
		static uint32_t seen[24];
		static int count = 0;
		const uint32_t key = (uint32_t)(dx0 >> 5) | (uint32_t)(dy0 >> 4) << 5 | (uint32_t)kind << 10 | (uint32_t)n << 14;
		if (count < 24 && std::find(seen, seen + count, key) == seen + count)
		{
			seen[count++] = key;
			DS_LOG("recomp gpu: drawn in VRAM: kind %u, %u corners, %d,%d..%d,%d (draw area %d,%d..%d,%d)%s", kind, n, dx0, dy0, dx1, dy1,
					g.area[0], g.area[1], g.area[2], g.area[3], is3D ? ", 3D" : "");
		}
	}
	bool overScene = false;
	if (dx0 < dx1 && dy0 < dy1)
	{
		if (sceneReadsOverlap(dx0, dy0, dx1 - dx0, dy1 - dy0))
			flushScene(-2);
		for (State::Scene& scn : g.scenes)
			if (scn.used && overlaps(dx0, dy0, dx1 - dx0, dy1 - dy0, scn.ax, scn.ay, scn.aw, scn.ah))
				overScene = true;
		if (overScene)
			sceneBeforeWrite(dx0, dy0, dx1 - dx0, dy1 - dy0, "drawing in VRAM over it");
		refreshScaled(dx0, dy0, dx1 - dx0, dy1 - dy0);
	}
	beginPass(sc.offset.x / g.scale, sc.offset.y / g.scale, (sc.offset.x + (int)sc.extent.width) / g.scale,
			(sc.offset.y + (int)sc.extent.height) / g.scale);
	s = &g.slots[g.current];
	if (s->vertexUsed + n * sizeof(Vertex) > VertexBytes)
	{
		submit();
		beginPass(sc.offset.x / g.scale, sc.offset.y / g.scale, (sc.offset.x + (int)sc.extent.width) / g.scale,
				(sc.offset.y + (int)sc.extent.height) / g.scale);
		s = &g.slots[g.current];
	}
	uint32_t key = kind | (blend << 4) | stencilBits();
	// the scene depth: 3D (a triangle with the depths of its corners), opaque or not; else 2D, a background before any 3D
	const bool depthOn = g.sceneDepth && !g.stencilOn;
	uint8_t step = 2;
	float depths[3] = { 1.f, 1.f, 1.f };
	if (depthOn)
	{
		if (is3D)
		{
			for (int i = 0; i < 3; i++)
				depths[i] = g.triDepth[i] * (1.f / 65536.f);
			step = blend == 0 && !twoPasses ? 0 : 2;
			key |= (step == 0 ? 2u : 1u) << 20;
			g.lastDepth = (depths[0] + depths[1] + depths[2]) / 3.f;
		}
		else if (!g.deferredAny3D)
			step = 1, key |= 1u << 20;	// behind every 3D primitive (depth 1, the cleared far end)
		else if (n == 3)
		{
			// a polygon without a depth among 3D ones: at the depth of the 3D primitive before it (the game's order)
			for (float& d : depths)
				d = g.lastDepth;
			key |= 1u << 20;
		}
	}
	if (depthOn)
	{
		const uint32_t first = (uint32_t)(s->vertexUsed / sizeof(Vertex));
		memcpy(s->vertexMap + s->vertexUsed, v, n * sizeof(Vertex));
		Vertex *at = (Vertex *)(s->vertexMap + s->vertexUsed);
		for (uint32_t i = 0; i < n; i++)
			at[i].depth = depths[i < 3 ? i : 0];
		s->vertexUsed += n * sizeof(Vertex);
		g.passPrims.push_back({ first, n, key, sc, p, twoPasses, step, (depths[0] + depths[1] + depths[2]) / 3.f });
		g.deferredAny3D = g.deferredAny3D || is3D;
		g.vertices += n;
	}
	else if (!g.runOpen || g.runKey != key || g.runScissor != sc || memcmp(&g.runPush, &p, sizeof(p)) != 0)
	{
		flushRun();
		g.runOpen = true;
		g.runKey = key, g.runScissor = sc, g.runPush = p;
		g.runTextured = textured, g.runSemi = twoPasses;
		g.runFirst = (uint32_t)(s->vertexUsed / sizeof(Vertex));
		g.runCount = 0;
	}
	if (!depthOn)
	{
		memcpy(s->vertexMap + s->vertexUsed, v, n * sizeof(Vertex));
		s->vertexUsed += n * sizeof(Vertex);
		g.runCount += n;
		g.vertices += n;
	}
	if (dx0 < dx1 && dy0 < dy1)
		markDirty(dx0, dy0, dx1 - dx0, dy1 - dy0);
	if (overScene)
		sceneAfterWrite(dx0, dy0, dx1 - dx0, dy1 - dy0, "drawing in VRAM");
}

// a plugin's colour over a corner (deadstation_tint)
inline void tintCorner(Vertex& v)
{
	if (!g.tinted)
		return;
	if ((v.flags & 1) != 0)
	{
		v.r = v.g = v.b = 128;
		v.flags &= (uint16_t)~1u;
	}
	v.r = (uint8_t)std::min<int>((v.r * g.tint[0]) >> 8, 255);
	v.g = (uint8_t)std::min<int>((v.g * g.tint[1]) >> 8, 255);
	v.b = (uint8_t)std::min<int>((v.b * g.tint[2]) >> 8, 255);
}

Vertex corner(int x, int y, uint16_t color15)
{
	Vertex v {};
	v.x = (float)x, v.y = (float)y;
	v.r = (uint8_t)((color15 & 31) * 255 / 31), v.g = (uint8_t)(((color15 >> 5) & 31) * 255 / 31), v.b = (uint8_t)(((color15 >> 10) & 31) * 255 / 31);
	tintCorner(v);
	return v;
}

Vertex corner(int x, int y, int u, int v, uint32_t color24, uint16_t page, uint16_t clutX, uint16_t clutY, bool raw)
{
	Vertex c {};
	c.x = (float)x, c.y = (float)y, c.u = (float)u, c.v = (float)v;
	c.r = color24 & 255, c.g = (color24 >> 8) & 255, c.b = (color24 >> 16) & 255;
	c.page = page, c.clutX = clutX, c.clutY = clutY, c.flags = raw ? 1 : 0;
	tintCorner(c);
	return c;
}

uint32_t modulation()
{
	return g.mod[0] | (g.mod[1] << 8) | (g.mod[2] << 16);
}

// a CPU upload (or one pixel): PSXRecomp's VRAM, the raw mirror, then the scaled image from it
void uploadVram(int x, int y, int w, int h, const uint16_t *data)
{
	if (w <= 0 || h <= 0)
		return;
	for (int py = y; py < y + h;)
	{
		const int ry = py & (VramH - 1), ph = std::min(y + h - py, VramH - ry);
		for (int px = x; px < x + w;)
		{
			const int rx = px & (VramW - 1), pw = std::min(x + w - px, VramW - rx);
			invalidateTextures(rx, ry, pw, ph, w * h <= 64 * 64);
			px += pw;
		}
		py += ph;
	}
	for (int j = 0; j < h; j++)
		for (int i = 0; i < w; i++)
			g.mirror[((y + j) & (VramH - 1)) * VramW + ((x + i) & (VramW - 1))] = data[j * w + i];
	if (!g.ready)
		return;
	// deferred (the common case: textures, CLUTs, movie frames): the tiles are the CPU's until the GPU next needs VRAM.
	// Not when the mask bit is in play, or over tiles the GPU drew since the last readback (the CPU's copy of the rest
	// of those tiles is stale).
	if (!g.maskSet && !g.maskCheck && x >= 0 && y >= 0 && x + w <= VramW && y + h <= VramH)
	{
		bool gpuThere = false;
		tiles(x, y, w, h, [&](int t) { gpuThere = gpuThere || ((g.gpuNewer[t >> 6] >> (t & 63)) & 1); });
		if (!gpuThere)
		{
			tiles(x, y, w, h, [](int t) { g.cpuNewer[t >> 6] |= 1ull << (t & 63); });
			g.deferred++;
			return;
		}
	}
	if (anyCpuNewer())
		flushUploads();
	// in pieces that don't wrap
	for (int py = y; py < y + h;)
	{
		const int ry = py & (VramH - 1), ph = std::min(y + h - py, VramH - ry);
		for (int px = x; px < x + w;)
		{
			const int rx = px & (VramW - 1), pw = std::min(x + w - px, VramW - rx);
			const vk::DeviceSize bytes = (vk::DeviceSize)pw * ph * 2;
			// the tiles drawn there packed into the mirror first (it then holds all of them), the upload into the mirror; the
			// scaled image takes those tiles when next read or drawn over (refreshScaled)
			if (overlapsDirty(rx, ry, pw, ph))
			{
				const int piece[4] = { rx, ry, pw, ph };
				pack(piece);
			}
			endPass();
			Slot *s = &slot();
			if (s->stagingUsed + bytes > StagingBytes)
			{
				submit();
				s = &slot();
			}
			for (int j = 0; j < ph; j++)
				memcpy(s->stagingMap + s->stagingUsed + (size_t)j * pw * 2, &g.mirror[(ry + j) * VramW + rx], (size_t)pw * 2);
			vk::BufferImageCopy copy(s->stagingUsed, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
					{ rx, ry, 0 }, { (uint32_t)pw, (uint32_t)ph, 1 });
			s->cmd->copyBufferToImage(*s->staging, *g.raw.image, vk::ImageLayout::eGeneral, copy);
			s->stagingUsed = (s->stagingUsed + bytes + 3) & ~(vk::DeviceSize)3;
			barrier(*s->cmd);
			enableStencil();
			g.uploads++;
			tiles(rx, ry, pw, ph, [](int t) { g.rawNewer[t >> 6] |= 1ull << (t & 63); });
			px += pw;
		}
		py += ph;
	}
}

// an upload: VRAM recorded scene draws read drawn first, the scenes over it given it after
void upload(int x, int y, int w, int h, const uint16_t *data)
{
	if (w <= 0 || h <= 0)
		return;
	if (g.perf)
		g.uploadPixels += (double)w * h, g.lastUpload[0] = x, g.lastUpload[1] = y, g.lastUpload[2] = w, g.lastUpload[3] = h;
	if (g.ready && sceneReadsOverlap(x, y, w, h))
		flushScene(-2);
	if (g.ready)
		sceneBeforeWrite(x, y, w, h, "an upload over it");
	uploadVram(x, y, w, h, data);
	if (g.ready)
		sceneAfterWrite(x, y, w, h, "an upload");
}

// the deferred uploads to the GPU: the tiles the CPU's copy holds newer, as runs along each tile row, copied into the
// raw mirror in one transfer and blitted into the scaled image in one pass
void flushUploads()
{
	if (!g.ready || !anyCpuNewer())
		return;
	struct Run { int x, y, w, h; };
	std::vector<Run> runs;
	for (int ty = 0; ty < 32; ty++)
		for (int tx = 0; tx < 32;)
		{
			const int t = ty * 32 + tx;
			if (!((g.cpuNewer[t >> 6] >> (t & 63)) & 1))
			{
				tx++;
				continue;
			}
			int e = tx;
			while (e < 32 && ((g.cpuNewer[(ty * 32 + e) >> 6] >> ((ty * 32 + e) & 63)) & 1))
				e++;
			runs.push_back({ tx * 32, ty * 16, (e - tx) * 32, 16 });
			tx = e;
		}
	std::fill(std::begin(g.cpuNewer), std::end(g.cpuNewer), 0);
	endPass();
	Slot *s = &slot();
	vk::DeviceSize bytes = 0;
	for (const Run& r : runs)
		bytes += (vk::DeviceSize)r.w * r.h * 2;
	if (s->stagingUsed + bytes > StagingBytes)
	{
		submit();
		s = &slot();
	}
	std::vector<vk::BufferImageCopy> copies;
	int x0 = VramW, y0 = VramH, x1 = 0, y1 = 0;
	for (const Run& r : runs)
	{
		for (int j = 0; j < r.h; j++)
			memcpy(s->stagingMap + s->stagingUsed + (size_t)j * r.w * 2, &g.mirror[(r.y + j) * VramW + r.x], (size_t)r.w * 2);
		copies.emplace_back(s->stagingUsed, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
				vk::Offset3D(r.x, r.y, 0), vk::Extent3D((uint32_t)r.w, (uint32_t)r.h, 1));
		s->stagingUsed += (vk::DeviceSize)r.w * r.h * 2;
		x0 = std::min(x0, r.x), y0 = std::min(y0, r.y), x1 = std::max(x1, r.x + r.w), y1 = std::max(y1, r.y + r.h);
	}
	const uint32_t stamp = stampOpen(*s, StampUpload);
	s->cmd->copyBufferToImage(*s->staging, *g.raw.image, vk::ImageLayout::eGeneral, copies);
	barrier(*s->cmd);
	stampClose(*s, stamp);
	g.uploads += (unsigned)runs.size();
	// the scaled image takes them when next read or drawn over (refreshScaled)
	(void)x0, (void)y0, (void)x1, (void)y1;
	for (const Run& r : runs)
		tiles(r.x, r.y, r.w, r.h, [](int t) { g.rawNewer[t >> 6] |= 1ull << (t & 63); });
}

// PSXRecomp's VRAM brought up to date from the GPU's (a region)
void readback(int x, int y, int w, int h)
{
	// what scenes drew, into VRAM first
	if (g.ready)
		sceneResolveAll("a read of VRAM");
	if (!g.ready || !g.cpuStale)
		return;
	if (anyCpuNewer())
		flushUploads();
	pack();
	endPass();
	Slot& s = slot();
	vk::BufferImageCopy copy(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { 0, 0, 0 },
			{ (uint32_t)VramW, (uint32_t)VramH, 1 });
	g.readbacks++;
	s.cmd->copyImageToBuffer(*g.raw.image, vk::ImageLayout::eGeneral, *g.readback, copy);
	const uint32_t done = g.current;
	submit();
	(void)g.dev.device.waitForFences(*g.slots[done].fence, true, UINT64_MAX);
	// all of it (the rest is as fresh): later reads need none until the GPU draws again
	(void)x, (void)y, (void)w, (void)h;
	memcpy(g.mirror, g.readbackMap, (size_t)VramW * VramH * 2);
	// PSXRecomp's too (with the render thread: every readback is a call the module's thread waits for)
	if (g.cpu != nullptr && g.cpu != g.mirror)
		memcpy(g.cpu, g.mirror, (size_t)VramW * VramH * 2);
	g.cpuStale = false;
	std::fill(std::begin(g.gpuNewer), std::end(g.gpuNewer), 0);
}

// DEADSTATION_PERF, once at start: the GPU's time for the same fill (20 draws of the upload blit, an integer texel fetch and
// a write, over a 736 x 960 area) three ways: into the VRAM image (GENERAL, 2048 x 1024) with that render area, the
// same with the whole image as render area, and into a 736 x 960 image of its own (colour attachment layout). Logged.
void gpuSelfTest()
{
	if (!g.perf || hostVideo().timestampNs() <= 0)
		return;
	vk::Device d = g.dev.device;
	try
	{
		const uint32_t W = 736, H = 960;
		Image own;
		makeImage(own, vk::Format::eR8G8B8A8Unorm, W, H, vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor);
		vk::AttachmentDescription attachment(vk::AttachmentDescriptionFlags(), vk::Format::eR8G8B8A8Unorm, vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
		vk::AttachmentReference ref(0, vk::ImageLayout::eColorAttachmentOptimal);
		vk::SubpassDescription sub(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, ref);
		vk::UniqueRenderPass ownPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachment, sub));
		vk::UniqueFramebuffer ownFb = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *ownPass,
				*own.view, W, H, 1));
		vk::UniqueQueryPool pool = d.createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 6));
		vk::UniqueCommandBuffer cmd = std::move(d.allocateCommandBuffersUnique(
				vk::CommandBufferAllocateInfo(*g.commandPool, vk::CommandBufferLevel::ePrimary, 1))[0]);
		vk::UniqueFence fence = d.createFenceUnique(vk::FenceCreateInfo());
		cmd->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		cmd->resetQueryPool(*pool, 0, 6);
		vk::ImageMemoryBarrier b(vk::AccessFlags(), vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
				vk::ImageLayout::eColorAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *own.image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
				vk::DependencyFlags(), nullptr, nullptr, b);
		const vk::Pipeline blitPipe = pipeline(Blit);
		auto fills = [&](vk::RenderPass pass, vk::Framebuffer fb, vk::Rect2D area, float vw, float vh, int q)
		{
			const vk::ClearValue clears[2] = { vk::ClearColorValue(), vk::ClearDepthStencilValue(1.f, 0) };
			cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, q);
			cmd->beginRenderPass(vk::RenderPassBeginInfo(pass, fb, area, g.sceneDepth && pass != *ownPass ? 2u : 0u, clears),
					vk::SubpassContents::eInline);
			cmd->setViewport(0, vk::Viewport(0, 0, vw, vh, 0, 1));
			cmd->setScissor(0, vk::Rect2D({ 0, 0 }, { W, H }));
			cmd->bindPipeline(vk::PipelineBindPoint::eGraphics, blitPipe);
			cmd->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
			Push p {};
			p.flags[2] = g.scale;
			p.rect[0] = -1.f, p.rect[1] = -1.f, p.rect[2] = 2.f * W / vw - 1.f, p.rect[3] = 2.f * H / vh - 1.f;
			cmd->pushConstants(*g.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(p), &p);
			for (int i = 0; i < 20; i++)
				cmd->draw(4, 1, 0, 0);
			cmd->endRenderPass();
			cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *pool, q + 1);
		};
		const float vw = (float)(VramW * g.scale), vh = (float)(VramH * g.scale);
		fills(*g.colorPass, *g.colorFramebuffer, vk::Rect2D({ 0, 0 }, { W, H }), vw, vh, 0);
		fills(*g.colorPass, *g.colorFramebuffer, vk::Rect2D({ 0, 0 }, { (uint32_t)vw, (uint32_t)vh }), vw, vh, 2);
		fills(*ownPass, *ownFb, vk::Rect2D({ 0, 0 }, { W, H }), (float)W, (float)H, 4);
		cmd->end();
		const vk::CommandBuffer c = *cmd;
		hostVideo().submit(&c, 1, *fence);
		(void)d.waitForFences(*fence, true, 3000000000ull);
		uint64_t t[6];
		if (d.getQueryPoolResults(*pool, 0, 6, sizeof(t), t, sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
				vk::Result::eSuccess)
		{
			const double ms = hostVideo().timestampNs() * 1e-6;
			DS_LOG("recomp gpu: self-test, 20 fills of 736 x 960 (14.1 M pixels): VRAM image, its area %.2f ms; VRAM image, all of it "
					"as render area %.2f ms; an image of its own %.2f ms", (t[1] - t[0]) * ms, (t[3] - t[2]) * ms, (t[5] - t[4]) * ms);
		}
		d.waitIdle();
	}
	catch (const std::exception& e)
	{
		DS_LOG("recomp gpu: self-test failed: %s", e.what());
	}
}

bool init()
{
	Video& video = hostVideo();
	if (!video.ready())
		return false;
	try
	{
		g.dev = video.gpuDevice();
		vk::Device d = g.dev.device;
		const uint32_t w = VramW * g.scale, h = VramH * g.scale;
		{
			// DEADSTATION_GPU_DEPTH=1 (off by default: it did not lower the drawing's GPU time)
			const char *depth = getenv("DEADSTATION_GPU_DEPTH");
			g.sceneDepth = depth != nullptr && strcmp(depth, "1") == 0;
		}
		for (vk::Format f : g.sceneDepth ? std::initializer_list<vk::Format> { vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint } :
				std::initializer_list<vk::Format> { vk::Format::eS8Uint, vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint })
			if (g.dev.physicalDevice.getFormatProperties(f).optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
			{
				g.stencilFormat = f;
				break;
			}
		if (g.stencilFormat == vk::Format::eUndefined && g.sceneDepth)
		{
			g.sceneDepth = false;
			for (vk::Format f : { vk::Format::eS8Uint })
				if (g.dev.physicalDevice.getFormatProperties(f).optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
					g.stencilFormat = f;
		}
		if (g.stencilFormat == vk::Format::eUndefined)
			throw std::runtime_error("no stencil format");
		using U = vk::ImageUsageFlagBits;
		makeImage(g.scaled, vk::Format::eR8G8B8A8Unorm, w, h, U::eColorAttachment | U::eSampled | U::eTransferSrc | U::eTransferDst,
				vk::ImageAspectFlagBits::eColor);
		makeImage(g.copy, vk::Format::eR8G8B8A8Unorm, w, h, U::eTransferSrc | U::eTransferDst, vk::ImageAspectFlagBits::eColor);
		makeImage(g.stencil, g.stencilFormat, w, h, U::eDepthStencilAttachment, g.stencilFormat == vk::Format::eS8Uint ?
				vk::ImageAspectFlags(vk::ImageAspectFlagBits::eStencil) : vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
		makeImage(g.raw, vk::Format::eR16Uint, VramW, VramH, U::eColorAttachment | U::eSampled | U::eTransferSrc | U::eTransferDst,
				vk::ImageAspectFlagBits::eColor);
		makeImage(g.rawCopy, vk::Format::eR16Uint, VramW, VramH, U::eTransferSrc | U::eTransferDst, vk::ImageAspectFlagBits::eColor);
		{
			const vk::FormatFeatureFlags need = vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
			if ((g.dev.physicalDevice.getFormatProperties(vk::Format::eA1R5G5B5UnormPack16).optimalTilingFeatures & need) == need)
				g.atlasFormat = vk::Format::eA1R5G5B5UnormPack16;
		}
		makeImage(g.atlas, g.atlasFormat, AtlasW, AtlasH, U::eColorAttachment | U::eSampled | U::eTransferDst, vk::ImageAspectFlagBits::eColor);
		makeImage(g.decodeImage, g.atlasFormat, 256, 256 * DecodeBands, U::eColorAttachment | U::eTransferSrc, vk::ImageAspectFlagBits::eColor);

		// drawing: the scaled image and the stencil; packing: the raw mirror. All kept as they are between passes.
		vk::AttachmentDescription drawAttachments[2] = {
			{ vk::AttachmentDescriptionFlags(), vk::Format::eR8G8B8A8Unorm, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad,
					vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
					vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral },
			{ vk::AttachmentDescriptionFlags(), g.stencilFormat, vk::SampleCountFlagBits::e1,
					g.sceneDepth ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
					g.sceneDepth ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eLoad,
					vk::AttachmentStoreOp::eStore, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal },
		};
		// the colour pass (no mask bit in play): with the order depth, the depth attachment too, its stencil not kept
		vk::AttachmentDescription colorAttachments[2] = { drawAttachments[0], drawAttachments[1] };
		colorAttachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		vk::AttachmentReference colorRef(0, vk::ImageLayout::eGeneral);
		vk::AttachmentReference stencilRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
		vk::SubpassDescription drawSubpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, colorRef, nullptr, &stencilRef);
		g.drawPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), drawAttachments, drawSubpass));
		vk::SubpassDescription colorSubpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, colorRef);
		if (g.sceneDepth)
			g.colorPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), colorAttachments, drawSubpass));
		else
			g.colorPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), drawAttachments[0], colorSubpass));
		vk::AttachmentDescription packAttachment(vk::AttachmentDescriptionFlags(), vk::Format::eR16Uint, vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
		vk::SubpassDescription packSubpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, colorRef);
		g.packPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), packAttachment, packSubpass));
		vk::ImageView drawViews[2] = { *g.scaled.view, *g.stencil.view };
		g.drawFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.drawPass, drawViews, w, h, 1));
		if (g.sceneDepth)
			g.colorFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.colorPass, drawViews,
					w, h, 1));
		else
			g.colorFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.colorPass,
					*g.scaled.view, w, h, 1));
		g.packFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.packPass, *g.raw.view,
				VramW, VramH, 1));
		vk::AttachmentDescription atlasAttachment = drawAttachments[0];
		atlasAttachment.format = g.atlasFormat;
		g.atlasPass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), atlasAttachment, colorSubpass));
		atlasAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
		g.atlasDecodePass = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), atlasAttachment, colorSubpass));
		g.decodeFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.atlasDecodePass,
				*g.decodeImage.view, 256, 256 * DecodeBands, 1));
		g.atlasFramebuffer = d.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.atlasPass,
				*g.atlas.view, AtlasW, AtlasH, 1));

		g.sampler = d.createSamplerUnique(vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eNearest, vk::Filter::eNearest,
				vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
				vk::SamplerAddressMode::eClampToEdge));
		g.samplerFiltered = d.createSamplerUnique(vk::SamplerCreateInfo(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear,
				vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
				vk::SamplerAddressMode::eClampToEdge));
		vk::DescriptorSetLayoutBinding bindings[5] = {
			{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			{ 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			{ 4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
		};
		g.setLayout = d.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings));
		vk::DescriptorPoolSize poolSize(vk::DescriptorType::eCombinedImageSampler, 5);
		g.pool = d.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, poolSize));
		g.set = d.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(*g.pool, *g.setLayout))[0];
		vk::DescriptorImageInfo rawInfo(*g.sampler, *g.raw.view, vk::ImageLayout::eGeneral);
		vk::DescriptorImageInfo scaledInfo(*g.sampler, *g.scaled.view, vk::ImageLayout::eGeneral);
		vk::DescriptorImageInfo atlasInfo(*g.sampler, *g.atlas.view, vk::ImageLayout::eGeneral);
		vk::DescriptorImageInfo atlasFilteredInfo(*g.samplerFiltered, *g.atlas.view, vk::ImageLayout::eGeneral);
		// the atlas again for the lean path's normalized fetches (the same nearest, clamped sampler), and filtered for what's far off in a world's mesh
		vk::WriteDescriptorSet writes[5] = {
			{ g.set, 0, 0, vk::DescriptorType::eCombinedImageSampler, rawInfo },
			{ g.set, 1, 0, vk::DescriptorType::eCombinedImageSampler, scaledInfo },
			{ g.set, 2, 0, vk::DescriptorType::eCombinedImageSampler, atlasInfo },
			{ g.set, 3, 0, vk::DescriptorType::eCombinedImageSampler, atlasInfo },
			{ g.set, 4, 0, vk::DescriptorType::eCombinedImageSampler, atlasFilteredInfo },
		};
		d.updateDescriptorSets(writes, nullptr);
		vk::PushConstantRange push(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(Push));
		g.layout = d.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *g.setLayout, push));
		g.vert = module(psx_gpu_vert, sizeof(psx_gpu_vert));
		g.flat = module(psx_gpu_flat_frag, sizeof(psx_gpu_flat_frag));
		g.tex = module(psx_gpu_tex_frag, sizeof(psx_gpu_tex_frag));
		g.cached = module(psx_gpu_cached_frag, sizeof(psx_gpu_cached_frag));
		g.cachedOpaque = module(psx_gpu_cached_opaque_frag, sizeof(psx_gpu_cached_opaque_frag));
		g.worldVert = worldShadersGiven.vert != nullptr ? module(worldShadersGiven.vert, worldShadersGiven.vertBytes) : module(psx_gpu_world_vert, sizeof(psx_gpu_world_vert));
		g.world = module(psx_gpu_world_frag, sizeof(psx_gpu_world_frag));
		g.worldOpaque = module(psx_gpu_world_opaque_frag, sizeof(psx_gpu_world_opaque_frag));
		if (worldShadersGiven.solidVert != nullptr)
		{
			g.worldOwnVert = module(worldShadersGiven.solidVert, worldShadersGiven.solidVertBytes);
			g.worldOwn = module(worldShadersGiven.solidFrag, worldShadersGiven.solidFragBytes);
			g.worldOwnEdge = module(worldShadersGiven.solidEdgeFrag, worldShadersGiven.solidEdgeFragBytes);
		}
		g.worldSkyVert = module(psx_gpu_world_sky_vert, sizeof(psx_gpu_world_sky_vert));
		g.worldSky = module(psx_gpu_world_sky_frag, sizeof(psx_gpu_world_sky_frag));
		g.worldEdge = module(psx_gpu_world_edge_frag, sizeof(psx_gpu_world_edge_frag));
		{
			const uint32_t pushMost = g.dev.physicalDevice.getProperties().limits.maxPushConstantsSize;
			g.worldPushSize = std::min<uint32_t>(sizeof(WorldPush), pushMost & ~15u);
			DS_LOG("recomp gpu: %u bytes of push constants here; a world's draw takes %u of the %zu it could use", pushMost, g.worldPushSize, sizeof(WorldPush));
			vk::PushConstantRange worldPush(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, g.worldPushSize);
			g.worldLayout = d.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *g.setLayout, worldPush));
		}
		g.leanVert = module(psx_gpu_lean_vert, sizeof(psx_gpu_lean_vert));
		g.lean = module(psx_gpu_lean_frag, sizeof(psx_gpu_lean_frag));
		g.leanOpaque = module(psx_gpu_lean_opaque_frag, sizeof(psx_gpu_lean_opaque_frag));
		g.decode = module(psx_gpu_decode_frag, sizeof(psx_gpu_decode_frag));
		g.rectVert = module(psx_gpu_rect_vert, sizeof(psx_gpu_rect_vert));
		g.blit = module(psx_gpu_blit_frag, sizeof(psx_gpu_blit_frag));
		g.pack = module(psx_gpu_pack_frag, sizeof(psx_gpu_pack_frag));

		g.commandPool = d.createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, g.dev.queueFamily));
		auto commands = d.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(*g.commandPool, vk::CommandBufferLevel::ePrimary, Slots));
		for (uint32_t i = 0; i < Slots; i++)
		{
			Slot& s = g.slots[i];
			s.cmd = std::move(commands[i]);
			s.fence = d.createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
			makeBuffer(s.vertices, s.verticesMemory, s.vertexMap, VertexBytes, vk::BufferUsageFlagBits::eVertexBuffer);
			makeBuffer(s.staging, s.stagingMemory, s.stagingMap, StagingBytes, vk::BufferUsageFlagBits::eTransferSrc);
		}
		makeBuffer(g.readback, g.readbackMemory, g.readbackMap, (vk::DeviceSize)VramW * VramH * 2, vk::BufferUsageFlagBits::eTransferDst);

		// the native scene: the panel's size, a depth format; blits between it and VRAM (scaled both ways, filtered)
		{
			const char *perspective = getenv("DEADSTATION_SCENE_W");
			g.scenePerspective = perspective == nullptr || strcmp(perspective, "0") != 0;
			const char *env = getenv("DEADSTATION_SCENE");
			g.sceneOn = env == nullptr || strcmp(env, "0") != 0;
			const vk::Extent2D panel = video.panel().extent();
			g.sceneW = panel.width > 0 ? panel.width : 1024, g.sceneH = panel.height > 0 ? panel.height : 768;
			// with a stencil (the mask bit) first
			for (vk::Format f : { vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint, vk::Format::eD32Sfloat, vk::Format::eD16Unorm, vk::Format::eX8D24UnormPack32 })
				if (g.dev.physicalDevice.getFormatProperties(f).optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
				{
					g.sceneDepthFormat = f;
					g.sceneStencil = f == vk::Format::eD24UnormS8Uint || f == vk::Format::eD32SfloatS8Uint;
					break;
				}
			const vk::FormatFeatureFlags blits = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
					vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
			if (g.sceneDepthFormat == vk::Format::eUndefined ||
					(g.dev.physicalDevice.getFormatProperties(vk::Format::eR8G8B8A8Unorm).optimalTilingFeatures & blits) != blits)
				g.sceneOn = false;
			for (int variant = 0; variant < 8 && g.sceneOn; variant++)
			{
				const bool loaded = variant & 2, stored = variant & 1;
				vk::AttachmentDescription attachments[2] = {
					{ vk::AttachmentDescriptionFlags(), SceneFormat, vk::SampleCountFlagBits::e1, (variant & 4) ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
							vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
							vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral },
					{ vk::AttachmentDescriptionFlags(), g.sceneDepthFormat, vk::SampleCountFlagBits::e1,
							loaded ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear,
							stored ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare,
							!g.sceneStencil ? vk::AttachmentLoadOp::eDontCare : loaded ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear,
							g.sceneStencil && stored ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare,
							loaded ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eUndefined,
							vk::ImageLayout::eDepthStencilAttachmentOptimal },
				};
				vk::AttachmentReference sceneColor(0, vk::ImageLayout::eGeneral), sceneDepth(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
				vk::SubpassDescription sub(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, sceneColor, nullptr, &sceneDepth);
				g.scenePasses[variant] = d.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachments, sub));
			}
		}

		// the images' layouts, VRAM black and clear, then PSXRecomp's VRAM as it is
		Slot& s = slot();
		const auto to = [&](vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout layout) {
			vk::ImageMemoryBarrier b(vk::AccessFlags(), vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
					vk::ImageLayout::eUndefined, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
					vk::ImageSubresourceRange(aspect, 0, 1, 0, 1));
			s.cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags(),
					nullptr, nullptr, b);
		};
		const vk::ImageAspectFlags stencilAspect = g.stencilFormat == vk::Format::eS8Uint ?
				vk::ImageAspectFlags(vk::ImageAspectFlagBits::eStencil) : vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		to(*g.scaled.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.copy.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.raw.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.rawCopy.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.atlas.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.decodeImage.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.stencil.image, stencilAspect, vk::ImageLayout::eDepthStencilAttachmentOptimal);
		const vk::ImageSubresourceRange color(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
		s.cmd->clearColorImage(*g.scaled.image, vk::ImageLayout::eGeneral, vk::ClearColorValue(std::array<float, 4> { 0, 0, 0, 0 }), color);
		s.cmd->clearColorImage(*g.raw.image, vk::ImageLayout::eGeneral, vk::ClearColorValue(std::array<uint32_t, 4> { 0, 0, 0, 0 }), color);
		barrier(*s.cmd);
		g.ready = true;
		g.texCache = getenv("DEADSTATION_TEXCACHE") == nullptr || strcmp(getenv("DEADSTATION_TEXCACHE"), "0") != 0;
		g.perf = getenv("DEADSTATION_PERF") != nullptr;
		if (g.perf && hostVideo().timestampNs() > 0)
			g.timestamps = g.dev.device.createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(),
					vk::QueryType::eTimestamp, Slots * StampsPerSlot));
		if (g.mirror != nullptr)
		{
			std::vector<uint16_t> all(g.mirror, g.mirror + VramW * VramH);
			upload(0, 0, VramW, VramH, all.data());
		}
		submit();
		gpuSelfTest();
		if (g.sceneOn)
			DS_LOG("recomp gpu: frame buffers drawn natively at %u x %u with a depth buffer (format %d); the texture cache's format %d",
					g.sceneW, g.sceneH, (int)g.sceneDepthFormat, (int)g.atlasFormat);
		startPipelines();
		DS_LOG("recomp module: the PlayStation's GPU drawn by Vulkan at %dx (%d x %d), stencil format %d", g.scale, w, h,
				(int)g.stencilFormat);
		return true;
	}
	catch (const std::exception& e)
	{
		DS_LOG("recomp module: no Vulkan GPU: %s", e.what());
		g.ready = false;
		return false;
	}
}

// ---- the backend's table (gpu_render.h) ----

void bInit(uint16_t *vram)
{
	g.cpu = vram;
	if (g.mirror == nullptr || g.mirror == vram || vram == nullptr)
		g.mirror = vram;
	else
		memcpy(g.mirror, vram, (size_t)VramW * VramH * 2);
	sw_renderer_init(vram);
}
void bSetScale(int scale)
{
	if (!g.ready && scale >= 1 && scale <= 4)
		g.scale = scale;
}
int bScale() { return g.scale; }
void bSetFilter(int bilinear) { g.filter = bilinear ? 1 : 0; }
int bFilter() { return g.filter; }
void bSetSemi(int enabled, int mode) { g.semi = enabled != 0, g.semiMode = mode & 3; }
void bSetMask(int set, int check) { g.maskSet = set != 0, g.maskCheck = check != 0; }
void bSetWindow(uint32_t raw) { g.window = raw; }
void bSetTriangleDepth(int enabled, float z0, float z1, float z2)
{
	g.triDepthValid = enabled != 0;
	g.triDepth[0] = z0, g.triDepth[1] = z1, g.triDepth[2] = z2;
}
void bSetPrecise(int enabled, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t x2, int32_t y2)
{
	g.preciseValid = enabled != 0;
	const int32_t c[6] = { x0, y0, x1, y1, x2, y2 };
	for (int i = 0; i < 6; i++)
		g.precise[i] = (float)c[i] * (1.f / 65536.f);
}
void bSetTint(int r, int gr, int b)
{
	g.tint[0] = r, g.tint[1] = gr, g.tint[2] = b;
	g.tinted = r != 256 || gr != 256 || b != 256;
}

void bSetModulation(int r, int gr, int b, int raw)
{
	g.mod[0] = (uint8_t)r, g.mod[1] = (uint8_t)gr, g.mod[2] = (uint8_t)b, g.modRaw = raw != 0;
}

void bFill(int x, int y, int w, int h, uint16_t color)
{
	if (!g.ready || w <= 0 || h <= 0)
		return;
	if (sceneReadsOverlap(x, y, w, h))
		flushScene(-2);
	// inside a scene's area: cleared there (colour and depth), VRAM left as it is
	for (int i = 0; i < State::SceneCount; i++)
	{
		State::Scene& scn = g.scenes[i];
		if (scn.used && x >= scn.ax && y >= scn.ay && x + w <= scn.ax + scn.aw && y + h <= scn.ay + scn.ah)
		{
			sceneSync(scn);
			const int sx0 = sceneX(scn, x), sy0 = sceneY(scn, y), sx1 = sceneX(scn, x + w), sy1 = sceneY(scn, y + h);
			g.sceneDraws.push_back({ i, 0, 0, 0, vk::Rect2D({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) }),
					Push {}, false, color });
			if (g.replayRecording == i)
				g.replays[i].draws.push_back(g.sceneDraws.back());
			sceneMarkDrawn(scn, x, y, x + w, y + h);
			scn.lastUsed = g.texFrame;
			g.fills++;
			return;
		}
	}
	sceneBeforeWrite(x, y, w, h, "a fill over it");
	if (anyCpuNewer())
		flushUploads();
	refreshScaled(x, y, std::min(w, VramW - x), std::min(h, VramH - y));
	g.fills++;
	// in pieces that don't wrap; the mask bit cleared
	for (int py = y; py < y + h;)
	{
		const int ry = py & (VramH - 1), ph = std::min(y + h - py, VramH - ry);
		for (int px = x; px < x + w;)
		{
			const int rx = px & (VramW - 1), pw = std::min(x + w - px, VramW - rx);
			beginPass(rx, ry, rx + pw, ry + ph, false);
			flushRun();
			flushDeferred();
			vk::CommandBuffer cmd = *g.slots[g.current].cmd;
			const vk::Rect2D r({ rx * g.scale, ry * g.scale }, vk::Extent2D((uint32_t)pw * g.scale, (uint32_t)ph * g.scale));
			vk::ClearAttachment clears[2] = {
				{ vk::ImageAspectFlagBits::eColor, 0, vk::ClearColorValue(std::array<float, 4> { (color & 31) / 31.f,
						((color >> 5) & 31) / 31.f, ((color >> 10) & 31) / 31.f, 0.f }) },
				{ vk::ImageAspectFlagBits::eStencil, 0, vk::ClearDepthStencilValue(1.f, 0) },
			};
			cmd.clearAttachments(vk::ArrayProxy<const vk::ClearAttachment>(g.stencilOn ? 2u : 1u, clears), vk::ClearRect(r, 0, 1));
			markDirty(rx, ry, pw, ph);
			px += pw;
		}
		py += ph;
	}
	sceneAfterWrite(x, y, w, h, "a fill");
}

void bCopy(int sx, int sy, int dx, int dy, int w, int h)
{
	if (!g.ready || w <= 0 || h <= 0)
		return;
	w = std::min({ w, VramW - sx, VramW - dx }), h = std::min({ h, VramH - sy, VramH - dy });
	if (w <= 0 || h <= 0)
		return;
	// the source in a scene's area: VRAM given what the scene drew first; the destination read by recorded scene draws:
	// those drawn first
	// onto itself (a game may copy a 2 x 512 strip onto itself every frame): nothing changes, unless the mask bit is set
	if (sx == dx && sy == dy && !g.maskSet)
		return;
	if (g.perf)
	{
		static uint64_t seen[16];
		static int count = 0;
		const uint64_t key = (uint64_t)sx | (uint64_t)sy << 10 | (uint64_t)dx << 19 | (uint64_t)dy << 29 | (uint64_t)w << 38 | (uint64_t)h << 49;
		if (count < 16 && std::find(seen, seen + count, key) == seen + count)
		{
			seen[count++] = key;
			DS_LOG("recomp gpu: a copy %d,%d -> %d,%d, %d x %d", sx, sy, dx, dy, w, h);
		}
	}
	// inside one scene's area, the scene drawn since VRAM took it (not overlapping itself): copied in the scene, VRAM left
	// as it is there (it lacks the scene's drawing anyway)
	for (State::Scene& scn : g.scenes)
		if (scn.used && scn.newer && sx >= scn.ax && sy >= scn.ay && sx + w <= scn.ax + scn.aw && sy + h <= scn.ay + scn.ah &&
				dx >= scn.ax && dy >= scn.ay && dx + w <= scn.ax + scn.aw && dy + h <= scn.ay + scn.ah && !overlaps(sx, sy, w, h, dx, dy, w, h))
		{
			sceneSync(scn);
			flushScene(-2);
			endPass();
			Slot& s = slot();
			const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
			const std::array<vk::Offset3D, 2> from { vk::Offset3D(sceneX(scn, sx), sceneY(scn, sy), 0), vk::Offset3D(sceneX(scn, sx + w), sceneY(scn, sy + h), 1) };
			const std::array<vk::Offset3D, 2> to { vk::Offset3D(sceneX(scn, dx), sceneY(scn, dy), 0), vk::Offset3D(sceneX(scn, dx + w), sceneY(scn, dy + h), 1) };
			restoreScenes();
			s.cmd->blitImage(*scn.color.image, vk::ImageLayout::eGeneral, *scn.color.image, vk::ImageLayout::eGeneral,
					vk::ImageBlit(layers, from, layers, to), vk::Filter::eNearest);
			barrier(*s.cmd);
			sceneMarkDrawn(scn, dx, dy, dx + w, dy + h);
			g.copies++;
			return;
		}
	// from a scene's area: VRAM given what the scene drew there first
	const int copySource[4] = { sx, sy, w, h };
	for (State::Scene& scn : g.scenes)
		if (scn.used && scn.newer)
			sceneResolve(scn, "a copy from it", copySource);
	if (sceneReadsOverlap(dx, dy, w, h))
		flushScene(-2);
	sceneBeforeWrite(dx, dy, w, h, "a copy over it");
	if (anyCpuNewer())
		flushUploads();
	refreshScaled(sx, sy, w, h);
	refreshScaled(dx, dy, w, h);
	// the source not drawn since the last pack: the raw mirror copied too, so the destination needs no pack (CLUT
	// animations copy palettes every frame)
	const bool packed = !overlapsDirty(sx, sy, w, h);
	endPass();
	Slot& s = slot();
	g.copies++;
	const uint32_t stamp = stampOpen(s, StampCopy);
	const int S = g.scale;
	const vk::ImageSubresourceLayers layers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
	s.cmd->copyImage(*g.scaled.image, vk::ImageLayout::eGeneral, *g.copy.image, vk::ImageLayout::eGeneral,
			vk::ImageCopy(layers, { sx * S, sy * S, 0 }, layers, { 0, 0, 0 }, { (uint32_t)(w * S), (uint32_t)(h * S), 1 }));
	barrier(*s.cmd);
	s.cmd->copyImage(*g.copy.image, vk::ImageLayout::eGeneral, *g.scaled.image, vk::ImageLayout::eGeneral,
			vk::ImageCopy(layers, { 0, 0, 0 }, layers, { dx * S, dy * S, 0 }, { (uint32_t)(w * S), (uint32_t)(h * S), 1 }));
	barrier(*s.cmd);
	if (packed)
	{
		s.cmd->copyImage(*g.raw.image, vk::ImageLayout::eGeneral, *g.rawCopy.image, vk::ImageLayout::eGeneral,
				vk::ImageCopy(layers, { sx, sy, 0 }, layers, { 0, 0, 0 }, { (uint32_t)w, (uint32_t)h, 1 }));
		barrier(*s.cmd);
		s.cmd->copyImage(*g.rawCopy.image, vk::ImageLayout::eGeneral, *g.raw.image, vk::ImageLayout::eGeneral,
				vk::ImageCopy(layers, { 0, 0, 0 }, layers, { dx, dy, 0 }, { (uint32_t)w, (uint32_t)h, 1 }));
		barrier(*s.cmd);
	}
	stampClose(s, stamp);
	if (packed)
	{
		// both images hold it: only PSXRecomp's copy is stale there
		invalidateTextures(dx, dy, w, h);
		tiles(dx, dy, w, h, [](int t) { g.gpuNewer[t >> 6] |= 1ull << (t & 63); });
		g.cpuStale = true;
	}
	else
		markDirty(dx, dy, w, h);
	sceneAfterWrite(dx, dy, w, h, "a copy");
}

void bFlatTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c)
{
	const Vertex v[3] = { corner(x0, y0, c), corner(x1, y1, c), corner(x2, y2, c) };
	primitive(false, v, 3, 0, 0, 0);
}
void bGouraudTriangle(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1, int x2, int y2, uint16_t c2)
{
	const Vertex v[3] = { corner(x0, y0, c0), corner(x1, y1, c1), corner(x2, y2, c2) };
	primitive(false, v, 3, 0, 0, 0);
}
void bTexturedTriangle(int x0, int y0, int u0, int v0, int x1, int y1, int u1, int v1, int x2, int y2, int u2, int v2, uint16_t cx,
		uint16_t cy, uint16_t page)
{
	const uint32_t m = modulation();
	const Vertex v[3] = { corner(x0, y0, u0, v0, m, page, cx, cy, g.modRaw), corner(x1, y1, u1, v1, m, page, cx, cy, g.modRaw),
		corner(x2, y2, u2, v2, m, page, cx, cy, g.modRaw) };
	primitive(true, v, 3, page, cx, cy);
}
void bShadedTexturedTriangle(int x0, int y0, int u0, int v0, uint32_t c0, int x1, int y1, int u1, int v1, uint32_t c1, int x2, int y2,
		int u2, int v2, uint32_t c2, uint16_t cx, uint16_t cy, uint16_t page, int raw)
{
	const Vertex v[3] = { corner(x0, y0, u0, v0, c0, page, cx, cy, raw), corner(x1, y1, u1, v1, c1, page, cx, cy, raw),
		corner(x2, y2, u2, v2, c2, page, cx, cy, raw) };
	primitive(true, v, 3, page, cx, cy);
}
void quad(bool textured, const Vertex& a, const Vertex& b, const Vertex& c, const Vertex& d, uint16_t page, uint16_t cx, uint16_t cy)
{
	const Vertex v[6] = { a, b, c, b, d, c };
	primitive(textured, v, 6, page, cx, cy);
}
// A plugin's polygons in one block (replay.h): opaque, textured, with depths, in one draw area: what primitive() works out for
// each of its triangles, here once (the scene, the area, the marking of tiles) or once per texture page. Anything else
// (blended, a page not in the cache yet, no scene) goes through primitive() as usual.
void bulkTriangles(const uint8_t *bytes, int count)
{
	const auto one = [](const DeadstationTriangle& t) {
		bSetSemi(t.semi != 0, t.semi != 0 ? t.semi - 1 : 0);
		const float behind = 32.f * (float)t.layer;
		bSetTriangleDepth(1, t.z[0] + behind, t.z[1] + behind, t.z[2] + behind);
		bSetPrecise(1, (int32_t)(t.x[0] * 65536.f), (int32_t)(t.y[0] * 65536.f), (int32_t)(t.x[1] * 65536.f), (int32_t)(t.y[1] * 65536.f),
				(int32_t)(t.x[2] * 65536.f), (int32_t)(t.y[2] * 65536.f));
		bShadedTexturedTriangle((int)std::lround(t.x[0]), (int)std::lround(t.y[0]), t.uv[0] & 0xFF, t.uv[0] >> 8, t.rgb[0], (int)std::lround(t.x[1]),
				(int)std::lround(t.y[1]), t.uv[1] & 0xFF, t.uv[1] >> 8, t.rgb[1], (int)std::lround(t.x[2]), (int)std::lround(t.y[2]), t.uv[2] & 0xFF,
				t.uv[2] >> 8, t.rgb[2], (uint16_t)((t.clut & 0x3F) * 16), (uint16_t)((t.clut >> 6) & 0x1FF), t.page, t.raw);
	};
	DeadstationTriangle t;
	if (!g.ready)
		return;
	if (anyCpuNewer())
		flushUploads();
	const vk::Rect2D sc = scissor();
	if (sc.extent.width == 0 || sc.extent.height == 0)
		return;
	const int ax0 = sc.offset.x / g.scale, ay0 = sc.offset.y / g.scale;
	const int ax1 = ax0 + (int)sc.extent.width / g.scale, ay1 = ay0 + (int)sc.extent.height / g.scale;
	enableStencil();
	const int si = g.window == 0 ? sceneForArea(ax0, ay0, ax1, ay1) : -1;
	if (si < 0)
	{
		for (int i = 0; i < count; i++, bytes += sizeof(t))
		{
			std::memcpy(&t, bytes, sizeof(t));
			one(t);
		}
		return;
	}
	State::Scene& S = g.scenes[si];
	sceneSync(S);
	bSetSemi(0, 0);
	replayStart(si);
	const bool recording = g.replayRecording == si;
	State::Replay& record = g.replays[si];
	const float kx = 1024.f / (float)S.aw, ky = 512.f / (float)S.ah, ax = (float)S.ax, ay = (float)S.ay;
	const bool perspective = g.scenePerspective && g.frameIs3D;
	const uint32_t baseKey = SceneBit | stencilBits() | 2u << 20;
	const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
	const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
	const vk::Rect2D box({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
	Push p {};
	p.flags[0] = g.maskSet, p.flags[2] = g.scale;
	if (g.fogMost > 0)
		p.rect[0] = g.fogRgb[0], p.rect[1] = g.fogRgb[1], p.rect[2] = g.fogRgb[2], p.rect[3] = 1.f;
	uint32_t memoKey = ~0u;
	int memoSlot = -1;
	uint8_t memoClass = 7;
	float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
	const uint8_t tag = (uint8_t)(g.replayObject & 0x7F);
	Slot *s = &slot();
	if (recording)
	{
		record.vertices.reserve(record.vertices.size() + (size_t)count * 3);
		record.z.reserve(record.z.size() + (size_t)count * 3);
		record.object.reserve(record.object.size() + (size_t)count * 3);
	}
	for (int i = 0; i < count; i++, bytes += sizeof(t))
	{
		std::memcpy(&t, bytes, sizeof(t));
		const bool painted = t.semi == DeadstationPainted;
		if (t.semi != 0 && !painted)
		{
			one(t);
			s = &slot();
			continue;
		}
		const uint16_t clutX = (uint16_t)((t.clut & 0x3F) * 16), clutY = (uint16_t)((t.clut >> 6) & 0x1FF);
		const uint32_t depth = (t.page >> 7) & 3;
		const uint32_t pageKey = (uint32_t)t.page | (uint32_t)t.clut << 16;
		if (pageKey != memoKey)
		{
			memoKey = pageKey;
			// its texels or CLUT in a scene (a frame read as a texture): VRAM given what the scene drew first
			const PageArea a = pageArea(depth, t.page, clutX, clutY);
			const int texels[4] = { a.px, a.py, a.pw, 256 }, clutArea[4] = { a.cx, a.cy, a.cw, 1 };
			for (State::Scene& scn : g.scenes)
				if (scn.used && scn.newer)
				{
					sceneResolve(scn, "a texture", texels);
					if (depth < 2)
						sceneResolve(scn, "a CLUT", clutArea);
				}
			memoSlot = cachedSlot(depth > 2 ? 2 : depth, t.page, clutX, clutY);
			memoClass = memoSlot >= 0 ? g.texSlots[memoSlot].texels : 7;
			s = &slot();
		}
		if (memoSlot < 0)
		{
			one(t);
			s = &slot();
			continue;
		}
		// no discard unless a transparent texel lies under it
		uint32_t kind = Lean;
		if (memoClass != 7)
		{
			if (!(memoClass & 1))
				kind = LeanOpaque;
			else
			{
				const auto& info = g.texSlots[memoSlot];
				const int u0 = t.uv[0] & 0xFF, u1 = t.uv[1] & 0xFF, u2 = t.uv[2] & 0xFF, v0 = t.uv[0] >> 8, v1 = t.uv[1] >> 8, v2 = t.uv[2] >> 8;
				const int cx0 = std::min({ u0, u1, u2 }) >> 4, cx1 = std::max({ u0, u1, u2 }) >> 4, cy0 = std::min({ v0, v1, v2 }) >> 4, cy1 = std::max({ v0, v1, v2 }) >> 4;
				const uint16_t columns = (uint16_t)(((2u << cx1) - 1u) & ~((1u << cx0) - 1u));
				bool clear = false;
				for (int r = cy0; r <= cy1 && !clear; r++)
					clear = (info.clearCells[r] & columns) != 0;
				if (!clear)
					kind = LeanOpaque;
			}
		}
		if (s->vertexUsed + 3 * sizeof(Vertex) > VertexBytes)
		{
			submit();
			s = &slot();
		}
		Vertex built[3];
		float zs[3];
		for (int j = 0; j < 3; j++)
		{
			Vertex& v = built[j];
			v.x = (t.x[j] - ax) * kx, v.y = (t.y[j] - ay) * ky;
			v.u = (float)(t.uv[j] & 0xFF), v.v = (float)(t.uv[j] >> 8);
			v.r = t.rgb[j] & 255, v.g = (t.rgb[j] >> 8) & 255, v.b = (t.rgb[j] >> 16) & 255, v.a = 0;
			v.page = t.page, v.clutX = (uint16_t)memoSlot, v.clutY = clutY, v.flags = t.raw ? 1 : 0;
			tintCorner(v);
			const float z = t.z[j], zd = z + 32.f * (float)t.layer;
			zs[j] = z;
			if (perspective)
				v.depth = scenePerspectiveDepth(zd), v.w = std::max(z, 1.f);
			else
				v.depth = sceneFlatDepth(zd), v.w = 1.f;
			if (g.fogMost > 0)
				v.a = (uint8_t)std::min(g.fogMost * std::sqrt(std::clamp((z - g.fogNear) * g.fogScale, 0.f, 1.f)) * 255.f, 255.f);
			bx0 = std::min(bx0, t.x[j]), by0 = std::min(by0, t.y[j]), bx1 = std::max(bx1, t.x[j]), by1 = std::max(by1, t.y[j]);
		}
		std::memcpy(s->vertexMap + s->vertexUsed, built, sizeof(built));
		const uint32_t first = (uint32_t)(s->vertexUsed / sizeof(Vertex));
		s->vertexUsed += sizeof(built);
		const uint32_t key = painted ? kind | SceneBit | stencilBits() | 1u << 20 : kind | baseKey;
		State::SceneDraw *last = g.sceneDraws.empty() ? nullptr : &g.sceneDraws.back();
		if (last != nullptr && last->scene == si && last->count != 0 && last->key == key && !last->twoPasses && last->scissor == box &&
				last->first + last->count == first && memcmp(&last->push, &p, sizeof(p)) == 0)
			last->count += 3;
		else
			g.sceneDraws.push_back({ si, first, 3, key, box, p, false, 0 });
		if (recording)
		{
			const uint32_t base = (uint32_t)record.vertices.size();
			record.vertices.insert(record.vertices.end(), built, built + 3);
			record.z.insert(record.z.end(), zs, zs + 3);
			record.object.insert(record.object.end(), 3, (uint8_t)(tag | (t.layer > 0 ? t.layer & 15 : 0) << 3));
			State::SceneDraw *r = record.draws.empty() ? nullptr : &record.draws.back();
			if (r != nullptr && r->count != 0 && r->key == key && !r->twoPasses && r->scissor == box && r->first + r->count == base)
				r->count += 3;
			else
				record.draws.push_back({ si, base, 3, key, box, p, false, 0 });
		}
		g.vertices += 3;
		g.sceneDrawCount++;
		g.drawnSinceShown++;
	}
	S.any3D = true;
	S.lastUsed = g.texFrame;
	if (bx1 >= bx0)
		sceneMarkDrawn(S, std::max((int)bx0, ax0), std::max((int)by0, ay0), std::min((int)bx1 + 1, ax1), std::min((int)by1 + 1, ay1));
}

// A plugin's added light (replay.h): one draw into the scene, in the shader the PlayStation's untextured polygons use, at full colour
// mode: 0 light added, 1 a veil, 2 a shadow darkening, 3 the stencil marked alone, 4 depth alone
void bulkGlows(const uint8_t *bytes, int count, int mode)
{
	if (!g.ready || count <= 0)
		return;
	const bool veil = mode == 1, shadow = mode == 2 || mode == 3;
	if (shadow && !g.stencilOn)
	{
		// (the scene's stencil, as the PlayStation's mask bit has it)
		const bool set = g.maskSet, check = g.maskCheck;
		g.maskSet = g.maskCheck = true;
		enableStencil();
		g.maskSet = set, g.maskCheck = check;
	}
	if (anyCpuNewer())
		flushUploads();
	const vk::Rect2D sc = scissor();
	if (sc.extent.width == 0 || sc.extent.height == 0)
		return;
	const int ax0 = sc.offset.x / g.scale, ay0 = sc.offset.y / g.scale;
	const int ax1 = ax0 + (int)sc.extent.width / g.scale, ay1 = ay0 + (int)sc.extent.height / g.scale;
	const int si = g.window == 0 ? sceneForArea(ax0, ay0, ax1, ay1) : -1;
	if (si < 0)
		return;
	State::Scene& S = g.scenes[si];
	sceneSync(S);
	replayStart(si);
	const bool recording = g.replayRecording == si;
	State::Replay& record = g.replays[si];
	const uint32_t recordFirst = (uint32_t)record.vertices.size();
	const uint8_t tag = (uint8_t)(g.replayObject & 0x7F);
	const bool perspective = g.scenePerspective && g.frameIs3D;
	bool anyDepth = false;
	const float kx = 1024.f / (float)S.aw, ky = 512.f / (float)S.ah, ox = (float)S.ax, oy = (float)S.ay;
	const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
	const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
	const vk::Rect2D box({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
	Push p {};
	p.flags[0] = g.maskSet, p.flags[2] = g.scale;
	const size_t bytesNeeded = ((size_t)count * 3 + 6) * sizeof(Vertex);
	Slot *s = &slot();
	if (s->vertexUsed + bytesNeeded > VertexBytes)
	{
		submit();
		s = &slot();
	}
	const uint32_t first = (uint32_t)(s->vertexUsed / sizeof(Vertex));
	float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
	uint32_t shadowRgb = 0;
	Vertex built[3];
	for (int i = 0; i < count; i++, bytes += sizeof(DeadstationGlow))
	{
		DeadstationGlow t;
		std::memcpy(&t, bytes, sizeof(t));
		shadowRgb = t.rgb[0];
		for (int j = 0; j < 3; j++)
		{
			Vertex& v = built[j];
			v = Vertex {};
			v.x = (t.x[j] - ox) * kx, v.y = (t.y[j] - oy) * ky;
			v.r = t.rgb[j] & 255, v.g = (t.rgb[j] >> 8) & 255, v.b = (t.rgb[j] >> 16) & 255;
			if (t.z[j] > 0)
			{
				anyDepth = true;
				if (perspective)
					v.depth = scenePerspectiveDepth(t.z[j]), v.w = std::max(t.z[j], 1.f);
				else
					v.depth = sceneFlatDepth(t.z[j]);
			}
			bx0 = std::min(bx0, t.x[j]), by0 = std::min(by0, t.y[j]), bx1 = std::max(bx1, t.x[j]), by1 = std::max(by1, t.y[j]);
			if (recording)
			{
				record.vertices.push_back(v);
				record.z.push_back(veil ? 0.f : t.z[j]);	// (a veil keeps its place: its corners are the screen's)
				record.object.push_back(tag);
			}
		}
		std::memcpy(s->vertexMap + s->vertexUsed, built, sizeof(built));
		s->vertexUsed += sizeof(built);
	}
	// added: the PlayStation's blend mode 1
	constexpr uint32_t Added = 1 + 1;
	const uint32_t blendKey = mode == 5 ? 0u : mode == 1 ? (uint32_t)VeilBlend : mode >= 2 ? (uint32_t)BlockBlend : Added;
	const uint32_t key = (mode == 2 ? StencilMark : 0u) | Flat | blendKey << 4 | SceneBit | (shadow ? 3u << 8 | 1u << 12 : stencilBits()) | (mode == 4 ? 2u << 20 : anyDepth ? 1u << 20 : 0u);
	g.sceneDraws.push_back({ si, first, (uint32_t)count * 3, key, box, p, false, 0 });
	if (recording)
		record.draws.push_back({ si, recordFirst, (uint32_t)count * 3, key, box, p, false, 0 });
	// A shadow: hundreds of polygons over each other, each one a blended object to this GPU if it were darkened as it's drawn: 8 ms a frame close up. So they only mark the stencil,
	// which runs no shader, and one quad over them darkens what they marked.
	if (mode == 2)
	{
		const float qx0 = (std::floor(bx0) - ox) * kx, qy0 = (std::floor(by0) - oy) * ky, qx1 = (std::ceil(bx1) - ox) * kx, qy1 = (std::ceil(by1) - oy) * ky;
		const float xs[6] = { qx0, qx1, qx0, qx1, qx1, qx0 }, ys[6] = { qy0, qy0, qy1, qy0, qy1, qy1 };
		Vertex quadVertices[6];
		for (int j = 0; j < 6; j++)
		{
			Vertex& v = quadVertices[j];
			v = Vertex {};
			v.x = xs[j], v.y = ys[j];
			// (the game's shade, 0 under a bright sun's hard shadow, up to a soft overcast one's grey: half of what's there, up to three quarters)
			v.r = (uint8_t)std::min(128u + (shadowRgb & 255), 255u), v.g = (uint8_t)std::min(128u + ((shadowRgb >> 8) & 255), 255u), v.b = (uint8_t)std::min(128u + ((shadowRgb >> 16) & 255), 255u);
			v.depth = 0, v.w = 1.f;
		}
		const uint32_t quadFirst = (uint32_t)(s->vertexUsed / sizeof(Vertex));
		std::memcpy(s->vertexMap + s->vertexUsed, quadVertices, sizeof(quadVertices));
		s->vertexUsed += sizeof(quadVertices);
		const uint32_t quadKey = StencilMarked | Flat | (uint32_t)DarkenBlend << 4 | SceneBit | 3u << 8 | 1u << 12;
		g.sceneDraws.push_back({ si, quadFirst, 6, quadKey, box, p, false, 0 });
		if (recording)
			g.replays[si].draws.push_back({ si, quadFirst, 6, quadKey, box, p, false, 0 });
		g.vertices += 6;
	}
	g.vertices += (uint32_t)count * 3;
	g.sceneDrawCount++;
	g.drawnSinceShown++;
	S.lastUsed = g.texFrame;
	sceneMarkDrawn(S, std::max((int)bx0, ax0), std::max((int)by0, ay0), std::min((int)bx1 + 1, ax1), std::min((int)by1 + 1, ay1));
}

void bFlatRect(int x, int y, int w, int h, uint16_t c)
{
	if (w <= 0 || h <= 0)
		return;
	quad(false, corner(x, y, c), corner(x + w, y, c), corner(x, y + h, c), corner(x + w, y + h, c), 0, 0, 0);
}
void bTexturedRectScaled(int x, int y, int w, int h, int u0, int v0, int u1, int v1, uint16_t cx, uint16_t cy, uint16_t page)
{
	if (w <= 0 || h <= 0)
		return;
	const uint32_t m = modulation();
	quad(true, corner(x, y, u0, v0, m, page, cx, cy, g.modRaw), corner(x + w, y, u1, v0, m, page, cx, cy, g.modRaw),
			corner(x, y + h, u0, v1, m, page, cx, cy, g.modRaw), corner(x + w, y + h, u1, v1, m, page, cx, cy, g.modRaw), page, cx, cy);
}
void bTexturedRect(int x, int y, int w, int h, int u, int v, uint16_t cx, uint16_t cy, uint16_t page)
{
	bTexturedRectScaled(x, y, w, h, u, v, u + w, v + h, cx, cy, page);
}
// a line one VRAM pixel thick, as a quad over both ends
void line(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1)
{
	if (std::abs(x1 - x0) >= std::abs(y1 - y0))
	{
		if (x1 < x0)
			std::swap(x0, x1), std::swap(y0, y1), std::swap(c0, c1);
		quad(false, corner(x0, y0, c0), corner(x1 + 1, y1, c1), corner(x0, y0 + 1, c0), corner(x1 + 1, y1 + 1, c1), 0, 0, 0);
	}
	else
	{
		if (y1 < y0)
			std::swap(x0, x1), std::swap(y0, y1), std::swap(c0, c1);
		quad(false, corner(x0, y0, c0), corner(x0 + 1, y0, c0), corner(x1, y1 + 1, c1), corner(x1 + 1, y1 + 1, c1), 0, 0, 0);
	}
}
void bLine(int x0, int y0, int x1, int y1, uint16_t c) { line(x0, y0, c, x1, y1, c); }
void bShadedLine(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1) { line(x0, y0, c0, x1, y1, c1); }

int bRenderDisplay(uint32_t *out, int pitch, int dx, int dy, int dw, int dh)
{
	readback(0, 0, VramW, VramH);
	return sw_render_display(out, pitch, dx, dy, dw, dh);
}
void bVramWrite(int x, int y, uint16_t pixel) { upload(x, y, 1, 1, &pixel); }
uint16_t bVramRead(int x, int y)
{
	readback(x, y, 1, 1);
	return g.mirror[(y & (VramH - 1)) * VramW + (x & (VramW - 1))];
}
void bTransferIn(int x, int y, int w, int h, const uint16_t *data) { upload(x, y, w, h, data); }
void bTransferOut(int x, int y, int w, int h, uint16_t *data)
{
	readback(x, y, w, h);
	for (int j = 0; j < h; j++)
		for (int i = 0; i < w; i++)
			data[j * w + i] = g.mirror[((y + j) & (VramH - 1)) * VramW + ((x + i) & (VramW - 1))];
}
void bSetArea(int x1, int y1, int x2, int y2) { g.area[0] = x1, g.area[1] = y1, g.area[2] = x2, g.area[3] = y2; }
void bGetArea(int *x1, int *y1, int *x2, int *y2) { *x1 = g.area[0], *y1 = g.area[1], *x2 = g.area[2], *y2 = g.area[3]; }
void bSetOffset(int, int) { }
void bWideConfigure(int, int) { }
void bWideView(int, int, int, int) { }
void bWideTarget(int) { }
void bWideNoTarget() { }
void bWideClear(int, int, int, uint16_t) { }
void bWideMargins(int, int, int, uint16_t, int) { }
int bWideDisplay(uint32_t *, int, int, int, int) { return 0; }
int bWideDump(uint32_t *, int, int *, int *, int) { return 0; }

// ---- the render side of PSXRecomp's Vulkan entry points ----

std::vector<uint32_t> black(64 * 64, 0);

void shutdownGpu()
{
	if (!g.ready)
		return;
	submit();
	hostVideo().submitDrain();
	g.dev.device.waitIdle();
}

// what a polygon shows of its page; a part nobody was showing may be out of date
inline void worldUses(int slot, int u0, int v0, int u1, int v1)
{
	auto& t = g.texSlots[slot];
	const uint16_t columns = (uint16_t)(((2u << (u1 >> 4)) - 1u) & ~((1u << (u0 >> 4)) - 1u));
	bool fresh = false;
	for (int r = v0 >> 4; r <= (v1 >> 4); r++)
		if ((t.usedCells[r] & columns) != columns)
			t.usedCells[r] |= columns, fresh = true;
	if (fresh && g.worldBuffer)
		t.valid = false, t.changed[2] = t.changed[3] = -1, t.changedFrame = g.texFrame;
}

// a page through a CLUT, in a slot that stays hers for good, decoded
int worldSlot(uint16_t page, uint16_t clut)
{
	const uint16_t clutX = (uint16_t)((clut & 0x3F) * 16), clutY = (uint16_t)((clut >> 6) & 0x1FF);
	const uint32_t depth = std::min((page >> 7) & 3u, 2u);
	const uint32_t key = (page & 0x1Fu) | depth << 5 | (depth < 2 ? ((uint32_t)(clutX >> 4) & 63u) << 7 | ((uint32_t)clutY & 511u) << 13 : 0u);
	auto have = g.texSlotOf.find(key);
	if (have == g.texSlotOf.end() || !g.texSlots[have->second].kept)
	{
		g.takingForWorld = true;
		cachedSlot(depth, page, clutX, clutY);
		g.takingForWorld = false;
		have = g.texSlotOf.find(key);
		if (have == g.texSlotOf.end())
			return -1;
		g.texSlots[have->second].kept = true;
		std::fill(std::begin(g.texSlots[have->second].usedCells), std::end(g.texSlots[have->second].usedCells), 0);
		g.texSlots[have->second].usedElsewhere = false;
		g.worldSlots.push_back(have->second);
	}
	if (!g.texSlots[have->second].valid)
		decodeForced(have->second);
	return have->second;
}

// How many of a page's texels are transparent, over any rectangle of it at once: a table of the running count (0..256 each way)
void pageClearCounts(int depth, uint16_t page, uint16_t clutX, uint16_t clutY, std::vector<uint32_t>& counts)
{
	counts.assign(257 * 257, 0);
	const PageArea a = pageArea((uint32_t)depth, page, clutX, clutY);
	bool clut[256] = {};
	if (depth < 2)
		for (int i = 0; i < (depth == 0 ? 16 : 256); i++)
			clut[i] = g.mirror[a.cy * VramW + ((a.cx + i) & (VramW - 1))] == 0;
	for (int v = 0; v < 256; v++)
	{
		const uint16_t *row = &g.mirror[(a.py + v) * VramW];
		uint32_t along = 0;
		for (int u = 0; u < 256; u++)
		{
			bool clear;
			if (depth == 0)
				clear = clut[(row[(a.px + (u >> 2)) & (VramW - 1)] >> ((u & 3) * 4)) & 15];
			else if (depth == 1)
				clear = clut[(row[(a.px + (u >> 1)) & (VramW - 1)] >> ((u & 1) * 8)) & 255];
			else
				clear = row[(a.px + u) & (VramW - 1)] == 0;
			along += clear ? 1 : 0;
			counts[(size_t)(v + 1) * 257 + (size_t)u + 1] = counts[(size_t)v * 257 + (size_t)u + 1] + along;
		}
	}
}

// A world's mesh (replay.h): its polygons into their buffer, by kind and cell, each naming the atlas slot it'll always find its texture in
void bWorldLoad(const DeadstationWorldTriangle *triangles, size_t count, unsigned cells)
{
	if (!g.ready)
		return;
	submit();
	g.dev.device.waitIdle();
	g.worldBuffer.reset();
	g.worldMemory.reset();
	for (auto& t : g.texSlots)
		t.kept = false;
	g.worldSlots.clear();
	for (auto& runs : g.worldRuns)
		runs.assign(cells, State::WorldRun {});
	if (count == 0)
		return;
	if (anyCpuNewer())
		flushUploads();
	std::unordered_map<uint32_t, int> slotOf;
	std::vector<int> slotFor(count);
	for (size_t i = 0; i < count; i++)
	{
		const DeadstationWorldTriangle& t = triangles[i];
		const uint32_t pageKey = (uint32_t)t.page | (uint32_t)t.clut << 16;
		auto found = slotOf.find(pageKey);
		if (found == slotOf.end())
		{
			found = slotOf.emplace(pageKey, worldSlot(t.page, t.clut)).first;
			if (found->second >= 0)
			{
				// (A page's texels are what was uploaded to it, in the CPU's copy, whatever's been drawn elsewhere in its tiles since VRAM was last read, and a game that never reads VRAM back
				// never clears those marks: without this every page would count as unknown, and every polygon be drawn as cut out. Reading it back is no answer: that packs what's in the
				// scaled image over the pages.)
				auto& info = g.texSlots[found->second];
				info.texels = texelClasses(info.depth, info.page, info.clutX, info.clutY, info.clearCells, true);
			}
		}
		slotFor[i] = found->second;
	}
	// Each polygon by what it shows. A discard turns off this GPU's hidden surface removal and costs every pixel it's asked about, in a jungle most of the frame, whatever it ends up showing. So:
	// 0 with nothing else under it, not drawn. 1 with no transparent texel under it or within 2 texels (a filter reaches that far), solid. 3 with none under it but some beside it, or a few
	// specks, solid too, far off averaged by what's not transparent (psx_gpu_world_edge.frag). 2 the rest, cut out. One that's largely holes, a frond, a fence, is cut along its page's 16 x 16
	// texel cells, for the same to go for each part of it. One that's not isn't: it's likely part of a surface, and any point added along its edge can open a crack against the polygon next to
	// it, that has no such point.
	size_t classes[6] = {};
	struct Piece
	{
		WorldVertex v[3];
		uint16_t cell;
		uint8_t kind;
	};
	std::vector<Piece> pieces;
	pieces.reserve(count + count / 4);
	std::unordered_map<int, std::vector<uint32_t>> clearCountsOf;
	for (const auto& found : slotOf)
		if (found.second >= 0 && clearCountsOf.find(found.second) == clearCountsOf.end())
		{
			const auto& info = g.texSlots[found.second];
			pageClearCounts((int)info.depth, info.page, info.clutX, info.clutY, clearCountsOf[found.second]);
		}
	size_t split = 0;
	for (size_t i = 0; i < count; i++)
	{
		const DeadstationWorldTriangle& t = triangles[i];
		if (slotFor[i] < 0 || t.cell >= cells)
			continue;
		const auto& info = g.texSlots[slotFor[i]];
		// (where the game will load other texels, anything can come to be cut out: as if mixed)
		uint16_t animated[16] = {};
		if (!g.animatedRects.empty() && info.texels != 7)
		{
			const PageArea area = pageArea(info.depth, info.page, info.clutX, info.clutY);
			const int perWord = info.depth == 0 ? 4 : info.depth == 1 ? 2 : 1;
			for (size_t r = 0; r + 4 <= g.animatedRects.size(); r += 4)
			{
				const int x0 = (g.animatedRects[r] - area.px) * perWord, x1 = (g.animatedRects[r] + g.animatedRects[r + 2] - area.px) * perWord - 1;
				const int y0 = g.animatedRects[r + 1] - area.py, y1 = y0 + g.animatedRects[r + 3] - 1;
				if (x1 < 0 || x0 > 255 || y1 < 0 || y0 > 255)
					continue;
				const uint16_t columns = (uint16_t)(((2u << (std::min(x1, 255) >> 4)) - 1u) & ~((1u << (std::max(x0, 0) >> 4)) - 1u));
				for (int row = std::max(y0, 0) >> 4; row <= (std::min(y1, 255) >> 4); row++)
					animated[row] |= columns;
			}
		}
		const std::vector<uint32_t>& clearCounts = clearCountsOf[slotFor[i]];
		const auto clearIn = [&](int u0, int v0, int u1, int v1) {	// the texels u0..u1, v0..v1 inclusive
			u0 = std::max(u0, 0), v0 = std::max(v0, 0), u1 = std::min(u1, 255), v1 = std::min(v1, 255);
			return clearCounts[(size_t)(v1 + 1) * 257 + (size_t)u1 + 1] - clearCounts[(size_t)v0 * 257 + (size_t)u1 + 1] - clearCounts[(size_t)(v1 + 1) * 257 + (size_t)u0] + clearCounts[(size_t)v0 * 257 + (size_t)u0];
		};
		const auto animatedIn = [&](int u0, int v0, int u1, int v1) {
			for (int row = v0 >> 4; row <= (v1 >> 4); row++)
				if (animated[row] & (uint16_t)(((2u << (u1 >> 4)) - 1u) & ~((1u << (u0 >> 4)) - 1u)))
					return true;
			return false;
		};
		// 0 nothing there, 1 solid, 2 both
		const auto classOf = [&](int u0, int v0, int u1, int v1) -> int {
			if (info.texels == 7 || animatedIn(u0, v0, u1, v1))
				return 2;
			const uint32_t under = clearIn(u0, v0, u1, v1), texels = (uint32_t)((u1 - u0 + 1) * (v1 - v0 + 1));
			if (under == texels)
				return 0;
			if (clearIn(u0 - 2, v0 - 2, u1 + 2, v1 + 2) == 0)
				return 1;
			return under * 32 <= texels ? 3 : 2;
		};
		const auto kindOf = [&](int cls) { return (uint8_t)(cls == 3 ? (t.twoSided ? 5 : 4) : (cls == 2 ? 1 : 0) | (t.twoSided ? 2 : 0)); };
		{
			const int ua = std::min({ t.uv[0][0], t.uv[1][0], t.uv[2][0] }), ub = std::max({ t.uv[0][0], t.uv[1][0], t.uv[2][0] });
			const int va = std::min({ t.uv[0][1], t.uv[1][1], t.uv[2][1] }), vb = std::max({ t.uv[0][1], t.uv[1][1], t.uv[2][1] });
			worldUses(slotFor[i], ua, va, ub, vb);
		}
		const auto vertexOf = [&](int n) {
			return WorldVertex { t.p[n][0], t.p[n][1], t.p[n][2], (int16_t)((int)t.layer * 32 + t.nudge), t.uv[n][0], t.uv[n][1], (uint8_t)(t.raw ? 1 : 0), (uint8_t)slotFor[i], t.rgb[n][0], t.rgb[n][1], t.rgb[n][2], 0 };
		};
		const int u0 = std::min({ t.uv[0][0], t.uv[1][0], t.uv[2][0] }), u1 = std::max({ t.uv[0][0], t.uv[1][0], t.uv[2][0] });
		const int v0 = std::min({ t.uv[0][1], t.uv[1][1], t.uv[2][1] }), v1 = std::max({ t.uv[0][1], t.uv[1][1], t.uv[2][1] });
		const int cu0 = u0 >> 4, cu1 = u1 >> 4, cv0 = v0 >> 4, cv1 = v1 >> 4;
		const double uA = (double)t.uv[1][0] - t.uv[0][0], vA = (double)t.uv[1][1] - t.uv[0][1], uB = (double)t.uv[2][0] - t.uv[0][0], vB = (double)t.uv[2][1] - t.uv[0][1];
		const double area2 = uA * vB - uB * vA;
		const int whole = classOf(u0, v0, u1, v1);
		const uint32_t texelsUnder = (uint32_t)((u1 - u0 + 1) * (v1 - v0 + 1));
		if (whole != 2 || std::fabs(area2) < 1.0 || (cu0 == cu1 && cv0 == cv1) || info.texels == 7 || animatedIn(u0, v0, u1, v1) || clearIn(u0, v0, u1, v1) * 4 < texelsUnder)
		{
			const int cls = whole;
			classes[cls]++;
			if (cls == 0)
			{
				continue;
			}
			pieces.push_back({ { vertexOf(0), vertexOf(1), vertexOf(2) }, t.cell, kindOf(cls) });
			continue;
		}
		split++;
		// the same plane and the same map, cut along the page's cells. A point's place comes from its (u, v), each kept to a fine grid, so pieces sharing it share it exactly. Where it's on the
		// polygon's own edge, its place rounds out from that edge, axis by axis as the edge's normal in the polygon's plane goes: a whole number, and no crack against what's next to it.
		const double centre[3] = { ((double)t.p[0][0] + t.p[1][0] + t.p[2][0]) / 3, ((double)t.p[0][1] + t.p[1][1] + t.p[2][1]) / 3, ((double)t.p[0][2] + t.p[1][2] + t.p[2][2]) / 3 };
		const auto pointAt = [&](double u, double v) {
			u = std::round(u * 4096.0) / 4096.0, v = std::round(v * 4096.0) / 4096.0;
			const double du = u - t.uv[0][0], dv = v - t.uv[0][1];
			const double b1 = (du * vB - uB * dv) / area2, b2 = (uA * dv - du * vA) / area2, b0 = 1.0 - b1 - b2;
			const int edgeOpposite = b0 < 0.0005 ? 0 : b1 < 0.0005 ? 1 : b2 < 0.0005 ? 2 : -1;
			double at[3], outward[3] = {};
			for (int n = 0; n < 3; n++)
				at[n] = b0 * t.p[0][n] + b1 * t.p[1][n] + b2 * t.p[2][n];
			if (edgeOpposite >= 0)
			{
				const int e0 = (edgeOpposite + 1) % 3, e1 = (edgeOpposite + 2) % 3;
				double e[3], w[3], ee = 0, we = 0;
				for (int n = 0; n < 3; n++)
				{
					e[n] = (double)t.p[e1][n] - t.p[e0][n], w[n] = at[n] - centre[n];
					ee += e[n] * e[n], we += w[n] * e[n];
				}
				for (int n = 0; n < 3; n++)
					outward[n] = w[n] - (ee > 0 ? we / ee : 0) * e[n];
			}
			WorldVertex out = vertexOf(0);
			int16_t *const place[3] = { &out.x, &out.y, &out.z };
			for (int n = 0; n < 3; n++)
				*place[n] = (int16_t)(edgeOpposite < 0 ? std::round(at[n]) : outward[n] >= 0 ? std::ceil(at[n] - 1e-6) : std::floor(at[n] + 1e-6));
			out.u = (uint8_t)std::clamp((int)std::lround(u), 0, 255), out.v = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
			out.r = (uint8_t)std::clamp((int)std::lround(b0 * t.rgb[0][0] + b1 * t.rgb[1][0] + b2 * t.rgb[2][0]), 0, 255);
			out.g = (uint8_t)std::clamp((int)std::lround(b0 * t.rgb[0][1] + b1 * t.rgb[1][1] + b2 * t.rgb[2][1]), 0, 255);
			out.b = (uint8_t)std::clamp((int)std::lround(b0 * t.rgb[0][2] + b1 * t.rgb[1][2] + b2 * t.rgb[2][2]), 0, 255);
			return out;
		};
		struct UV { double u, v; };
		static std::vector<UV> shape, cutTo;
		const auto clipTo = [&](int axis, double bound, bool keepBelow) {
			cutTo.clear();
			for (size_t n = 0; n < shape.size(); n++)
			{
				const UV a = shape[n], b = shape[(n + 1) % shape.size()];
				const double da = (axis == 0 ? a.u : a.v) - bound, db = (axis == 0 ? b.u : b.v) - bound;
				const bool inA = keepBelow ? da <= 0 : da >= 0, inB = keepBelow ? db <= 0 : db >= 0;
				if (inA)
					cutTo.push_back(a);
				if (inA != inB)
				{
					const double f = da / (da - db);
					cutTo.push_back({ axis == 0 ? bound : a.u + (b.u - a.u) * f, axis == 1 ? bound : a.v + (b.v - a.v) * f });
				}
			}
			shape.swap(cutTo);
		};
		for (int cv = cv0; cv <= cv1; cv++)
			for (int cu = cu0; cu <= cu1; cu++)
			{
				const int cls = classOf(cu * 16, cv * 16, cu * 16 + 15, cv * 16 + 15);
				if (cls == 0)
					continue;
				shape.clear();
				for (int n = 0; n < 3; n++)
					shape.push_back({ (double)t.uv[n][0], (double)t.uv[n][1] });
				clipTo(0, cu * 16.0, false);
				if (shape.size() >= 3)
					clipTo(0, cu * 16.0 + 16.0, true);
				if (shape.size() >= 3)
					clipTo(1, cv * 16.0, false);
				if (shape.size() >= 3)
					clipTo(1, cv * 16.0 + 16.0, true);
				if (shape.size() < 3)
					continue;
				const WorldVertex first = pointAt(shape[0].u, shape[0].v);
				WorldVertex last = pointAt(shape[1].u, shape[1].v);
				for (size_t n = 2; n < shape.size(); n++)
				{
					const WorldVertex next = pointAt(shape[n].u, shape[n].v);
					pieces.push_back({ { first, last, next }, t.cell, kindOf(cls) });
					last = next;
				}
			}
	}
	size_t used = pieces.size() * 3, cutOuts = 0;
	for (const Piece& piece : pieces)
	{
		g.worldRuns[piece.kind][piece.cell].count += 3;
		cutOuts += piece.kind == 1 || piece.kind == 3 ? 1 : 0;
	}
	uint32_t next = 0;
	for (auto& runs : g.worldRuns)
		for (State::WorldRun& r : runs)
		{
			r.first = next;
			next += r.count;
			r.count = 0;
		}
	uint8_t *map = nullptr;
	makeBuffer(g.worldBuffer, g.worldMemory, map, std::max<size_t>(used, 3) * sizeof(WorldVertex), vk::BufferUsageFlagBits::eVertexBuffer);
	WorldVertex *out = (WorldVertex *)map;
	// (each corner knows how big its triangle is, for the light at it: half its longest edge, in 8s)
	const auto sizeOf = [](const WorldVertex *v, int a, int b) {
		const float dx = (float)v[a].x - v[b].x, dy = (float)v[a].y - v[b].y, dz = (float)v[a].z - v[b].z;
		return dx * dx + dy * dy + dz * dz;
	};
	for (Piece& piece : pieces)
	{
		State::WorldRun& run = g.worldRuns[piece.kind][piece.cell];
		const uint8_t size = (uint8_t)std::min(std::sqrt(std::max({ sizeOf(piece.v, 0, 1), sizeOf(piece.v, 1, 2), sizeOf(piece.v, 2, 0) })) / 16.f, 255.f);
		piece.v[0].a = piece.v[1].a = piece.v[2].a = size;
		std::memcpy(out + run.first + run.count, piece.v, sizeof(piece.v));
		run.count += 3;
	}
	DS_LOG("recomp gpu: of %zu polygons' triangles, whole: %zu show nothing, %zu solid, %zu solid beside what's transparent or with specks, %zu cut out; %zu cut along their pages' cells; %zu triangles in all, %zu need a discard",
			count, classes[0], classes[1], classes[3], classes[2], split, pieces.size(), cutOuts);
	g.dev.device.unmapMemory(*g.worldMemory);
	DS_LOG("recomp gpu: a world of %zu triangles in %d of the atlas's slots, %.1f MB", used / 3, (int)g.worldSlots.size(), (double)(used * sizeof(WorldVertex)) / 1048576.0);
}

// the world's tint and fog, for what's drawn with the mesh's shaders, and a module's own shaders' numbers: which, 0 none, 1 the solid mesh's, 2 for the rest
void worldLook(WorldPush& push, int which = 0)
{
	for (int i = 0; i < 3; i++)
		push.a[i] = g.tinted ? (float)g.tint[i] / 256.f : 1.f, push.b[i] = g.fogRgb[i];
	push.b[3] = g.fogMost, push.c[0] = g.fogNear, push.c[1] = g.fogScale;
	// (where the device has too few bytes for them, they stay 0)
	if (g.worldPushSize < sizeof(WorldPush) || which == 0)
		return;
	std::memcpy(push.numbers, g.worldNumbers[which - 1], sizeof(push.numbers));
}

// the cells in view: a draw for each run of cells next to each other in the buffer, of each kind
void bWorldDraw(const DeadstationWorldCamera& camera, const uint16_t *cells, int count, int parts = 3)
{
	if (!g.ready || !g.worldBuffer || count <= 0)
		return;
	// its pages as they are now (a texture animated)
	decodeWorldPages();
	if (anyCpuNewer())
		flushUploads();
	const vk::Rect2D sc = scissor();
	if (sc.extent.width == 0 || sc.extent.height == 0)
		return;
	const int ax0 = sc.offset.x / g.scale, ay0 = sc.offset.y / g.scale;
	const int ax1 = ax0 + (int)sc.extent.width / g.scale, ay1 = ay0 + (int)sc.extent.height / g.scale;
	enableStencil();
	const int si = sceneForArea(ax0, ay0, ax1, ay1);
	if (si < 0)
		return;
	State::Scene& S = g.scenes[si];
	sceneSync(S);
	const float kx = 1024.f / (float)S.aw, ky = 512.f / (float)S.ah;
	WorldPush push {};
	for (int i = 0; i < 3; i++)
	{
		push.rows[4 * i] = camera.r[3 * i], push.rows[4 * i + 1] = camera.r[3 * i + 1], push.rows[4 * i + 2] = camera.r[3 * i + 2];
		push.rows[4 * i + 3] = camera.t[i];
	}
	push.project[0] = camera.h * kx / 512.f, push.project[1] = camera.h * ky / 256.f;
	push.project[2] = (camera.ofx + camera.drawX - (float)S.ax) * kx / 512.f - 1.f;
	push.project[3] = (camera.ofy + camera.drawY - (float)S.ay) * ky / 256.f - 1.f;
	// (what's solid by a module's own shaders, if it has given them and says so now)
	const bool lit = (parts & 1) != 0 && g.worldOwnSolid && g.worldOwnVert && g.worldPushSize == sizeof(WorldPush);
	worldLook(push, (parts & 1) != 0 ? 1 : 2);
	const int pushAt = (int)g.worldPushes.size();
	g.worldPushes.push_back(push);
	const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
	const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
	const vk::Rect2D box({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
	// What's solid first, in the buffer's order (the fewest draws: this GPU's hidden surface removal doesn't care for the order). Then what's cut out, near to far, as handed in: a discard
	// turns that removal off, so what it costs is what passes the depth test when it comes to it, and in a jungle that's most of the frame.
	static std::vector<uint16_t> inBufferOrder;
	inBufferOrder.assign(cells, cells + count);
	std::sort(inBufferOrder.begin(), inBufferOrder.end());
	for (int kind : { 0, 2, 4, 5, 1, 3 })
	{
		const bool cutOut = kind == 1 || kind == 3, twoSided = kind == 2 || kind == 3 || kind == 5;
		if (((deadstation_world_skip >> kind) & 1) || (cutOut ? (parts & 2) == 0 : (parts & 1) == 0))
			continue;
		const auto& runs = g.worldRuns[kind];
		const uint16_t *order = cutOut ? cells : inBufferOrder.data();
		uint32_t key = (cutOut ? World : kind >= 4 ? (lit ? WorldOwnEdge : WorldEdge) : (lit ? WorldOwnSolid : WorldOpaque)) | SceneBit | stencilBits() | 2u << 20;
		if (!twoSided)
			key |= CullBit;
		uint32_t first = 0, length = 0;
		const auto flush = [&]() {
			if (length == 0)
				return;
			State::SceneDraw d { si, first, length, key, box, Push {}, false, 0 };
			d.world = pushAt;
			g.sceneDraws.push_back(d);
			g.vertices += length;
			g.sceneDrawCount += length / 3;
			g.drawnSinceShown++;
		};
		for (int i = 0; i < count; i++)
		{
			if (order[i] >= runs.size())
				continue;
			const State::WorldRun& run = runs[order[i]];
			if (run.count == 0)
				continue;
			if (length != 0 && first + length == run.first)
				length += run.count;
			else
			{
				flush();
				first = run.first, length = run.count;
			}
		}
		flush();
	}
	S.any3D = true;
	S.lastUsed = g.texFrame;
	sceneMarkDrawn(S, ax0, ay0, ax1, ay1);
}

// A sky's mesh
void bWorldSkyLoad(const DeadstationWorldTriangle *triangles, size_t count)
{
	if (!g.ready)
		return;
	submit();
	g.dev.device.waitIdle();
	g.skyBuffer.reset();
	g.skyMemory.reset();
	g.skyVertices = 0;
	if (count == 0)
		return;
	uint8_t *map = nullptr;
	makeBuffer(g.skyBuffer, g.skyMemory, map, count * 3 * sizeof(WorldVertex), vk::BufferUsageFlagBits::eVertexBuffer);
	WorldVertex *out = (WorldVertex *)map;
	for (size_t i = 0; i < count; i++)
	{
		const DeadstationWorldTriangle& t = triangles[i];
		const bool flat = t.page == 0xFFFF;
		const int slot = flat ? 0 : worldSlot(t.page, t.clut);
		if (slot < 0)
			continue;
		if (!flat)
			worldUses(slot, std::min({ t.uv[0][0], t.uv[1][0], t.uv[2][0] }), std::min({ t.uv[0][1], t.uv[1][1], t.uv[2][1] }), std::max({ t.uv[0][0], t.uv[1][0], t.uv[2][0] }), std::max({ t.uv[0][1], t.uv[1][1], t.uv[2][1] }));
		for (int j = 0; j < 3; j++)
			out[g.skyVertices++] = { t.p[j][0], t.p[j][1], t.p[j][2], 0, t.uv[j][0], t.uv[j][1], (uint8_t)((t.raw ? 1 : 0) | (flat ? 2 : 0)), (uint8_t)slot, t.rgb[j][0], t.rgb[j][1], t.rgb[j][2], 0 };
	}
	g.dev.device.unmapMemory(*g.skyMemory);
}

// the scene being drawn into and what's common to its world draws; -1: none
int worldScene(const DeadstationWorldCamera& camera, WorldPush& push, vk::Rect2D& box)
{
	const vk::Rect2D sc = scissor();
	if (sc.extent.width == 0 || sc.extent.height == 0)
		return -1;
	const int ax0 = sc.offset.x / g.scale, ay0 = sc.offset.y / g.scale;
	const int ax1 = ax0 + (int)sc.extent.width / g.scale, ay1 = ay0 + (int)sc.extent.height / g.scale;
	enableStencil();
	const int si = sceneForArea(ax0, ay0, ax1, ay1);
	if (si < 0)
		return -1;
	State::Scene& S = g.scenes[si];
	sceneSync(S);
	const float kx = 1024.f / (float)S.aw, ky = 512.f / (float)S.ah;
	push = WorldPush {};
	for (int i = 0; i < 3; i++)
	{
		push.rows[4 * i] = camera.r[3 * i], push.rows[4 * i + 1] = camera.r[3 * i + 1], push.rows[4 * i + 2] = camera.r[3 * i + 2];
		push.rows[4 * i + 3] = camera.t[i];
	}
	push.project[0] = camera.h * kx / 512.f, push.project[1] = camera.h * ky / 256.f;
	push.project[2] = (camera.ofx + camera.drawX - (float)S.ax) * kx / 512.f - 1.f;
	push.project[3] = (camera.ofy + camera.drawY - (float)S.ay) * ky / 256.f - 1.f;
	const int sx0 = std::clamp(sceneX(S, g.area[0]), 0, (int)g.sceneW), sy0 = std::clamp(sceneY(S, g.area[1]), 0, (int)g.sceneH);
	const int sx1 = std::clamp(sceneX(S, g.area[2] + 1), 0, (int)g.sceneW), sy1 = std::clamp(sceneY(S, g.area[3] + 1), 0, (int)g.sceneH);
	box = vk::Rect2D({ sx0, sy0 }, { (uint32_t)std::max(0, sx1 - sx0), (uint32_t)std::max(0, sy1 - sy0) });
	S.any3D = true;
	S.lastUsed = g.texFrame;
	sceneMarkDrawn(S, ax0, ay0, ax1, ay1);
	return si;
}

void bWorldSky(const DeadstationWorldCamera& camera)
{
	if (!g.ready || !g.skyBuffer || g.skyVertices == 0)
		return;
	decodeWorldPages();
	if (anyCpuNewer())
		flushUploads();
	WorldPush push;
	vk::Rect2D box;
	const int si = worldScene(camera, push, box);
	if (si < 0)
		return;
	worldLook(push);
	const int pushAt = (int)g.worldPushes.size();
	g.worldPushes.push_back(push);
	State::SceneDraw d { si, 0, g.skyVertices, WorldSky | SceneBit | 1u << 20, box, Push {}, false, 0 };
	d.world = pushAt, d.source = 1;
	g.sceneDraws.push_back(d);
	g.drawnSinceShown++;
}

// the frame's moving things, as quads in the world
void bWorldQuads(const DeadstationWorldCamera& camera, const uint8_t *bytes, int count)
{
	if (!g.ready || count <= 0)
		return;
	WorldPush push;
	vk::Rect2D box;
	// (each one's texture ready first: a decode ends the pass but not the frame)
	uint32_t memoKey = ~0u;
	int memoSlot = -1;
	DeadstationWorldQuad q;
	for (int i = 0; i < count; i++)
	{
		std::memcpy(&q, bytes + (size_t)i * sizeof(q), sizeof(q));
		const uint32_t pageKey = (uint32_t)q.page | (uint32_t)q.clut << 16;
		if (pageKey != memoKey)
			memoKey = pageKey, memoSlot = worldSlot(q.page, q.clut);
	}
	if (anyCpuNewer())
		flushUploads();
	const int si = worldScene(camera, push, box);
	if (si < 0)
		return;
	// (what's solid takes a module's numbers, as the mesh's cut-outs. What's added or blended, not)
	{
		DeadstationWorldQuad first;
		std::memcpy(&first, bytes, sizeof(first));
		worldLook(push, first.semi == 0 ? 2 : 0);
	}
	int pushAt = (int)g.worldPushes.size();
	g.worldPushes.push_back(push);
	WorldPush pushAdded = push;
	pushAdded.c[2] = 1.f;		// (psx_gpu_world.vert: dimmed by the fog, not mixed with it)
	int pushAddedAt = (int)g.worldPushes.size();
	g.worldPushes.push_back(pushAdded);
	Slot *s = &slot();
	s->vertexUsed = (s->vertexUsed + sizeof(WorldVertex) - 1) & ~(vk::DeviceSize)(sizeof(WorldVertex) - 1);
	memoKey = ~0u;
	uint32_t runKey = ~0u, runFirst = 0, runCount = 0;
	uint8_t memoClass = 7;
	const auto endRun = [&]() {
		if (runCount == 0)
			return;
		State::SceneDraw d { si, runFirst, runCount, runKey, box, Push {}, false, 0 };
		d.world = ((runKey >> 4) & 15) == 2 ? pushAddedAt : pushAt, d.source = 2;
		g.sceneDraws.push_back(d);
		g.vertices += runCount;
		g.sceneDrawCount += runCount / 3;
		g.drawnSinceShown++;
		runCount = 0;
	};
	for (int i = 0; i < count; i++)
	{
		std::memcpy(&q, bytes + (size_t)i * sizeof(q), sizeof(q));
		const uint32_t pageKey = (uint32_t)q.page | (uint32_t)q.clut << 16;
		if (pageKey != memoKey)
		{
			memoKey = pageKey;
			memoSlot = worldSlot(q.page, q.clut);
			memoClass = memoSlot >= 0 ? g.texSlots[memoSlot].texels : 7;
		}
		if (memoSlot < 0)
			continue;
		worldUses(memoSlot, std::min({ q.uv[0][0], q.uv[1][0], q.uv[2][0], q.uv[3][0] }), std::min({ q.uv[0][1], q.uv[1][1], q.uv[2][1], q.uv[3][1] }),
				std::max({ q.uv[0][0], q.uv[1][0], q.uv[2][0], q.uv[3][0] }), std::max({ q.uv[0][1], q.uv[1][1], q.uv[2][1], q.uv[3][1] }));
		if (s->vertexUsed + 6 * sizeof(WorldVertex) > VertexBytes)
		{
			endRun();
			submit();
			s = &slot();
			s->vertexUsed = (s->vertexUsed + sizeof(WorldVertex) - 1) & ~(vk::DeviceSize)(sizeof(WorldVertex) - 1);
			pushAt = (int)g.worldPushes.size();
			g.worldPushes.push_back(push);
			pushAddedAt = (int)g.worldPushes.size();
			g.worldPushes.push_back(pushAdded);
		}
		// blended only if the page has texels that are; what's cut out, cut out
		const bool blended = q.semi != 0 && (memoClass == 7 || (memoClass & 4));
		const uint32_t key = (blended ? World | (uint32_t)q.semi << 4 | 1u << 20 : ((memoClass & 1) != 0 || memoClass == 7 ? World : WorldOpaque) | 2u << 20) | SceneBit | stencilBits();
		if (key != runKey || runCount == 0)
		{
			endRun();
			runKey = key, runFirst = (uint32_t)(s->vertexUsed / sizeof(WorldVertex));
		}
		WorldVertex built[6];
		static constexpr int Order[6] = { 0, 1, 2, 1, 3, 2 };
		const float across[3] = { (float)q.p[3][0] - q.p[0][0], (float)q.p[3][1] - q.p[0][1], (float)q.p[3][2] - q.p[0][2] };
		const uint8_t size = (uint8_t)std::min(std::sqrt(across[0] * across[0] + across[1] * across[1] + across[2] * across[2]) / 16.f, 255.f);
		for (int j = 0; j < 6; j++)
		{
			const int n = Order[j];
			built[j] = { q.p[n][0], q.p[n][1], q.p[n][2], (int16_t)((int)q.layer * 32), q.uv[n][0], q.uv[n][1], (uint8_t)(q.raw ? 1 : 0), (uint8_t)memoSlot, q.rgb[n][0], q.rgb[n][1], q.rgb[n][2], size };
		}
		std::memcpy(s->vertexMap + s->vertexUsed, built, sizeof(built));
		s->vertexUsed += sizeof(built);
		runCount += 6;
	}
	endRun();
	// (what's written next, by its own stride, finds its index by dividing)
	s = &slot();
	s->vertexUsed = (s->vertexUsed + sizeof(Vertex) - 1) / sizeof(Vertex) * sizeof(Vertex);
}

// the kept frame drawn again half way to the next, into the replay scene. A corner's place in view space from its screen
// position and depth; moved by half its object's motion to the next frame; projected back.
bool drawReplay(int scene)
{
	using deadstation::replay::Frame;
	using deadstation::replay::Transform;
	if (g.perf)
	{
		static unsigned told = 0;
		if (told++ % 120 == 0)
			DS_LOG("recomp replay: scene %d, newest %d, recording %d, this frame %s with %zu draws, h %.0f; the newest %s", scene, g.replayNewest,
					g.replayRecording, g.replays[scene].frame.valid ? "valid" : "invalid", g.replays[scene].draws.size(), g.replays[scene].frame.h,
					g.replayNewest >= 0 && g.replays[g.replayNewest].frame.valid ? "valid" : "invalid");
	}
	// the real step, if the game gave it; else half way to the next frame, if there is one yet
	const bool given = g.replayStepsFor == scene;
	if (!given && (g.replayNewest < 0 || g.replayNewest == scene))
		return false;
	State::Scene& shown = g.scenes[scene];
	State::Replay& R = g.replays[scene];
	const Frame& now = R.frame, &next = given ? R.frame : g.replays[g.replayNewest].frame;
	if (!now.valid || !next.valid || R.draws.empty() || now.h <= 0)
		return false;
	using Step = deadstation::replay::Step;
	Step step[deadstation::replay::MaxObjects];
	for (unsigned o = 0; o < deadstation::replay::MaxObjects; o++)
	{
		if (given)
		{
			step[o] = g.replaySteps.objects[o];
			continue;
		}
		Step& st = step[o];
		const Transform& a = next.objects[o], &b = now.objects[o];
		for (int i = 0; i < 9; i++)
			st.r[i] = i % 4 == 0 ? 1.f : 0.f;
		st.t[0] = st.t[1] = st.t[2] = 0;
		if (!a.present || !b.present)
			continue;
		// the rotation to the next frame: a.r x inverse(b.r)
		const float *m = b.r;
		const float c00 = m[4] * m[8] - m[5] * m[7], c01 = m[5] * m[6] - m[3] * m[8], c02 = m[3] * m[7] - m[4] * m[6];
		const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
		if (std::fabs(det) < 1e-6f)
			continue;
		const float k = 1.f / det;
		const float inv[9] = { c00 * k, (m[2] * m[7] - m[1] * m[8]) * k, (m[1] * m[5] - m[2] * m[4]) * k, c01 * k, (m[0] * m[8] - m[2] * m[6]) * k,
			(m[2] * m[3] - m[0] * m[5]) * k, c02 * k, (m[1] * m[6] - m[0] * m[7]) * k, (m[0] * m[4] - m[1] * m[3]) * k };
		float rel[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				rel[3 * i + j] = a.r[3 * i] * inv[j] + a.r[3 * i + 1] * inv[3 + j] + a.r[3 * i + 2] * inv[6 + j];
		float tr[3];
		for (int i = 0; i < 3; i++)
			tr[i] = a.t[i] - (rel[3 * i] * b.t[0] + rel[3 * i + 1] * b.t[1] + rel[3 * i + 2] * b.t[2]);
		// a camera cut or an object put back in its place: no step
		if (tr[0] * tr[0] + tr[1] * tr[1] + tr[2] * tr[2] > 4000.f * 4000.f || rel[0] + rel[4] + rel[8] < 2.0f)
			continue;
		for (int i = 0; i < 9; i++)
			st.r[i] = (rel[i] + (i % 4 == 0 ? 1.f : 0.f)) * 0.5f;
		for (int i = 0; i < 3; i++)
			st.t[i] = tr[i] * 0.5f;
	}
	const size_t bytes = R.vertices.size() * sizeof(Vertex);
	if (bytes > VertexBytes)
		return false;
	if (!g.replayScene.color.image)
	{
		using U = vk::ImageUsageFlagBits;
		makeImage(g.replayScene.color, SceneFormat, g.sceneW, g.sceneH, U::eColorAttachment | U::eSampled | U::eTransferSrc | U::eTransferDst,
				vk::ImageAspectFlagBits::eColor);
		const vk::ImageAspectFlags depthAspect = g.sceneStencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil :
				vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth);
		makeImage(g.replayScene.depth, g.sceneDepthFormat, g.sceneW, g.sceneH, U::eDepthStencilAttachment, depthAspect);
		const vk::ImageView views[2] = { *g.replayScene.color.view, *g.replayScene.depth.view };
		g.replayScene.framebuffer = g.dev.device.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *g.scenePasses[0],
				views, g.sceneW, g.sceneH, 1));
		endPass();
		Slot& s0 = slot();
		const auto to = [&](vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout layout) {
			vk::ImageMemoryBarrier b(vk::AccessFlags(), vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
					vk::ImageLayout::eUndefined, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
					vk::ImageSubresourceRange(aspect, 0, 1, 0, 1));
			s0.cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags(),
					nullptr, nullptr, b);
		};
		to(*g.replayScene.color.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral);
		to(*g.replayScene.depth.image, depthAspect, vk::ImageLayout::eDepthStencilAttachmentOptimal);
	}
	endPass();
	Slot *s = &slot();
	if (s->vertexUsed + bytes > VertexBytes)
	{
		submit();
		s = &slot();
	}
	Vertex *at = (Vertex *)(s->vertexMap + s->vertexUsed);
	const uint32_t base = (uint32_t)(s->vertexUsed / sizeof(Vertex));
	const float kx = 1024.f / (float)shown.aw, ky = 512.f / (float)shown.ah, ax = (float)shown.ax, ay = (float)shown.ay;
	const float ofx = now.ofx, ofy = now.ofy, h = now.h, ox = now.drawX, oy = now.drawY;
	const size_t n = R.vertices.size();
	for (size_t i = 0; i < n; i++)
	{
		Vertex v = R.vertices[i];
		const float z = R.z[i];
		if (z > 0)
		{
			const uint8_t tag = R.object[i];
			const Step& st = step[tag & 7];
			const float sx = v.x / kx + ax - ox, sy = v.y / ky + ay - oy;
			const float X = (sx - ofx) * z / h, Y = (sy - ofy) * z / h;
			const float x1 = st.r[0] * X + st.r[1] * Y + st.r[2] * z + st.t[0];
			const float y1 = st.r[3] * X + st.r[4] * Y + st.r[5] * z + st.t[1];
			const float z1 = std::max(st.r[6] * X + st.r[7] * Y + st.r[8] * z + st.t[2], 1.f);
			v.x = (ofx + x1 * h / z1 + ox - ax) * kx;
			v.y = (ofy + y1 * h / z1 + oy - ay) * ky;
			const float zd = z1 + 32.f * (float)((tag >> 3) & 15) - ((tag & 0x80) != 0 ? 32.f : 0.f);
			if (g.scenePerspective && g.frameIs3D)
				v.depth = scenePerspectiveDepth(zd), v.w = z1;
			else
				v.depth = sceneFlatDepth(zd);
		}
		at[i] = v;
	}
	s->vertexUsed += bytes;
	vk::CommandBuffer cmd = *s->cmd;
	const uint32_t stamp = stampOpen(*s, StampScene);
	const vk::Rect2D all({ 0, 0 }, { g.sceneW, g.sceneH });
	const vk::ClearValue clears[2] = { vk::ClearColorValue(), vk::ClearDepthStencilValue(0.f, 0) };
	cmd.beginRenderPass(vk::RenderPassBeginInfo(*g.scenePasses[0], *g.replayScene.framebuffer, all, 2, clears), vk::SubpassContents::eInline);
	cmd.setViewport(0, vk::Viewport(0, 0, (float)g.sceneW, (float)g.sceneH, 0, 1));
	vk::DeviceSize offset = 0;
	cmd.bindVertexBuffers(0, *s->vertices, offset);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *g.layout, 0, g.set, nullptr);
	const vk::ClearAttachment black(vk::ImageAspectFlagBits::eColor, 0, vk::ClearColorValue(std::array<float, 4> { 0.f, 0.f, 0.f, 0.f }));
	cmd.clearAttachments(black, vk::ClearRect(all, 0, 1));
	issueSceneDraws(cmd, R.draws.data(), R.draws.size(), base);
	cmd.endRenderPass();
	barrier(cmd);
	stampClose(*s, stamp);
	g.replayFrames++;
	return true;
}

int presentVram(int dx, int dy, int w, int h)
{
	if (!g.ready)
		return 0;
	if (anyCpuNewer())
		flushUploads();
	dx = std::clamp(dx, 0, VramW - 1), dy = std::clamp(dy, 0, VramH - 1);
	w = std::clamp(w, 1, VramW - dx), h = std::clamp(h, 1, VramH - dy);
	const uint32_t S = (uint32_t)g.scale;
	g.displayW = w, g.displayH = h;
	{
		State::ShownRect *slot = nullptr;
		for (State::ShownRect& r : g.shownRects)
			if (r.used && r.x == dx && r.y == dy && r.w == w && r.h == h)
				slot = &r;
		if (slot == nullptr)
		{
			slot = &g.shownRects[0];
			for (State::ShownRect& r : g.shownRects)
				if (!r.used || r.frame < slot->frame)
					slot = &r;
		}
		*slot = { dx, dy, w, h, g.texFrame, true };
	}
	// the scene holding the frame shown (one only overlapping it given to VRAM, which shows it), its draws drawn
	int shown = -1;
	for (int i = 0; i < State::SceneCount; i++)
	{
		State::Scene& scn = g.scenes[i];
		if (!scn.used)
			continue;
		if (dx >= scn.ax && dy >= scn.ay && dx + w <= scn.ax + scn.aw && dy + h <= scn.ay + scn.ah)
			shown = i;
		else if (overlaps(dx, dy, w, h, scn.ax, scn.ay, scn.aw, scn.ah))
			sceneResolve(scn, "a frame shown over part of it");
	}
	// shown from the scene: with drawing of its own, VRAM's newer writes taken first; without (VRAM holds all it does), from
	// VRAM while VRAM was written since
	if (shown >= 0)
	{
		State::Scene& scn = g.scenes[shown];
		if (scn.newer)
			sceneSync(scn);
		else if (scn.vramAll || !scn.vramRects.empty())
			shown = -1;
	}
	// A whole frame of one scene: shown as soon as it's drawn, and drawn straight into the page. As the PlayStation has it, a frame goes into the buffer off the screen and comes on at
	// the next flip: here that's a frame later than need be, what's shown in one submission with all of the next frame's drawing, and a copy across that took the last 2 ms of this
	// GPU's budget on every frame.
	g.framePage = -1, g.frameOnPage = false;
	const bool noDraws = g.sceneDraws.empty();
	int whole = -1;
	if (g.sceneWhole && !noDraws)
	{
		whole = g.sceneDraws.front().scene;
		for (const State::SceneDraw& d : g.sceneDraws)
			if (d.scene != whole)
				whole = -1;
		if (whole >= 0 && (w != g.scenes[whole].aw || h != g.scenes[whole].ah || g.scenes[whole].depthStarted))
			whole = -1;
		if (whole >= 0)
			g.framePage = hostVideo().pageForFrame(g.sceneW, g.sceneH);
	}
	flushScene(g.framePage >= 0 ? whole : shown >= 0 ? shown : -2);
	g.framePage = -1;
	// (tests: the first presents after a frame's left this path or come back to it)
	if (g.perf)
	{
		static bool wasOnPage = false;
		static int telling = 0;
		if (g.frameOnPage != wasOnPage)
			telling = 8;
		wasOnPage = g.frameOnPage;
		if (telling > 0)
		{
			timespec up {};
			clock_gettime(CLOCK_BOOTTIME, &up);
			DS_LOG("recomp present: %.3f shown %d whole %d onPage %d noDraws %d display %d,%d last %d holding %d,%d", (double)up.tv_sec + up.tv_nsec * 1e-9, shown, whole, g.frameOnPage ? 1 : 0, noDraws ? 1 : 0, dx, dy, g.lastOnPage, g.scenes[0].pageHolding, g.scenes[1].pageHolding);
			telling--;
		}
	}
	if (g.frameOnPage)
		shown = whole, g.lastOnPage = whole;
	else
	{
		// (a blank with no frame for it: that one again, not the buffer the game has on the screen, which is older. By that scene's own place in VRAM: the display's is the other's,
		// off the edge of this one, and a column of it stretched over the screen for a frame)
		if (noDraws && g.lastOnPage >= 0 && g.scenes[g.lastOnPage].used)
		{
			shown = g.lastOnPage;
			dx = g.scenes[shown].ax, dy = g.scenes[shown].ay;
		}
		g.lastOnPage = -1;
		restoreScenes();
	}
	if (shown < 0)
		refreshScaled(dx, dy, w, h);
	// a blank showing the same frame again: its in-between frame, when its module gives the motion
	const bool repeat = shown >= 0 && shown == g.lastShownScene && dx == g.lastShownX && dy == g.lastShownY;
	// (the last frame came late: the GPU is behind, and an in-between frame would make the game wait for it: a shown frame
	// is given up rather than the game's speed)
	const auto presentAt = std::chrono::steady_clock::now();
	const double sinceLast = std::chrono::duration<double>(presentAt - g.lastPresentAt).count();
	const bool late = sinceLast > 0.0195;
	if (g.perf && sinceLast > 0.06 && sinceLast < 2.0)
	{
		timespec up {};
		clock_gettime(CLOCK_BOOTTIME, &up);
		DS_LOG("recomp stall: %.0f ms between frames, at uptime %.3f; the last frame's commands %.1f ms, its page %.1f ms", sinceLast * 1000.0, (double)up.tv_sec + up.tv_nsec * 1e-9,
				g.lastRecordMs, g.lastPageMs);
	}
	g.lastPresentAt = presentAt;
	const bool replayed = repeat && !late && !g.frameOnPage && drawReplay(shown);
	const auto recordedAt = std::chrono::steady_clock::now();
	g.lastShownScene = shown, g.lastShownX = dx, g.lastShownY = dy;
	// the texture cache's queued pages, for the next frame
	decodePending();
	g.quickDecodes = 0;
	g.texFrame++;
	// the GPU's commands since the last frame go in the frame's submission (one submission a frame)
	endPass();
	Slot& s = g.slots[g.current];
	vk::CommandBuffer before = nullptr;
	if (s.recording)
	{
		barrier(*s.cmd);
		if (s.timing)
		{
			s.cmd->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, stampPool(s), g.current * StampsPerSlot + 1);
			s.timed = true;
		}
		s.cmd->end();
		before = *s.cmd;
	}
	uint32_t shownW = w * S, shownH = h * S;
	vk::Fence fence;
	if (shown >= 0)
	{
		State::Scene& scn = g.scenes[shown];
		const int x0 = std::clamp(sceneX(scn, dx), 0, (int)g.sceneW - 1), y0 = std::clamp(sceneY(scn, dy), 0, (int)g.sceneH - 1);
		const int x1 = std::clamp(sceneX(scn, dx + w), x0 + 1, (int)g.sceneW), y1 = std::clamp(sceneY(scn, dy + h), y0 + 1, (int)g.sceneH);
		shownW = (uint32_t)(x1 - x0), shownH = (uint32_t)(y1 - y0);
		if (g.frameOnPage)
			fence = hostVideo().framePage(before);
		else
			fence = hostVideo().frameImage(replayed ? *g.replayScene.color.view : *scn.color.view, g.sceneW, g.sceneH, (uint32_t)x0, (uint32_t)y0, shownW, shownH, before);
		scn.depthStarted = false, scn.any3D = false, scn.lastUsed = g.texFrame;
	}
	else
		fence = hostVideo().frameImage(*g.scaled.view, VramW * S, VramH * S, dx * S, dy * S, w * S, h * S, before);
	if (g.perf)
	{
		const auto pageDone = std::chrono::steady_clock::now();
		g.lastRecordMs = std::chrono::duration<double>(recordedAt - presentAt).count() * 1000.0, g.lastPageMs = std::chrono::duration<double>(pageDone - recordedAt).count() * 1000.0;
	}
	if (before)
	{
		if (fence)
			s.borrowed = (int)hostVideo().lastPage();
		else
		{
			g.dev.device.resetFences(*s.fence);
			const vk::CommandBuffer cmd = *s.cmd;
			hostVideo().submit(&cmd, 1, *s.fence);
		}
		s.recording = false;
		g.current = (g.current + 1) % Slots;
	}
	if (g.perf)
	{
		g.frames++;
		const auto now = std::chrono::steady_clock::now();
		if (g.perfStart.time_since_epoch().count() == 0)
			g.perfStart = now;
		if (std::chrono::duration<double>(now - g.perfStart).count() >= 1.0 && g.frames > 0)
		{
			const unsigned f = g.frames;
			double layer = 0, page = 0;
			hostVideo().takeGpu(layer, page);
			DS_LOG("recomp frames: %u of %u straight into their page", g.framesOnPages, f);
			g.framesOnPages = 0;
			DS_LOG("recomp gpu: a frame: %.0f draws, %.1f passes, %.1f packs, %.1f uploads (%.1f deferred), %.1f copies, %.1f fills, %.0f vertices; "
					"%u readbacks, %.1f pages decoded; GPU time a frame: VRAM %.1f ms (drawing passes %.1f, packs %.1f, copies %.1f, "
					"upload copies %.1f, decodes %.1f), "
					"3D layer's passes %.1f ms, page %.1f ms", (double)g.draws / f, (double)g.passes / f, (double)g.packs / f, (double)g.uploads / f,
					(double)g.deferred / f, (double)g.copies / f, (double)g.fills / f, (double)g.vertices / f, g.readbacks,
					(double)g.decodes / f, g.gpuSeconds * 1000 / f, g.gpuKind[StampPass] * 1000 / f, g.gpuKind[StampPack] * 1000 / f,
					g.gpuKind[StampCopy] * 1000 / f, g.gpuKind[StampUpload] * 1000 / f, g.gpuKind[StampDecode] * 1000 / f, layer * 1000 / f,
					page * 1000 / f);
			g.decodes = 0;
			g.draws = 0;
			DS_LOG("recomp gpu: a frame: %.2f M pixels drawn (%.2f M blended, %.2f M textured from VRAM, %.2f M untextured)",
					g.pixels / f / 1e6, g.pixelsBlended / f / 1e6, g.pixelsVram / f / 1e6, g.pixelsFlat / f / 1e6);
			g.pixels = g.pixelsBlended = g.pixelsVram = g.pixelsFlat = 0;
			DS_LOG("recomp gpu: a frame: the native scene %.1f passes, %.0f primitives, %.1f resolves, %.1f blits from VRAM; GPU %.1f ms",
					(double)g.scenePassCount / f, (double)g.sceneDrawCount / f, (double)g.sceneResolves / f, (double)g.sceneBlits / f,
					g.gpuKind[StampScene] * 1000 / f);
			g.scenePassCount = g.sceneResolves = g.sceneBlits = g.sceneDrawCount = 0;
			{
				char scenes[160] = "";
				for (const State::Scene& sc : g.scenes)
					if (sc.used)
						std::snprintf(scenes + strlen(scenes), sizeof(scenes) - strlen(scenes), " %d,%d %dx%d%s", sc.ax, sc.ay, sc.aw, sc.ah,
								sc.newer ? "*" : "");
				DS_LOG("recomp gpu: a frame: %.0f primitives in VRAM (the last: draw area %d,%d..%d,%d, at %d,%d..%d,%d); %.0f texels "
						"uploaded (the last %d,%d %d x %d); scenes:%s", (double)g.vramPrims / f, g.lastVramArea[0], g.lastVramArea[1],
						g.lastVramArea[2], g.lastVramArea[3], g.lastVramBounds[0], g.lastVramBounds[1], g.lastVramBounds[2], g.lastVramBounds[3],
						g.uploadPixels / f, g.lastUpload[0], g.lastUpload[1], g.lastUpload[2], g.lastUpload[3], scenes);
				g.vramPrims = 0;
				g.uploadPixels = 0;
			}
			DS_LOG("recomp gpu: a frame: the render thread busy %.1f ms (waiting there for slots' earlier work %.1f ms), the module's "
					"thread blocked on it %.1f ms", g.rtBusy * 1000 / f, g.rtFenceWait * 1000 / f, takeBlockedSeconds() * 1000 / f);
			g.rtBusy = g.rtFenceWait = 0;
			g.gpuSeconds = 0;
			std::fill(std::begin(g.gpuKind), std::end(g.gpuKind), 0);
			g.perfStart = now;
			g.frames = g.passes = g.packs = g.uploads = g.copies = g.fills = g.readbacks = g.deferred = 0;
			g.vertices = 0;
		}
	}
	hostFrameShown((int)shownW, (int)shownH, g.drawnSinceShown != 0 || replayed);
	g.drawnSinceShown = 0;
	return 1;
}

void presentCpu(const uint32_t *pixels, int w, int h)
{
	if (!g.ready || pixels == nullptr || w <= 0 || h <= 0)
		return;
	submit();
	hostVideo().frame(pixels, (unsigned)w, (unsigned)h, (size_t)w * 4, Video::Format::XRGB8888);
	hostFrameShown(w, h, true);
}

void presentBlank()
{
	if (!g.ready)
		return;
	submit();
	hostVideo().frame(black.data(), 64, 64, 64 * 4, Video::Format::XRGB8888);
	hostFrameShown(64, 64, false);
}

void syncCpu()
{
	readback(0, 0, VramW, VramH);
}

void restage()
{
	std::fill(std::begin(g.gpuNewer), std::end(g.gpuNewer), ~0ull);	// all of it given to the GPU at once below
	if (g.ready && g.cpu != nullptr)
	{
		std::vector<uint16_t> all(g.cpu, g.cpu + VramW * VramH);	// PSXRecomp's (restored), the renderer's from it
		upload(0, 0, VramW, VramH, all.data());
	}
}


// ---- the render thread ----
// Every Vulkan call of the module's on a thread of its own (DEADSTATION_GPU_THREAD=0: all on the module's thread): the
// backend's calls are recorded on the module's thread as commands (their arguments, a transfer's pixels) in a batch,
// handed over with each frame (and at every MiB), and run in order on the render thread; a call that answers (a VRAM
// read, the display in software) runs there while the module's thread waits. The module's thread is at most a frame
// ahead. The render thread draws from its own copy of VRAM (g.mirror); PSXRecomp's (g.cpu) is written there only during
// such a call.
namespace rt
{
enum Op : uint8_t
{
	OpFilter, OpSemi, OpMask, OpWindow, OpFrameIs3D, OpReplayFrame, OpReplaySteps, OpReplayObject, OpTriangles, OpWorldDraw, OpWorldSky, OpWorldQuads, OpFog, OpSceneWhole, OpWorldLights, OpGlows, OpTint, OpModulation, OpFill, OpCopy, OpFlatTri, OpGouraudTri, OpTexTri, OpShadedTexTri,
	OpFlatRect, OpTexRect, OpTexRectScaled, OpLine, OpShadedLine, OpVramWrite, OpTransferIn, OpArea, OpPresentVram,
	OpPresentCpu, OpPresentBlank, OpCall, OpInit, OpTriDepth, OpPrecise
};
struct F4 { int on; float z0, z1, z2; };
struct FogOp { float nearZ, farZ, most, r, g, b; };
inline void bSetFog(const FogOp& a)
{
	g.fogNear = a.nearZ, g.fogScale = a.farZ > a.nearZ ? 1.f / (a.farZ - a.nearZ) : 0.f, g.fogMost = a.most;
	g.fogRgb[0] = a.r, g.fogRgb[1] = a.g, g.fogRgb[2] = a.b;
}
struct P7 { int on; int32_t c[6]; };
struct I2 { int a, b; };
struct I4 { int a, b, c, d; };
struct Fill { int x, y, w, h; uint16_t c; };
struct Copy { int sx, sy, dx, dy, w, h; };
struct Flat { int x0, y0, x1, y1, x2, y2; uint16_t c; };
struct Gouraud { int x0, y0, x1, y1, x2, y2; uint16_t c0, c1, c2; };
struct Tex { int x0, y0, u0, v0, x1, y1, u1, v1, x2, y2, u2, v2; uint16_t cx, cy, page; };
struct ShadedTex { int x0, y0, u0, v0, x1, y1, u1, v1, x2, y2, u2, v2; uint32_t c0, c1, c2; uint16_t cx, cy, page; int raw; };
struct Rect { int x, y, w, h; uint16_t c; };
struct TexRect { int x, y, w, h, u0, v0, u1, v1; uint16_t cx, cy, page; };
struct Line { int x0, y0, x1, y1; uint16_t c0, c1; };
struct Pixel { int x, y; uint16_t p; };
struct Call
{
	std::function<void()> f;
	bool done = false;
};

bool on = false;
std::thread thread;
std::mutex m;
std::condition_variable cv;
std::vector<uint8_t> filling;
std::deque<std::vector<uint8_t>> handed;
std::vector<std::vector<uint8_t>> spare;
bool stop = false;
unsigned presentsQueued = 0, presentsDone = 0;
// DEADSTATION_PERF: the module's thread's nanoseconds blocked on the render thread (a frame ahead, or a call)
std::atomic<uint64_t> blockedNs { 0 };
std::vector<uint16_t> mirror;
// what the module's thread asks back of the state it set
int filter = 0;
// PSXRecomp's VRAM current (read back, nothing drawn or uploaded since): its reads need no call
bool fresh = false;
int area[4] = { 0, 0, VramW - 1, VramH - 1 };

template<typename T> void put(Op op, const T& args)
{
	const size_t at = filling.size();
	filling.resize(at + 1 + sizeof(T));
	filling[at] = op;
	std::memcpy(filling.data() + at + 1, &args, sizeof(T));
}
void putOp(Op op)
{
	filling.push_back(op);
}
void putBytes(const void *data, size_t n)
{
	const size_t at = filling.size();
	filling.resize(at + n);
	std::memcpy(filling.data() + at, data, n);
}
void handOver()
{
	if (filling.empty())
		return;
	{
		std::lock_guard<std::mutex> hold(m);
		handed.push_back(std::move(filling));
		if (!spare.empty())
		{
			filling = std::move(spare.back());
			spare.pop_back();
		}
		else
			filling = std::vector<uint8_t>();
	}
	filling.clear();
	cv.notify_all();
}
inline void bounded()
{
	fresh = false;
	if (filling.size() >= (1u << 20))
		handOver();
}

// f on the render thread after everything recorded before it
void call(std::function<void()> f)
{
	Call c { std::move(f) };
	Call *pc = &c;
	put(OpCall, pc);
	handOver();
	const auto from = std::chrono::steady_clock::now();
	std::unique_lock<std::mutex> hold(m);
	cv.wait(hold, [&] { return c.done; });
	blockedNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count();
}
// PSXRecomp's VRAM brought up to date (one call), for reads from it
void freshen()
{
	if (fresh)
		return;
	call([] { syncCpu(); });
	fresh = true;
}
// a frame recorded: the module's thread waits while the render thread is a frame behind already
void presented()
{
	{
		std::lock_guard<std::mutex> hold(m);
		presentsQueued++;
	}
	handOver();
	const auto from = std::chrono::steady_clock::now();
	std::unique_lock<std::mutex> hold(m);
	cv.wait(hold, [] { return presentsQueued - presentsDone <= 1 || stop; });
	blockedNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count();
}

template<typename T> T get(const uint8_t *&p)
{
	T v;
	std::memcpy(&v, p, sizeof(T));
	p += sizeof(T);
	return v;
}
void run(const std::vector<uint8_t>& batch)
{
	const uint8_t *p = batch.data(), *end = p + batch.size();
	while (p < end)
	{
		const Op op = (Op)*p++;
		switch (op)
		{
		case OpFilter: bSetFilter(get<int>(p)); break;
		case OpSemi: { const I2 a = get<I2>(p); bSetSemi(a.a, a.b); break; }
		case OpMask: { const I2 a = get<I2>(p); bSetMask(a.a, a.b); break; }
		case OpWindow: bSetWindow(get<uint32_t>(p)); break;
		case OpFrameIs3D: { const I4 a = get<I4>(p); g.frameIs3D = a.a != 0; break; }
		case OpReplayFrame:
		{
			const deadstation::replay::Frame a = get<deadstation::replay::Frame>(p);
			g.replayLatest = a;
			g.replayPending = a.valid;
			if (!a.valid)
				g.replayRecording = g.replayNewest = g.replayStepsFor = -1;
			break;
		}
		case OpReplaySteps:
			g.replaySteps = get<deadstation::replay::Steps>(p);
			g.replayStepsFor = g.replayNewest;
			break;
		case OpReplayObject: { const I4 a = get<I4>(p); g.replayObject = a.a; break; }
		case OpTriangles:
		{
			const I4 a = get<I4>(p);
			bulkTriangles(p, a.a);
			p += (size_t)a.a * sizeof(DeadstationTriangle);
			break;
		}
		case OpGlows:
		{
			const I4 a = get<I4>(p);
			bulkGlows(p, a.a, a.b);
			p += (size_t)a.a * sizeof(DeadstationGlow);
			break;
		}
		case OpWorldDraw:
		{
			const I4 a = get<I4>(p);
			const DeadstationWorldCamera camera = get<DeadstationWorldCamera>(p);
			bWorldDraw(camera, (const uint16_t *)p, a.a, a.b);
			p += (size_t)a.a * sizeof(uint16_t) + ((size_t)a.a & 1) * sizeof(uint16_t);
			break;
		}
		case OpSceneWhole:
		{
			const F4 a = get<F4>(p);
			g.sceneWhole = true, g.sceneClear[0] = a.z0, g.sceneClear[1] = a.z1, g.sceneClear[2] = a.z2;
			break;
		}
		case OpWorldLights:
		{
			const WorldNumbersOp n = get<WorldNumbersOp>(p);
			std::memcpy(g.worldNumbers, n.numbers, sizeof(g.worldNumbers));
			g.worldOwnSolid = n.ownSolid != 0;
			break;
		}
		case OpWorldSky:
		{
			const DeadstationWorldCamera camera = get<DeadstationWorldCamera>(p);
			bWorldSky(camera);
			break;
		}
		case OpWorldQuads:
		{
			const I4 a = get<I4>(p);
			const DeadstationWorldCamera camera = get<DeadstationWorldCamera>(p);
			bWorldQuads(camera, p, a.a);
			p += (size_t)a.a * sizeof(DeadstationWorldQuad);
			break;
		}
		case OpFog:
		{
			const FogOp a = get<FogOp>(p);
			bSetFog(a);
			break;
		}
		case OpTint: { const I4 a = get<I4>(p); bSetTint(a.a, a.b, a.c); break; }
		case OpModulation: { const I4 a = get<I4>(p); bSetModulation(a.a, a.b, a.c, a.d); break; }
		case OpFill: { const Fill a = get<Fill>(p); bFill(a.x, a.y, a.w, a.h, a.c); break; }
		case OpCopy: { const Copy a = get<Copy>(p); bCopy(a.sx, a.sy, a.dx, a.dy, a.w, a.h); break; }
		case OpFlatTri: { const Flat a = get<Flat>(p); bFlatTriangle(a.x0, a.y0, a.x1, a.y1, a.x2, a.y2, a.c); break; }
		case OpGouraudTri:
		{
			const Gouraud a = get<Gouraud>(p);
			bGouraudTriangle(a.x0, a.y0, a.c0, a.x1, a.y1, a.c1, a.x2, a.y2, a.c2);
			break;
		}
		case OpTexTri:
		{
			const Tex a = get<Tex>(p);
			bTexturedTriangle(a.x0, a.y0, a.u0, a.v0, a.x1, a.y1, a.u1, a.v1, a.x2, a.y2, a.u2, a.v2, a.cx, a.cy, a.page);
			break;
		}
		case OpShadedTexTri:
		{
			const ShadedTex a = get<ShadedTex>(p);
			bShadedTexturedTriangle(a.x0, a.y0, a.u0, a.v0, a.c0, a.x1, a.y1, a.u1, a.v1, a.c1, a.x2, a.y2, a.u2, a.v2, a.c2, a.cx,
					a.cy, a.page, a.raw);
			break;
		}
		case OpFlatRect: { const Rect a = get<Rect>(p); bFlatRect(a.x, a.y, a.w, a.h, a.c); break; }
		case OpTexRect:
		case OpTexRectScaled:
		{
			const TexRect a = get<TexRect>(p);
			bTexturedRectScaled(a.x, a.y, a.w, a.h, a.u0, a.v0, a.u1, a.v1, a.cx, a.cy, a.page);
			break;
		}
		case OpLine: { const Line a = get<Line>(p); bLine(a.x0, a.y0, a.x1, a.y1, a.c0); break; }
		case OpShadedLine: { const Line a = get<Line>(p); bShadedLine(a.x0, a.y0, a.c0, a.x1, a.y1, a.c1); break; }
		case OpVramWrite: { const Pixel a = get<Pixel>(p); bVramWrite(a.x, a.y, a.p); break; }
		case OpTransferIn:
		{
			const I4 a = get<I4>(p);
			bTransferIn(a.a, a.b, a.c, a.d, (const uint16_t *)p);
			p += (size_t)a.c * a.d * 2;
			break;
		}
		case OpArea: { const I4 a = get<I4>(p); bSetArea(a.a, a.b, a.c, a.d); break; }
		case OpPresentVram:
		case OpPresentCpu:
		case OpPresentBlank:
		{
			if (op == OpPresentVram)
			{
				const I4 a = get<I4>(p);
				presentVram(a.a, a.b, a.c, a.d);
			}
			else if (op == OpPresentCpu)
			{
				const I2 a = get<I2>(p);
				presentCpu((const uint32_t *)p, a.a, a.b);
				p += (size_t)a.a * a.b * 4;
			}
			else
				presentBlank();
			{
				std::lock_guard<std::mutex> hold(m);
				presentsDone++;
			}
			cv.notify_all();
			break;
		}
		case OpCall:
		{
			Call *c = get<Call *>(p);
			c->f();
			{
				std::lock_guard<std::mutex> hold(m);
				c->done = true;
			}
			cv.notify_all();
			break;
		}
		case OpInit: bInit(get<uint16_t *>(p)); break;
		case OpTriDepth: { const F4 a = get<F4>(p); bSetTriangleDepth(a.on, a.z0, a.z1, a.z2); break; }
		case OpPrecise: { const P7 a = get<P7>(p); bSetPrecise(a.on, a.c[0], a.c[1], a.c[2], a.c[3], a.c[4], a.c[5]); break; }
		}
	}
}
void loop()
{
	for (;;)
	{
		std::vector<uint8_t> batch;
		{
			std::unique_lock<std::mutex> hold(m);
			cv.wait(hold, [] { return stop || !handed.empty(); });
			if (handed.empty())
				return;
			batch = std::move(handed.front());
			handed.pop_front();
		}
		const auto from = std::chrono::steady_clock::now();
		run(batch);
		if (g.perf)
			g.rtBusy += std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
		batch.clear();
		std::lock_guard<std::mutex> hold(m);
		spare.push_back(std::move(batch));
	}
}
void start()
{
	if (on || !g.ready)
		return;
	blockedNs = 0;
	const char *env = getenv("DEADSTATION_GPU_THREAD");
	if (env != nullptr && strcmp(env, "0") == 0)
		return;
	mirror.assign((size_t)VramW * VramH, 0);
	if (g.cpu != nullptr)
		memcpy(mirror.data(), g.cpu, mirror.size() * 2);
	g.mirror = mirror.data();
	stop = false;
	on = true;
	thread = std::thread(loop);
	DS_LOG("recomp gpu: drawing on a thread of its own");
}
}

double takeBlockedSeconds() { return (double)rt::blockedNs.exchange(0) * 1e-9; }

// the backend's table: recorded for the render thread, or run at once without it
void fInit(uint16_t *vram) { if (!rt::on) bInit(vram); else rt::call([vram] { bInit(vram); }); }
void fSetScale(int scale) { bSetScale(scale); }
int fScale() { return bScale(); }
void fSetFilter(int bilinear)
{
	if (!rt::on)
		return bSetFilter(bilinear);
	rt::filter = bilinear ? 1 : 0;
	rt::put(rt::OpFilter, bilinear);
}
int fFilter() { return rt::on ? rt::filter : bFilter(); }
void fSetSemi(int enabled, int mode) { if (!rt::on) bSetSemi(enabled, mode); else rt::put(rt::OpSemi, rt::I2 { enabled, mode }); }
void fSetMask(int set, int check) { if (!rt::on) bSetMask(set, check); else rt::put(rt::OpMask, rt::I2 { set, check }); }
void fSetWindow(uint32_t raw) { if (!rt::on) bSetWindow(raw); else rt::put(rt::OpWindow, raw); }
void fSetTriangleDepth(int enabled, float z0, float z1, float z2)
{
	if (psx_gpu_triangle_source_hook != nullptr)
		psx_gpu_triangle_source_hook(psx_gpu_source_addr());
	if (!rt::on)
		bSetTriangleDepth(enabled, z0, z1, z2);
	else
		rt::put(rt::OpTriDepth, rt::F4 { enabled, z0, z1, z2 });
}
void fSetPrecise(int enabled, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t x2, int32_t y2)
{
	if (!rt::on)
		bSetPrecise(enabled, x0, y0, x1, y1, x2, y2);
	else
		rt::put(rt::OpPrecise, rt::P7 { enabled, { x0, y0, x1, y1, x2, y2 } });
}
void fFrameIs3D(int on)
{
	if (!rt::on)
		g.frameIs3D = on != 0;
	else
		rt::put(rt::OpFrameIs3D, rt::I4 { on, 0, 0, 0 });
}
void fSetModulation(int r, int gr, int b, int raw)
{
	if (!rt::on)
		bSetModulation(r, gr, b, raw);
	else
		rt::put(rt::OpModulation, rt::I4 { r, gr, b, raw });
}
void fFill(int x, int y, int w, int h, uint16_t color)
{
	if (!rt::on)
		return bFill(x, y, w, h, color);
	rt::put(rt::OpFill, rt::Fill { x, y, w, h, color });
	rt::bounded();
}
void fCopy(int sx, int sy, int dx, int dy, int w, int h)
{
	if (!rt::on)
		return bCopy(sx, sy, dx, dy, w, h);
	rt::put(rt::OpCopy, rt::Copy { sx, sy, dx, dy, w, h });
	rt::bounded();
}
void fFlatTriangle(int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c)
{
	if (!rt::on)
		return bFlatTriangle(x0, y0, x1, y1, x2, y2, c);
	rt::put(rt::OpFlatTri, rt::Flat { x0, y0, x1, y1, x2, y2, c });
	rt::bounded();
}
void fGouraudTriangle(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1, int x2, int y2, uint16_t c2)
{
	if (!rt::on)
		return bGouraudTriangle(x0, y0, c0, x1, y1, c1, x2, y2, c2);
	rt::put(rt::OpGouraudTri, rt::Gouraud { x0, y0, x1, y1, x2, y2, c0, c1, c2 });
	rt::bounded();
}
void fTexturedTriangle(int x0, int y0, int u0, int v0, int x1, int y1, int u1, int v1, int x2, int y2, int u2, int v2, uint16_t cx,
		uint16_t cy, uint16_t page)
{
	if (!rt::on)
		return bTexturedTriangle(x0, y0, u0, v0, x1, y1, u1, v1, x2, y2, u2, v2, cx, cy, page);
	rt::put(rt::OpTexTri, rt::Tex { x0, y0, u0, v0, x1, y1, u1, v1, x2, y2, u2, v2, cx, cy, page });
	rt::bounded();
}
void fShadedTexturedTriangle(int x0, int y0, int u0, int v0, uint32_t c0, int x1, int y1, int u1, int v1, uint32_t c1, int x2, int y2,
		int u2, int v2, uint32_t c2, uint16_t cx, uint16_t cy, uint16_t page, int raw)
{
	if (!rt::on)
		return bShadedTexturedTriangle(x0, y0, u0, v0, c0, x1, y1, u1, v1, c1, x2, y2, u2, v2, c2, cx, cy, page, raw);
	rt::put(rt::OpShadedTexTri, rt::ShadedTex { x0, y0, u0, v0, x1, y1, u1, v1, x2, y2, u2, v2, c0, c1, c2, cx, cy, page, raw });
	rt::bounded();
}
void fFlatRect(int x, int y, int w, int h, uint16_t c)
{
	if (psx_gpu_triangle_source_hook != nullptr)
		psx_gpu_triangle_source_hook(psx_gpu_source_addr());
	if (!rt::on)
		return bFlatRect(x, y, w, h, c);
	rt::put(rt::OpFlatRect, rt::Rect { x, y, w, h, c });
	rt::bounded();
}
void fTexturedRectScaled(int x, int y, int w, int h, int u0, int v0, int u1, int v1, uint16_t cx, uint16_t cy, uint16_t page)
{
	if (psx_gpu_triangle_source_hook != nullptr)
		psx_gpu_triangle_source_hook(psx_gpu_source_addr());
	if (!rt::on)
		return bTexturedRectScaled(x, y, w, h, u0, v0, u1, v1, cx, cy, page);
	rt::put(rt::OpTexRectScaled, rt::TexRect { x, y, w, h, u0, v0, u1, v1, cx, cy, page });
	rt::bounded();
}
void fTexturedRect(int x, int y, int w, int h, int u, int v, uint16_t cx, uint16_t cy, uint16_t page)
{
	fTexturedRectScaled(x, y, w, h, u, v, u + w, v + h, cx, cy, page);
}
void fLine(int x0, int y0, int x1, int y1, uint16_t c)
{
	if (psx_gpu_triangle_source_hook != nullptr)
		psx_gpu_triangle_source_hook(psx_gpu_source_addr());
	if (!rt::on)
		return bLine(x0, y0, x1, y1, c);
	rt::put(rt::OpLine, rt::Line { x0, y0, x1, y1, c, c });
	rt::bounded();
}
void fShadedLine(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1)
{
	if (psx_gpu_triangle_source_hook != nullptr)
		psx_gpu_triangle_source_hook(psx_gpu_source_addr());
	if (!rt::on)
		return bShadedLine(x0, y0, c0, x1, y1, c1);
	rt::put(rt::OpShadedLine, rt::Line { x0, y0, x1, y1, c0, c1 });
	rt::bounded();
}
int fRenderDisplay(uint32_t *out, int pitch, int dx, int dy, int dw, int dh)
{
	if (!rt::on)
		return bRenderDisplay(out, pitch, dx, dy, dw, dh);
	int r = 0;
	rt::call([&] { r = bRenderDisplay(out, pitch, dx, dy, dw, dh); });
	return r;
}
void fVramWrite(int x, int y, uint16_t pixel)
{
	if (!rt::on)
		return bVramWrite(x, y, pixel);
	rt::put(rt::OpVramWrite, rt::Pixel { x, y, pixel });
	rt::fresh = false;
}
uint16_t fVramRead(int x, int y)
{
	if (!rt::on)
		return bVramRead(x, y);
	rt::freshen();
	return g.cpu[(y & (VramH - 1)) * VramW + (x & (VramW - 1))];
}
void fTransferIn(int x, int y, int w, int h, const uint16_t *data)
{
	if (!rt::on)
		return bTransferIn(x, y, w, h, data);
	if (w <= 0 || h <= 0)
		return;
	rt::put(rt::OpTransferIn, rt::I4 { x, y, w, h });
	rt::putBytes(data, (size_t)w * h * 2);
	rt::bounded();
}
void fTransferOut(int x, int y, int w, int h, uint16_t *data)
{
	if (!rt::on)
		return bTransferOut(x, y, w, h, data);
	rt::freshen();
	for (int j = 0; j < h; j++)
		for (int i = 0; i < w; i++)
			data[j * w + i] = g.cpu[((y + j) & (VramH - 1)) * VramW + ((x + i) & (VramW - 1))];
}
void fSetArea(int x1, int y1, int x2, int y2)
{
	if (!rt::on)
		return bSetArea(x1, y1, x2, y2);
	rt::area[0] = x1, rt::area[1] = y1, rt::area[2] = x2, rt::area[3] = y2;
	rt::put(rt::OpArea, rt::I4 { x1, y1, x2, y2 });
}
void fGetArea(int *x1, int *y1, int *x2, int *y2)
{
	if (!rt::on)
		return bGetArea(x1, y1, x2, y2);
	*x1 = rt::area[0], *y1 = rt::area[1], *x2 = rt::area[2], *y2 = rt::area[3];
}

GpuRenderBackend makeBackend()
{
	GpuRenderBackend b {};
	b.name = "Deadstation Vulkan";
	b.init = fInit;
	b.set_scale = fSetScale;
	b.scale = fScale;
	b.set_texture_filter = fSetFilter;
	b.texture_filter = fFilter;
	b.set_semi_transparency = fSetSemi;
	b.set_mask_bits = fSetMask;
	b.set_texture_window = fSetWindow;
	b.set_color_modulation = fSetModulation;
	b.set_precise_triangle = fSetPrecise;
	b.set_perspective_triangle = nullptr;
	b.fill_rect = fFill;
	b.copy_rect = fCopy;
	b.draw_flat_triangle = fFlatTriangle;
	b.draw_gouraud_triangle = fGouraudTriangle;
	b.draw_textured_triangle = fTexturedTriangle;
	b.draw_shaded_textured_triangle = fShadedTexturedTriangle;
	b.draw_flat_rect = fFlatRect;
	b.draw_textured_rect = fTexturedRect;
	b.draw_textured_rect_scaled = fTexturedRectScaled;
	b.draw_line = fLine;
	b.draw_shaded_line = fShadedLine;
	b.render_display = fRenderDisplay;
	b.render_display_hires = fRenderDisplay;
	b.vram_write = fVramWrite;
	b.vram_read = fVramRead;
	b.vram_transfer_in = fTransferIn;
	b.vram_transfer_out = fTransferOut;
	b.set_draw_area = fSetArea;
	b.get_draw_area = fGetArea;
	b.set_draw_offset = bSetOffset;
	b.wide_configure = bWideConfigure;
	b.wide_set_view = bWideView;
	b.wide_set_target = bWideTarget;
	b.wide_disable_target = bWideNoTarget;
	b.wide_clear = bWideClear;
	b.wide_clear_margins = bWideMargins;
	b.render_wide_display = bWideDisplay;
	b.wide_dump_full = bWideDump;
	b.set_triangle_depth = fSetTriangleDepth;
	return b;
}
const GpuRenderBackend backend = makeBackend();

}

// every Vulkan object of the GPU's, in an order Vulkan allows (command buffers before their pool), after its work
// finished (each slot's fence, a second at most)
void psx_gpu_stop()
{
	if (!rt::on)
		return;
	{
		std::lock_guard<std::mutex> hold(rt::m);
		rt::stop = true;
	}
	rt::cv.notify_all();
	if (rt::thread.joinable() && rt::thread.get_id() != std::this_thread::get_id())
		rt::thread.join();
	rt::on = false;
}

extern "C" const uint16_t *deadstation_vram_view(void)
{
	return g.mirror;
}

void psx_gpu_idle()
{
	if (rt::on)
		rt::call([] {});
}

void psx_gpu_release()
{
	psx_gpu_stop();
	if (!g.ready)
		return;
	g.ready = false;
	vk::Device d = g.dev.device;
	finishPipelines();
	for (Slot& s : g.slots)
	{
		if (!s.recording && s.borrowed >= 0)
			hostVideo().panel().waitDrawn((uint32_t)s.borrowed);
		else if (!s.recording && s.fence)
			(void)d.waitForFences(*s.fence, true, 1000000000ull);
		s.cmd.reset();
		s.fence.reset();
		s.vertices.reset();
		s.verticesMemory.reset();
		s.staging.reset();
		s.stagingMemory.reset();
		s.vertexMap = s.stagingMap = nullptr;
		s.recording = false;
		s.borrowed = -1;
	}
	g.timestamps.reset();
	g.commandPool.reset();
	g.pipelines.clear();
	g.layout.reset();
	g.pool.reset();
	g.setLayout.reset();
	g.sampler.reset();
	g.samplerFiltered.reset();
	g.vert.reset(), g.flat.reset(), g.tex.reset(), g.cached.reset(), g.cachedOpaque.reset(), g.leanVert.reset(), g.worldVert.reset(), g.world.reset(), g.worldOpaque.reset(), g.worldOwnVert.reset(), g.worldOwn.reset(), g.worldSkyVert.reset(), g.worldSky.reset(), g.worldOwnEdge.reset(), g.worldEdge.reset(), g.skyBuffer.reset(), g.skyMemory.reset(), g.worldLayout.reset(), g.worldBuffer.reset(), g.worldMemory.reset(), g.lean.reset(),
			g.leanOpaque.reset(), g.decode.reset(), g.rectVert.reset(), g.blit.reset(), g.pack.reset();
	g.atlasFramebuffer.reset(), g.decodeFramebuffer.reset(), g.atlasPass.reset(), g.atlasDecodePass.reset();
	g.drawFramebuffer.reset(), g.colorFramebuffer.reset(), g.packFramebuffer.reset();
	g.drawPass.reset(), g.colorPass.reset(), g.packPass.reset();
	for (State::Scene& sc : g.scenes)
	{
		sc.framebuffer.reset();
		for (vk::UniqueFramebuffer& there : sc.onPage)
			there.reset();
		sc.pageHolding = -1;
		for (Image *i : { &sc.color, &sc.depth })
			i->view.reset(), i->image.reset(), i->memory.reset();
		sc.used = false;
	}
	for (vk::UniqueRenderPass& pass : g.scenePasses)
		pass.reset();
	for (Image *i : { &g.scaled, &g.stencil, &g.raw, &g.copy, &g.rawCopy, &g.atlas, &g.decodeImage })
		i->view.reset(), i->image.reset(), i->memory.reset();
	g.readback.reset();
	g.readbackMemory.reset();
	g.readbackMap = nullptr;
}

// ---- PSXRecomp's Vulkan entry points (gpu_vk_renderer.h), Deadstation's on the Brick ----

extern "C" {

const GpuRenderBackend *vk_backend_get(void)
{
	return &backend;
}

int vk_renderer_init_context(struct SDL_Window *)
{
	if (!g.ready && !init())
		return 0;
	rt::start();
	return 1;
}

void vk_renderer_shutdown(void)
{
	if (!rt::on)
		shutdownGpu();
	else
		rt::call([] { shutdownGpu(); });
}

int deadstation_no_present = 0;

int vk_renderer_present_vram(int dx, int dy, int w, int h, int, int)
{
	if (deadstation_no_present)
	{
		deadstation_no_present = 0;
		hostInput();
		return 1;
	}
	if (!rt::on)
	{
		const int shown = presentVram(dx, dy, w, h);
		hostInput();
		return shown;
	}
	rt::put(rt::OpPresentVram, rt::I4 { dx, dy, w, h });
	rt::presented();
	hostInput();
	return 1;
}

int vk_renderer_present_wide(int, int, int, int)
{
	return 0;
}

void vk_renderer_present_cpu(const uint32_t *pixels, int w, int h, int, int)
{
	if (!rt::on)
	{
		presentCpu(pixels, w, h);
		hostInput();
		return;
	}
	if (pixels == nullptr || w <= 0 || h <= 0)
		return;
	rt::put(rt::OpPresentCpu, rt::I2 { w, h });
	rt::putBytes(pixels, (size_t)w * h * 4);
	rt::presented();
	hostInput();
}

void vk_renderer_present_blank(void)
{
	if (!rt::on)
		presentBlank();
	else
	{
		rt::putOp(rt::OpPresentBlank);
		rt::presented();
	}
	hostInput();
}

void vk_renderer_sync_cpu(void)
{
	if (!rt::on)
		syncCpu();
	else
		rt::call([] { syncCpu(); });
}

void vk_renderer_restage_vram_after_savestate(void)
{
	if (!rt::on)
		restage();
	else
		rt::call([] { restage(); });
}

void vk_renderer_set_present_mode(int) { }
void vk_renderer_set_display_aspect(int, int) { }
int vk_perf_json(char *out, int cap, int)
{
	return cap > 2 ? snprintf(out, cap, "[]") : 0;
}

}

// a module's plugin: the frame now going to the GPU is 3D whose depths are true (perspective-correct in the scene), or not
extern "C" void deadstation_scene_perspective(int on)
{
	fFrameIs3D(on);
}

extern "C" void deadstation_replay_frame(const deadstation::replay::Frame *frame)
{
	deadstation::replay::Frame none {};
	const deadstation::replay::Frame& f = frame != nullptr ? *frame : none;
	if (rt::on)
		rt::put(rt::OpReplayFrame, f);
	else
	{
		g.replayLatest = f;
		g.replayPending = f.valid;
	}
}
extern "C" void deadstation_replay_steps(const deadstation::replay::Steps *steps)
{
	if (rt::on)
		rt::put(rt::OpReplaySteps, *steps);
	else
	{
		g.replaySteps = *steps;
		g.replayStepsFor = g.replayNewest;
	}
}
extern "C" void deadstation_replay_object(int object)
{
	if (rt::on)
		rt::put(rt::OpReplayObject, rt::I4 { object, 0, 0, 0 });
	else
		g.replayObject = object;
}
extern "C" unsigned deadstation_replay_take_count(void)
{
	const unsigned n = g.replayFrames;
	g.replayFrames = 0;
	return n;
}

extern "C" void deadstation_tint(int r, int g, int b)
{
	if (rt::on)
		rt::put(rt::OpTint, rt::I4 { r, g, b, 0 });
	else
		bSetTint(r, g, b);
}

static void putGlows(const DeadstationGlow *glows, size_t count, int mode)
{
	if (count == 0)
		return;
	if (rt::on)
	{
		rt::put(rt::OpGlows, rt::I4 { (int)count, mode, 0, 0 });
		rt::putBytes(glows, count * sizeof(DeadstationGlow));
		rt::bounded();
		return;
	}
	bulkGlows((const uint8_t *)glows, (int)count, mode);
}
extern "C" void deadstation_glows(const DeadstationGlow *glows, size_t count)
{
	putGlows(glows, count, 0);
}
extern "C" void deadstation_veil(const DeadstationGlow *glows, size_t count)
{
	putGlows(glows, count, 1);
}
extern "C" void deadstation_shadows(const DeadstationGlow *polygons, size_t count, int darken)
{
	putGlows(polygons, count, darken != 0 ? 2 : 3);
}
extern "C" void deadstation_flat(const DeadstationGlow *polygons, size_t count)
{
	putGlows(polygons, count, 5);
}
extern "C" void deadstation_depth_only(const DeadstationGlow *polygons, size_t count)
{
	putGlows(polygons, count, 4);
}

extern "C" void deadstation_world_load(const DeadstationWorldTriangle *triangles, size_t count, unsigned cells)
{
	if (rt::on)
		rt::call([=] { bWorldLoad(triangles, count, cells); });
	else
		bWorldLoad(triangles, count, cells);
}
// (to measure with, by a module: kinds of the mesh left out, by bit)
int deadstation_world_skip = 0;
extern "C" void deadstation_world_shaders(const DeadstationWorldShaders *shaders)
{
	worldShadersGiven = *shaders;
}
extern "C" void deadstation_world_numbers(const float *solid, const float *others, int ownSolid)
{
	WorldNumbersOp n {};
	if (solid != nullptr)
		std::memcpy(n.numbers[0], solid, sizeof(n.numbers[0]));
	if (others != nullptr)
		std::memcpy(n.numbers[1], others, sizeof(n.numbers[1]));
	n.ownSolid = ownSolid;
	if (rt::on)
	{
		rt::putOp(rt::OpWorldLights);
		rt::putBytes(&n, sizeof(n));
	}
	else
	{
		std::memcpy(g.worldNumbers, n.numbers, sizeof(g.worldNumbers));
		g.worldOwnSolid = ownSolid != 0;
	}
}
extern "C" void deadstation_scene_whole(float r, float g_, float b)
{
	if (rt::on)
		rt::put(rt::OpSceneWhole, rt::F4 { 0, r, g_, b });
	else
		g.sceneWhole = true, g.sceneClear[0] = r, g.sceneClear[1] = g_, g.sceneClear[2] = b;
}
extern "C" void deadstation_world_sky_load(const DeadstationWorldTriangle *triangles, size_t count)
{
	if (rt::on)
		rt::call([=] { bWorldSkyLoad(triangles, count); });
	else
		bWorldSkyLoad(triangles, count);
}
extern "C" void deadstation_world_sky(const DeadstationWorldCamera *camera)
{
	if (rt::on)
	{
		rt::put(rt::OpWorldSky, *camera);
		return;
	}
	bWorldSky(*camera);
}
extern "C" void deadstation_world_quads(const DeadstationWorldCamera *camera, const DeadstationWorldQuad *quads, size_t count)
{
	if (count == 0)
		return;
	if (rt::on)
	{
		rt::put(rt::OpWorldQuads, rt::I4 { (int)count, 0, 0, 0 });
		rt::putBytes(camera, sizeof(*camera));
		rt::putBytes(quads, count * sizeof(DeadstationWorldQuad));
		rt::bounded();
		return;
	}
	bWorldQuads(*camera, (const uint8_t *)quads, (int)count);
}
extern "C" void deadstation_fog(float nearZ, float farZ, float most, float r, float g, float b)
{
	const rt::FogOp a { nearZ, farZ, most, r, g, b };
	if (rt::on)
		rt::put(rt::OpFog, a);
	else
		rt::bSetFog(a);
}
extern "C" void deadstation_world_animated(const int16_t *rects, size_t count)
{
	const auto set = [=] { g.animatedRects.assign(rects, rects + count * 4); };
	if (rt::on)
		rt::call(set);
	else
		set();
}
extern "C" void deadstation_world_draw(const DeadstationWorldCamera *camera, const uint16_t *cells, size_t count, int parts)
{
	if (count == 0)
		return;
	if (rt::on)
	{
		rt::put(rt::OpWorldDraw, rt::I4 { (int)count, parts, 0, 0 });
		rt::putBytes(camera, sizeof(*camera));
		rt::putBytes(cells, count * sizeof(uint16_t));
		if (count & 1)
			rt::putBytes(cells, sizeof(uint16_t));	// (what follows stays as it was aligned)
		rt::bounded();
		return;
	}
	bWorldDraw(*camera, cells, (int)count, parts);
}
extern "C" void deadstation_triangles(const DeadstationTriangle *triangles, size_t count)
{
	if (count == 0)
		return;
	if (rt::on)
	{
		rt::put(rt::OpTriangles, rt::I4 { (int)count, 0, 0, 0 });
		rt::putBytes(triangles, count * sizeof(DeadstationTriangle));
		rt::bounded();
		return;
	}
	bulkTriangles((const uint8_t *)triangles, (int)count);
}
