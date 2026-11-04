// The PlayStation SDK (PsyQ) as host code for a recomp module: the library functions a game links statically,
// replaced at their entries (PSXRecomp function filters) so they no longer drive the simulated hardware. Each module
// gives the addresses of its own copies (its game.toml lists them in mod_function_entry_funcs) and calls
// psx_sdk_install from its plugin. A zero address leaves that function as the game's.
#pragma once
#include <cstdint>

struct PsxSdkAddresses
{
	// libgpu
	uint32_t drawOTag = 0;		// DrawOTag(u_long *ot): the display list straight to the GPU, no DMA
	uint32_t loadImage = 0;		// LoadImage(RECT *, u_long *): pixels straight to the GPU
	uint32_t moveImage = 0;		// MoveImage(RECT *, int x, int y)
	uint32_t gpuSize = 0;		// libgpu's limits for images: width, height (s16 each; 0: 1024 x 512)
	// libetc: VSync's wait for the vertical blank counter (v_wait(target, timeout)) and that counter
	uint32_t vsyncWait = 0;
	uint32_t vsyncCounter = 0;
	void (*beforeDrawOTag)() = nullptr;	// told as the game draws an ordering table
	uint32_t (*drawOTagFrom)(uint32_t ot) = nullptr;	// where in it to start: a module that's drawn its frame itself, what's left; 0, nothing
};

void psx_sdk_install(const char *pluginId, const PsxSdkAddresses& addresses);

// VRAM as the renderer holds it, 1024 x 512 of 16-bit pixels, or nullptr. To read, from the module's thread, what the game
// uploaded and leaves alone (a sprite sheet, a CLUT): the render thread may still have the latest uploads to come.
extern "C" const uint16_t *deadstation_vram_view(void);

// The settings a module's plugin applies while the game runs, rather than as the module starts (Options, host_link.cpp):
// apply is asked of each one as it changes (key, value); true if it is one.
extern "C" void deadstation_live_settings(bool (*apply)(const char *key, const char *value));
