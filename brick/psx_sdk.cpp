// The PlayStation SDK as host code (psx_sdk.h). Each replacement does what the library call achieves, directly:
//  - DrawOTag: the ordering table's packets to the GPU at once (PSXRecomp's psx_native_gpu_list, patch 0017), where
//    the library would start a linked-list DMA that the runtime walks packet by packet in device time;
//  - LoadImage / MoveImage: the GPU's own commands and pixels to its port, no DMA block transfer;
//  - VSync's wait: guest time straight to the next vertical blank, where the library polls the blank counter for the
//    rest of the frame (and times out after 32768 polls, a count, not a time: with the runtime's lean CPU timing a
//    frame's wait could run out before the blank).
// libgpu runs these through its command queue only to wait for a busy GPU; done here at once when the GPU is free (no
// GPU DMA running, GP0 between commands), the queue (and DrawSync) stay as the game's and see an idle GPU. A call made
// while the GPU is busy stays the game's, queued behind the work in order.
#include "psx_sdk.h"

#include "cpu_state.h"
#include "mod_plugins.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "interrupts.h"
#include "psx_cycles.h"
void gpu_write_gp0(uint32_t val);
void gpu_write_gp1(uint32_t val);
uint32_t psx_native_gpu_list(uint32_t addr);
int psx_gpu_gp0_idle(void);
int psx_gpu_dma_busy(void);
}

namespace
{
PsxSdkAddresses sdk;

uint16_t half(uint32_t a) { return psx_mod_read_half(a); }

// the GPU free for a whole command from the host: no GPU DMA running (libgpu would queue the call behind it, in order)
// and GP0 between commands. Otherwise the call stays the game's, which queues it.
bool gpuFree()
{
	return !psx_gpu_dma_busy() && psx_gpu_gp0_idle();
}

int onDrawOTag(CPUState *cpu, uint32_t)
{
	if (sdk.beforeDrawOTag != nullptr)
		sdk.beforeDrawOTag();
	if (!gpuFree())
		return 0;
	// (0: nothing left to walk)
	const uint32_t from = sdk.drawOTagFrom != nullptr ? sdk.drawOTagFrom(cpu->gpr[4]) : cpu->gpr[4];
	if (from != 0)
		psx_native_gpu_list(from);
	return 1;
}

// RECT: x, y, w, h (signed 16-bit). As libgpu's driver: w and h clamped to [0, the GPU's size], nothing sent for an
// empty image (-1), the texture cache cleared, then the transfer's header and (w * h + 1) / 2 words.
int onLoadImage(CPUState *cpu, uint32_t)
{
	if (!gpuFree())
		return 0;
	const uint32_t rect = cpu->gpr[4], p = cpu->gpr[5];
	const int32_t maxW = sdk.gpuSize ? (int16_t)half(sdk.gpuSize) : 1024, maxH = sdk.gpuSize ? (int16_t)half(sdk.gpuSize + 2) : 512;
	const int32_t w = std::clamp<int32_t>((int16_t)half(rect + 4), 0, maxW), h = std::clamp<int32_t>((int16_t)half(rect + 6), 0, maxH);
	const int32_t words = (w * h + 1) / 2;
	if (words < 1)
	{
		cpu->gpr[2] = (uint32_t)-1;
		return 1;
	}
	gpu_write_gp1(0x04000000u);
	gpu_write_gp0(0x01000000u);
	gpu_write_gp0(0xA0000000u);
	gpu_write_gp0(psx_mod_read_word(rect));
	gpu_write_gp0((uint32_t)h << 16 | (uint32_t)w);
	for (int32_t i = 0; i < words; i++)
		gpu_write_gp0(psx_mod_read_word(p + 4u * (uint32_t)i));
	cpu->gpr[2] = 0;
	return 1;
}

int onMoveImage(CPUState *cpu, uint32_t)
{
	if (!gpuFree())
		return 0;
	const uint32_t rect = cpu->gpr[4];
	const uint16_t x = half(rect), y = half(rect + 2), w = half(rect + 4), h = half(rect + 6);
	if (w == 0 || h == 0)
	{
		cpu->gpr[2] = (uint32_t)-1;
		return 1;
	}
	gpu_write_gp0(0x80000000u);
	gpu_write_gp0((uint32_t)y << 16 | x);
	gpu_write_gp0((cpu->gpr[6] & 0xFFFF) << 16 | (cpu->gpr[5] & 0xFFFF));
	gpu_write_gp0((uint32_t)h << 16 | w);
	cpu->gpr[2] = 0;
	return 1;
}

// v_wait(target, timeout): when the blank counter is short of target, guest time goes straight to the next blank (the
// devices catch up in one service, its interrupt pending); then the library's own wait runs, and its first branch takes
// that interrupt at a normal check point (libetc's handler counts it, the game's VSync callback runs), so it returns
// after one poll instead of polling a frame away. Interrupts are never delivered from here (a filter is not a point the
// runtime's exception path may unwind).
int onVsyncWait(CPUState *cpu, uint32_t)
{
	const int32_t target = (int32_t)cpu->gpr[4];
	if ((int32_t)psx_mod_read_word(sdk.vsyncCounter) < target)
	{
		psx_cyc_batch_flush();
		psx_advance_cycles(interrupts_cycles_to_vblank() + 1u);
	}
	return 0;
}

bool off(const char *name)
{
	const char *list = std::getenv("DEADSTATION_SDK_OFF");
	return list != nullptr && std::strstr(list, name) != nullptr;
}
}

void psx_sdk_install(const char *pluginId, const PsxSdkAddresses& addresses)
{
	// DEADSTATION_SDK_OFF=drawotag,loadimage,moveimage,vsync: those left as the game's (comparisons)
	sdk = addresses;
	if (sdk.drawOTag && !off("drawotag"))
		psx_mod_register_function_filter_plugin(pluginId, sdk.drawOTag, onDrawOTag);
	if (sdk.loadImage && !off("loadimage"))
		psx_mod_register_function_filter_plugin(pluginId, sdk.loadImage, onLoadImage);
	if (sdk.moveImage && !off("moveimage"))
		psx_mod_register_function_filter_plugin(pluginId, sdk.moveImage, onMoveImage);
	if (sdk.vsyncWait && sdk.vsyncCounter && !off("vsync"))
		psx_mod_register_function_filter_plugin(pluginId, sdk.vsyncWait, onVsyncWait);
}
