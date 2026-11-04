// The Brick's screen. VK_KHR_display tears here, so Vulkan renders straight into /dev/fb0: its memory is
// imported as a dma-buf (Allwinner's FBIOGET_DMABUF) and its pages from FirstPage on are linear BGRA8
// render targets. A thread pans to each frame once its fence signals (FBIOPAN_DISPLAY waits for the
// vblank). Frames are 640 x 480, with fb0 set to that mode and the display engine scaling it to the
// panel, which cuts memory traffic. NextUI's mode and page are restored at the end.
#pragma once
#include "vk.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <linux/fb.h>

class Panel
{
public:
	static constexpr uint32_t Pages = 4;

	~Panel() { term(); }
	// false, with the reason in error, without fb0 or its import
	// native: fb0 at the panel's own size (1024 x 768), else 640 x 480 scaled by the display engine
	bool init(vk::Device device, bool native, std::string& error, uint32_t smallWidth = 640);
	void term();

	vk::Extent2D extent() const { return { mode.xres, mode.yres }; }
	bool nativeMode() const { return mode.xres == var.xres && mode.yres == var.yres; }	// frames at the panel's own size
	static constexpr vk::Format format() { return vk::Format::eB8G8R8A8Unorm; }
	vk::ImageView view(uint32_t page) const { return *views[page]; }
	vk::Image image(uint32_t page) const { return *images[page]; }
	float refreshRate() const { return refresh; }

	// the frame on the screen as RGB rows, read back from fb0
	bool readShown(std::vector<uint8_t>& rgb, int& width, int& height);

	// CPU-drawn pictures (the menu, the start page) between frames, at the panel's native size: fb0 in
	// NextUI's mode, two pages of their own after ours, written then shown in turn. endPictures restores
	// Deadstation's mode with its last frame (LastFrame), or with the next one (NextFrame, LastFrame without
	// one yet); Stay leaves the last picture up, in NextUI's page too (through Deadstation's end).
	enum class After { LastFrame, NextFrame, Stay };
	bool beginPictures(int& width, int& height);
	void showPicture(const uint32_t *bgra);
	void endPictures(After after);

	// before a frame is recorded into page: until it may be written (off the screen) and at most one
	// frame waits to be shown
	void waitWritable(uint32_t page);
	// the GPU done with every frame queued for this page. This thread alone waits on their fences: the driver releases a fence's kernel handle as a wait on it ends, so two threads
	// ending their waits on one fence release it twice (PVRSRV_FENCE_SERVER already freed, in dmesg)
	void waitDrawn(uint32_t page);
	// the frame recorded into page, submitted with fence: shown once the fence has signalled
	void queue(uint32_t page, vk::Fence fence);
	// until every frame queued is shown
	void drain();

	// small pictures copied onto every frame once it is drawn, just before it is shown, at the bottom left: only their text's
	// pixels (not the colour Corner::Box), by the CPU into the page, so that the GPU's frame and its cost are untouched
	struct Corner
	{
		std::vector<uint32_t> pixels;	// BGRA, width x height
		int width = 0, height = 0;
		int bottom = 0;					// its bottom edge, up from the screen's
		static constexpr uint32_t Box = 0x202020;
	};
	void setCorners(std::vector<Corner> corners);
	// the frames shown since init
	uint64_t framesShown();

private:
	void showLoop();
	void drawCorners(uint32_t page);
	void frameMode(uint32_t page);
	void pan(uint32_t page);
	void restoreMode();

	vk::Device device;
	int fd = -1;
	// fb0 mapped uncached, as the display reads it: write() and read() go through a cached kernel
	// mapping, which left parts of pictures stale on the screen
	uint8_t *mapped = nullptr;
	size_t mappedSize = 0;
	fb_var_screeninfo var{};		// as found: NextUI's mode and page
	fb_var_screeninfo mode{};		// Deadstation's
	uint32_t firstRow = 0;			// the first page's
	uint32_t lineLength = 0;		// fb0's, in bytes
	std::atomic<int> shownPage { -1 };	// the page panned to last
	uint32_t nativeLine = 0;		// fb0's line, in bytes, in NextUI's mode
	uint32_t pictureRow = 0;		// the first picture page's, in NextUI's mode
	uint32_t picturePage = 0;		// the one written next
	bool pictures = false;			// between beginPictures and endPictures
	bool pictureMode = false;		// fb0 in NextUI's mode for them
	std::atomic<bool> modeAtFrame { false };	// to Deadstation's mode with the next frame shown
	float refresh = 60.f;
	vk::UniqueDeviceMemory memory;
	std::vector<vk::UniqueImage> images;
	std::vector<vk::UniqueImageView> views;

	struct Frame
	{
		uint32_t page;
		vk::Fence fence;
	};
	std::thread thread;
	std::mutex mutex;
	std::condition_variable changed;
	std::deque<Frame> frames;		// queued, not shown yet
	uint64_t queuedCount = 0;
	uint64_t queuedFor[Pages] = {}, drawnFor[Pages] = {};
	uint64_t shownCount = 0;
	bool stopping = false;
	std::mutex cornersMutex;
	struct CornerRun
	{
		size_t offset;		// in a page, in bytes
		uint32_t source;	// in cornerPixels
		uint32_t count;
	};
	std::vector<uint32_t> cornerPixels;
	std::vector<CornerRun> cornerRuns;
};
