#include "panel.h"
#include "log.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace
{
constexpr uint32_t FirstPage = 16;		// in NextUI's mode: its pages before, ours from there

// Allwinner's dma-buf export of the framebuffer's memory
struct FbDmabuf
{
	uint32_t fd;
	uint32_t flags;
};
constexpr unsigned long FBIOGET_DMABUF = _IOR('F', 0x21, FbDmabuf);
}

bool Panel::init(vk::Device device, bool native, std::string& error, uint32_t smallWidth)
{
	this->device = device;
	fd = open("/dev/fb0", O_RDWR | O_CLOEXEC);
	if (fd < 0)
	{
		error = std::string("/dev/fb0: ") + strerror(errno);
		return false;
	}
	fb_fix_screeninfo fix{};
	if (ioctl(fd, FBIOGET_VSCREENINFO, &var) != 0 || ioctl(fd, FBIOGET_FSCREENINFO, &fix) != 0)
	{
		error = "fb0: no screen info";
		return false;
	}
	if (var.bits_per_pixel != 32 || var.red.offset != 16 || var.green.offset != 8 || var.blue.offset != 0)
	{
		error = "fb0 is not BGRA8";
		return false;
	}
	nativeLine = fix.line_length;
	void *map = mmap(nullptr, fix.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (map != MAP_FAILED)
	{
		mapped = (uint8_t *)map;
		mappedSize = fix.smem_len;
	}
	else
		DS_LOG("panel: fb0 not mapped: %s", strerror(errno));
	const double line = var.xres + var.left_margin + var.right_margin + var.hsync_len;
	const double rows = var.yres + var.upper_margin + var.lower_margin + var.vsync_len;
	const double hz = var.pixclock != 0 ? 1e12 / (var.pixclock * line * rows) : 60.0;
	refresh = hz >= 30 && hz <= 120 ? (float)hz : 60.f;

	// smallWidth at 4:3 (640 x 480 unless told), scaled by the display engine, unless native
	mode = var;
	const uint32_t smallHeight = smallWidth * 3 / 4;
	if (!native && var.xres >= smallWidth && var.yres >= smallHeight)
	{
		mode.xres = smallWidth;
		mode.yres = smallHeight;
		mode.xres_virtual = smallWidth;
		mode.xoffset = mode.yoffset = 0;
		if (ioctl(fd, FBIOPUT_VSCREENINFO, &mode) != 0 || ioctl(fd, FBIOGET_VSCREENINFO, &mode) != 0
				|| ioctl(fd, FBIOGET_FSCREENINFO, &fix) != 0 || mode.xres != smallWidth || mode.yres != smallHeight)
		{
			DS_LOG("fb0 cannot be %u x %u, the panel's size then", smallWidth, smallHeight);
			restoreMode();
			mode = var;
			ioctl(fd, FBIOGET_FSCREENINFO, &fix);
		}
	}
	firstRow = FirstPage * var.yres;
	lineLength = fix.line_length;
	const uint32_t imageWidth = fix.line_length / 4;
	const vk::DeviceSize pageBytes = (vk::DeviceSize)fix.line_length * mode.yres;
	const vk::DeviceSize size = (vk::DeviceSize)fix.line_length * (firstRow + mode.yres * Pages);
	if (fix.line_length % 4 != 0 || imageWidth < mode.xres || size > fix.smem_len || mode.yres_virtual < firstRow + mode.yres * Pages)
	{
		error = "fb0 has too few pages";
		restoreMode();
		return false;
	}

	FbDmabuf dmabuf{};
	if (ioctl(fd, FBIOGET_DMABUF, &dmabuf) != 0)
	{
		error = std::string("fb0: no dma-buf (") + strerror(errno) + ")";
		restoreMode();
		return false;
	}
	vk::MemoryFdPropertiesKHR fdProperties = device.getMemoryFdPropertiesKHR(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, (int)dmabuf.fd);
	if (fdProperties.memoryTypeBits == 0)
	{
		::close((int)dmabuf.fd);
		error = "fb0's dma-buf: no memory type";
		restoreMode();
		return false;
	}
	const uint32_t type = __builtin_ctz(fdProperties.memoryTypeBits);
	vk::ImportMemoryFdInfoKHR import(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, (int)dmabuf.fd);	// the memory's once imported
	memory = device.allocateMemoryUnique(vk::MemoryAllocateInfo(size, type, &import));
	for (uint32_t i = 0; i < Pages; i++)
	{
		vk::ExternalMemoryImageCreateInfo external(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT);
		// as wide as fb0's lines: frames are drawn in its first mode.xres columns
		vk::ImageCreateInfo imageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, format(), vk::Extent3D(imageWidth, mode.yres, 1), 1, 1,
				vk::SampleCountFlagBits::e1, vk::ImageTiling::eLinear, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
				vk::SharingMode::eExclusive, nullptr, vk::ImageLayout::eUndefined, &external);
		images.push_back(device.createImageUnique(imageCreateInfo));
		const vk::SubresourceLayout layout = device.getImageSubresourceLayout(*images.back(), vk::ImageSubresource(vk::ImageAspectFlagBits::eColor, 0, 0));
		const vk::MemoryRequirements req = device.getImageMemoryRequirements(*images.back());
		const vk::DeviceSize offset = (vk::DeviceSize)fix.line_length * firstRow + i * pageBytes;
		if (layout.rowPitch != fix.line_length || layout.offset != 0 || !(req.memoryTypeBits & (1u << type))
				|| offset % req.alignment != 0 || offset + req.size > size)
		{
			error = "fb0's pages cannot be images";
			images.clear();
			memory.reset();
			restoreMode();
			return false;
		}
		device.bindImageMemory(*images.back(), *memory, offset);
		views.push_back(device.createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), *images.back(), vk::ImageViewType::e2D,
				format(), vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1))));
	}
	DS_LOG("panel %d x %d on %d x %d, %.2f Hz, fb0 rows %d to %d", mode.xres, mode.yres, var.xres, var.yres, refresh,
			firstRow, firstRow + mode.yres * Pages - 1);
	// NextUI's mode and picture until the first frame
	if (mode.xres != var.xres || mode.yres != var.yres)
	{
		fb_var_screeninfo v = var;
		ioctl(fd, FBIOPUT_VSCREENINFO, &v);
		v = var;
		ioctl(fd, FBIOPAN_DISPLAY, &v);
		modeAtFrame = true;
	}
	stopping = false;
	pictures = pictureMode = false;
	shownPage = -1;
	queuedCount = shownCount = 0;
	thread = std::thread([this] { showLoop(); });
	return true;
}

void Panel::term()
{
	if (thread.joinable())
	{
		{
			std::lock_guard<std::mutex> lock(mutex);
			stopping = true;
		}
		changed.notify_all();
		thread.join();
	}
	views.clear();
	images.clear();
	if (fd >= 0)
	{
		restoreMode();		// NextUI's mode and page again
		if (mapped != nullptr)
			munmap(mapped, mappedSize);
		mapped = nullptr;
		::close(fd);
		fd = -1;
	}
	memory.reset();
}

void Panel::waitWritable(uint32_t)
{
	std::unique_lock<std::mutex> lock(mutex);
	changed.wait(lock, [this] { return queuedCount - shownCount < 2; });
}

void Panel::waitDrawn(uint32_t page)
{
	std::unique_lock<std::mutex> lock(mutex);
	changed.wait(lock, [this, page] { return drawnFor[page] == queuedFor[page]; });
}

void Panel::queue(uint32_t page, vk::Fence fence)
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		frames.push_back({ page, fence });
		queuedCount++;
		queuedFor[page]++;
	}
	changed.notify_all();
}

void Panel::drain()
{
	std::unique_lock<std::mutex> lock(mutex);
	changed.wait(lock, [this] { return shownCount == queuedCount; });
}

void Panel::showLoop()
{
	pthread_setname_np(pthread_self(), "ds-panel");
	std::unique_lock<std::mutex> lock(mutex);
	while (true)
	{
		changed.wait(lock, [this] { return stopping || !frames.empty(); });
		if (frames.empty())
			break;
		Frame frame = frames.front();
		lock.unlock();
		const auto from = std::chrono::steady_clock::now();
		vk::Result res = device.waitForFences(frame.fence, true, UINT64_MAX);
		if (res != vk::Result::eSuccess)
			DS_LOG("panel: waitForFences failed %d", (int)res);
		{
			std::lock_guard<std::mutex> drawn(mutex);
			drawnFor[frame.page]++;
		}
		changed.notify_all();
		const auto fenced = std::chrono::steady_clock::now();
		// (tests: a frame much brighter or darker than the last, as the panel will show it, wherever that came from: a flash, if it's for a frame)
		static const bool watching = getenv("DEADSTATION_PERF") != nullptr;
		if (watching && mapped != nullptr)
		{
			static float lastLight = -1.f;
			uint32_t sum = 0;
			const uint32_t pageRow = firstRow + frame.page * mode.yres;
			for (uint32_t row = 1; row < 9; row++)
				for (uint32_t column = 1; column < 9; column++)
				{
					const uint8_t *px = mapped + (size_t)lineLength * (pageRow + mode.yres * row / 9) + (size_t)(mode.xres * column / 9) * 4;
					sum += ((uint32_t)px[0] + px[1] + px[2]) * (px[3] == 255 ? 1u : 0u);
				}
			const float light = (float)sum / (64.f * 3.f);
			if (lastLight >= 0.f && std::fabs(light - lastLight) > std::max(10.f, lastLight * 0.3f))
			{
				timespec up {};
				clock_gettime(CLOCK_BOOTTIME, &up);
				DS_LOG("panel: light %.0f to %.0f at uptime %.3f, page %u", lastLight, light, (double)up.tv_sec + up.tv_nsec * 1e-9, frame.page);
			}
			lastLight = light;
		}
		drawCorners(frame.page);
		if (modeAtFrame.exchange(false))
			frameMode(frame.page);
		pan(frame.page);
		const auto panned = std::chrono::steady_clock::now();
		const double fenceMs = std::chrono::duration<double>(fenced - from).count() * 1000.0, flipMs = std::chrono::duration<double>(panned - fenced).count() * 1000.0;
		if (fenceMs > 50.0 || flipMs > 50.0)
			DS_LOG("panel: a slow frame: waiting for the GPU %.0f ms, the flip %.0f ms", fenceMs, flipMs);
		shownPage = (int)frame.page;
		lock.lock();
		frames.pop_front();
		shownCount++;
		changed.notify_all();
	}
}

void Panel::setCorners(std::vector<Corner> corners)
{
	if (mapped == nullptr && !corners.empty())
		DS_LOG("panel: fb0 is not mapped: no counter or messages on the screen");
	std::vector<uint32_t> pixels;
	std::vector<CornerRun> runs;
	for (const Corner& corner : corners)
	{
		const int top = (int)mode.yres - corner.bottom - corner.height;
		if (top < 0 || corner.width > (int)mode.xres)
			continue;
		for (int y = 0; y < corner.height; y++)
		{
			const uint32_t *row = &corner.pixels[(size_t)y * corner.width];
			for (int x = 0; x < corner.width; )
			{
				if ((row[x] & 0xffffff) == Corner::Box)
				{
					x++;
					continue;
				}
				int end = x;
				while (end < corner.width && (row[end] & 0xffffff) != Corner::Box)
					end++;
				runs.push_back({ (size_t)lineLength * (top + y) + (size_t)x * 4, (uint32_t)pixels.size(), (uint32_t)(end - x) });
				pixels.insert(pixels.end(), row + x, row + end);
				x = end;
			}
		}
	}
	std::lock_guard<std::mutex> lock(cornersMutex);
	cornerPixels = std::move(pixels);
	cornerRuns = std::move(runs);
}

uint64_t Panel::framesShown()
{
	std::lock_guard<std::mutex> lock(mutex);
	return shownCount;
}

// the frame's fence has signalled and nothing shows this page yet
void Panel::drawCorners(uint32_t page)
{
	std::lock_guard<std::mutex> lock(cornersMutex);
	if (mapped == nullptr)
		return;
	uint8_t *start = mapped + (size_t)lineLength * (firstRow + page * mode.yres);
	for (const CornerRun& run : cornerRuns)
		memcpy(start + run.offset, &cornerPixels[run.source], (size_t)run.count * 4);
}

bool Panel::readShown(std::vector<uint8_t>& rgb, int& width, int& height)
{
	const int page = shownPage;
	if (fd < 0 || page < 0)
		return false;
	width = mode.xres;
	height = mode.yres;
	std::vector<uint8_t> row(lineLength);
	rgb.resize((size_t)width * height * 3);
	// (the display blends by each pixel's alpha, over black: what's not opaque here isn't what's on the panel, whatever these colours say)
	size_t notOpaque = 0;
	for (int y = 0; y < height; y++)
	{
		const off_t offset = (off_t)lineLength * (firstRow + page * mode.yres + y);
		if (mapped != nullptr)
			memcpy(row.data(), mapped + offset, (size_t)width * 4);
		else if (pread(fd, row.data(), lineLength, offset) != (ssize_t)lineLength)
			return false;
		uint8_t *out = &rgb[(size_t)y * width * 3];
		for (int x = 0; x < width; x++)
		{
			out[x * 3] = row[x * 4 + 2];
			out[x * 3 + 1] = row[x * 4 + 1];
			out[x * 3 + 2] = row[x * 4];
			notOpaque += row[x * 4 + 3] != 255;
		}
	}
	DS_LOG("panel: a picture of the page shown: %.2f%% of it not opaque", 100.0 * (double)notOpaque / ((double)width * height));
	return true;
}

bool Panel::beginPictures(int& width, int& height)
{
	if (fd < 0 || nativeLine < var.xres * 4)
		return false;
	drain();
	// past our pages, and past the one NextUI showed last (shown again at the end)
	const uint64_t ours = (uint64_t)lineLength * (firstRow + Pages * mode.yres);
	uint32_t row = (uint32_t)((ours + nativeLine - 1) / nativeLine);
	if (row < var.yoffset + var.yres && row + 2 * var.yres > var.yoffset)
		row = var.yoffset + var.yres;
	if (row + 2 * var.yres > var.yres_virtual)
	{
		// at the panel's own size ours reach too far: before them then, off NextUI's page (NextUI is not
		// running while a game is; its page is put back at the end)
		const uint32_t before = (uint32_t)((uint64_t)lineLength * firstRow / nativeLine);
		row = var.yoffset >= 2 * var.yres ? 0 : var.yoffset + var.yres;
		if (row + 2 * var.yres > before)
		{
			DS_LOG("panel: no room in fb0 for pictures");
			return false;
		}
	}
	pictureRow = row;
	picturePage = 0;
	pictures = true;
	// still in NextUI's mode if no frame came after the last ones
	pictureMode = modeAtFrame.exchange(false);
	width = var.xres;
	height = var.yres;
	return true;
}

void Panel::showPicture(const uint32_t *bgra)
{
	if (!pictures)
		return;
	const uint32_t row = pictureRow + picturePage * var.yres;
	const size_t rowBytes = (size_t)var.xres * 4;
	if (mapped != nullptr)
	{
		for (uint32_t y = 0; y < var.yres; y++)
			memcpy(mapped + (size_t)nativeLine * (row + y), bgra + (size_t)y * var.xres, rowBytes);
		// all of it in memory before the display is pointed at it
		__asm__ volatile("dsb st" ::: "memory");
	}
	else
	{
		for (uint32_t y = 0; y < var.yres; y++)
			if (pwrite(fd, bgra + (size_t)y * var.xres, rowBytes, (off_t)nativeLine * (row + y)) != (ssize_t)rowBytes)
			{
				DS_LOG("panel: picture not written: %s", strerror(errno));
				break;
			}
	}
	fb_var_screeninfo v = var;
	v.xoffset = 0;
	v.yoffset = row;
	// the first: NextUI's mode, already at this page
	if (!pictureMode && (mode.xres != var.xres || mode.yres != var.yres) && ioctl(fd, FBIOPUT_VSCREENINFO, &v) != 0)
		DS_LOG("panel: fb0 not in its own mode for pictures: %s", strerror(errno));
	pictureMode = true;
	v = var;
	v.xoffset = 0;
	v.yoffset = row;
	ioctl(fd, FBIOPAN_DISPLAY, &v);
	picturePage ^= 1;
}

void Panel::endPictures(After after)
{
	if (!pictures)
		return;
	pictures = false;
	if (after == After::Stay && pictureMode)
	{
		// the last picture into NextUI's page too, so that it stays up through the return to NextUI's
		// mode and page as Deadstation ends
		const uint32_t from = pictureRow + (picturePage ^ 1) * var.yres;
		const size_t bytes = (size_t)nativeLine * var.yres;
		if (mapped != nullptr)
		{
			memmove(mapped + (size_t)nativeLine * var.yoffset, mapped + (size_t)nativeLine * from, bytes);
			__asm__ volatile("dsb st" ::: "memory");
		}
	}
	if (after == After::Stay || !pictureMode)
		return;
	pictureMode = false;
	const int page = shownPage;
	if (after == After::NextFrame || page < 0)
	{
		// the pictures stay up until the next frame (the first, if none yet)
		modeAtFrame = true;
		return;
	}
	frameMode(page);
	pan(page);
}

// Deadstation's mode (from NextUI's), at page
void Panel::frameMode(uint32_t page)
{
	fb_var_screeninfo v = mode;
	v.xoffset = 0;
	v.yoffset = firstRow + page * mode.yres;
	fb_fix_screeninfo fix{};
	if ((mode.xres != var.xres || mode.yres != var.yres)
			&& (ioctl(fd, FBIOPUT_VSCREENINFO, &v) != 0 || ioctl(fd, FBIOGET_FSCREENINFO, &fix) != 0 || fix.line_length != lineLength))
		DS_LOG("panel: fb0 not in Deadstation's mode");
}

void Panel::restoreMode()
{
	if (mode.xres != var.xres || mode.yres != var.yres)
	{
		fb_var_screeninfo v = var;
		ioctl(fd, FBIOPUT_VSCREENINFO, &v);
		mode = var;
	}
	fb_var_screeninfo v = var;
	ioctl(fd, FBIOPAN_DISPLAY, &v);
}

void Panel::pan(uint32_t page)
{
	fb_var_screeninfo v = mode;
	v.xoffset = 0;
	v.yoffset = firstRow + page * mode.yres;
	if (ioctl(fd, FBIOPAN_DISPLAY, &v) != 0)
	{
		static bool told;
		if (!told)
			DS_LOG("panel: FBIOPAN_DISPLAY failed: %s", strerror(errno));
		told = true;
	}
}
