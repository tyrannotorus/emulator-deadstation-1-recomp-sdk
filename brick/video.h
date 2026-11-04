// The core's frames to the screen through Vulkan: each frame is uploaded into a texture and drawn,
// filtered, over a whole page of fb0 (panel.h), 4:3 as the PlayStation's picture is.
#pragma once
#include "panel.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Something drawn with each frame (a recomp module's 3D layer: recomp/brick/scene_light.h): its own passes
// before the page's, then its drawing over the frame, in the page's commands.
class VideoLayer
{
public:
	struct Target
	{
		vk::PhysicalDevice physicalDevice;
		vk::Device device;
		vk::RenderPass renderPass;				// the page's: colour attachment 0, depth attachment 1
		vk::DescriptorSetLayout frameLayout;	// set 0 at drawing: the frame (binding 0, combined image sampler)
		vk::Extent2D extent;
		uint32_t pages;
	};
	struct Frame
	{
		float uvScale[2];	// the frame's part of its texture: uvOffset + screen position (0..1) * uvScale
		float uvMax[2];
		float uvOffset[2];
		float uvMin[2];
	};
	virtual ~VideoLayer() = default;
	virtual void prepare(vk::CommandBuffer cmd, uint32_t page) = 0;
	virtual void draw(vk::CommandBuffer cmd, uint32_t page, vk::DescriptorSet frame, const Frame& f) = 0;
};

class Video
{
public:
	enum class Format { RGB565, XRGB8888 };

	~Video() { term(); }
	// native: fb0 at the panel's size (1024 x 768); smooth: the frame filtered as it is scaled; the largest
	// frame it takes (the emulator's: 1024 x 512; a recomp module's at 2x: up to 1280 x 1024)
	// layered: pages with a depth attachment, for a layer (setLayer)
	bool init(bool native, bool smooth, std::string& error, uint32_t maxWidth = 1024, uint32_t maxHeight = 512,
			bool layered = false, uint32_t smallWidth = 640);
	void term();
	bool ready() const { return isReady; }
	Panel& panel() { return screen; }
	// what a layer draws into (init's layered), and the layer (nullptr: none; it outlives its use here)
	VideoLayer::Target layerTarget() const;
	void setLayer(VideoLayer *l) { layer = l; }

	// a frame from the core: width x height pixels, pitch bytes per row
	void frame(const void *data, unsigned width, unsigned height, size_t pitch, Format format);
	// the last frame again (text changed while the game repeats its frame)
	void again();
	// the device, for a recomp module's GPU (recomp/brick/psx_gpu.h): it submits its work to the same queue before
	// each frameImage
	struct Device
	{
		vk::PhysicalDevice physicalDevice;
		vk::Device device;
		vk::Queue queue;
		uint32_t queueFamily;
	};
	Device gpuDevice() const { return { physicalDevice, *device, queue, queueFamily }; }
	// a frame straight from a GPU image (in eGeneral, its writes visible to fragment shaders): the region x, y,
	// width x height of that imageWidth x imageHeight image. before: the GPU's own commands (ended), submitted with the
	// frame's in one submission; the fence returned signals when both are done.
	vk::Fence frameImage(vk::ImageView view, uint32_t imageWidth, uint32_t imageHeight, uint32_t x, uint32_t y, uint32_t width,
			uint32_t height, vk::CommandBuffer before = nullptr);
	// A frame a module's GPU draws straight into the next page, for no copy of it: the page's number, where the frame is the page's
	// size and nothing of ours goes over it, or -1. Draw it in eGeneral (its format, Panel::format()) in the commands given to framePage.
	int pageForFrame(uint32_t width, uint32_t height) const;
	vk::ImageView pageView(uint32_t page) const { return screen.view(page); }
	vk::Image pageImage(uint32_t page) const { return screen.image(page); }
	vk::Fence framePage(vk::CommandBuffer before);
	// the page of the last frame given (its fence is the panel's to wait on: Panel::waitDrawn)
	uint32_t lastPage() const { return lastPageGiven; }
	// the last frame as RGB24 (the menu's backdrop, a save state's picture)
	bool lastFrame(std::vector<uint8_t>& rgb, int& width, int& height) const;
	// seconds spent waiting in frame() for a page (the panel, the GPU) since the last call
	double takeWaited() { const double w = waited; waited = 0; return w; }

	// Submissions on a thread of their own (a recomp module's host; before init): the driver's work for a submission (its
	// kernel calls) off the caller's thread. Every submission to the queue then goes through submit, in order.
	void setAsyncSubmit(bool on) { asyncSubmit = on; }
	// cmds (ended) to the queue, fence signalled when done (reset by the caller); page >= 0: then shown (the panel)
	void submit(const vk::CommandBuffer *cmds, uint32_t count, vk::Fence fence, int page = -1);
	// every submission made (none waiting for the thread)
	void submitDrain();
	// seconds spent in the driver's submit since the last call (on the submit thread when async)
	double takeSubmitting();
	// the GPU's seconds since the last call (DEADSTATION_PERF): the layer's own passes, the rest of the pages' commands
	// (one page in TimedEvery timed, the sums scaled to all)
	void takeGpu(double& layerSeconds, double& pageSeconds)
	{
		layerSeconds = gpuLayer * TimedEvery, pageSeconds = gpuPage * TimedEvery;
		gpuLayer = gpuPage = 0;
	}
	// the device's nanoseconds a timestamp tick, 0: no timestamps
	double timestampNs() const { return timestampPeriod; }

private:
	uint32_t TexWidth = 1024, TexHeight = 512;
	struct Page
	{
		vk::UniqueBuffer staging;
		vk::UniqueDeviceMemory stagingMemory;
		uint8_t *mapped = nullptr;
		vk::UniqueImage texture;
		vk::UniqueDeviceMemory textureMemory;
		vk::UniqueImageView textureView;
		bool textureReady = false;		// in eShaderReadOnlyOptimal since a frame drew it
		vk::DescriptorSet set;
		vk::DescriptorSet externalSet;	// frameImage's image
		VkImageView externalView = VK_NULL_HANDLE;
		vk::UniqueCommandBuffer commands;
		vk::UniqueFence fence;
		vk::UniqueFramebuffer framebuffer;
		vk::UniqueImage depth;
		vk::UniqueDeviceMemory depthMemory;
		vk::UniqueImageView depthView;
		bool timed = false;		// its timestamps written (gpuTimes)
	};
	// the GPU's time for each page's commands: begin, after the layer's own passes, end (DEADSTATION_PERF)
	vk::UniqueQueryPool timestamps;
	// timestamps written for one page in TimedEvery: the PowerVR driver keeps kernel memory for every timestamp written
	// until the process ends (about 2 MB a second when every page and pass is timed)
	static constexpr unsigned TimedEvery = 8;
	unsigned pagesSeen = 0;
	double timestampPeriod = 0;		// ns a tick
	double gpuLayer = 0, gpuPage = 0;	// seconds since takeGpu

	uint32_t memoryType(uint32_t bits, vk::MemoryPropertyFlags flags) const;
	void makeTextures(Format format);
	vk::UniquePipeline makePipeline(const uint32_t *vert, size_t vertSize, const uint32_t *frag, size_t fragSize,
			vk::PipelineLayout layout, vk::PrimitiveTopology topology);
	struct External
	{
		vk::ImageView view;
		uint32_t imageWidth, imageHeight, x, y;
		bool drawnThere = false;	// in the page already, by the commands given with it (framePage)
	};
	vk::Fence draw(unsigned width, unsigned height, Format format, bool upload, const External *external = nullptr,
			vk::CommandBuffer before = nullptr);

	vk::UniqueInstance instance;
	vk::PhysicalDevice physicalDevice;
	vk::UniqueDevice device;
	vk::Queue queue;
	uint32_t queueFamily = 0;
	Panel screen;
	vk::UniqueRenderPass renderPass;
	vk::UniqueDescriptorSetLayout setLayout;
	vk::UniquePipelineLayout pipelineLayout;
	vk::UniquePipeline pipeline;
	vk::UniquePipeline exactPipeline;		// a frame already the page's size
	vk::UniqueSampler sampler;
	vk::UniqueDescriptorPool descriptorPool;
	vk::UniqueCommandPool commandPool;
	Page pages[Panel::Pages];
	uint32_t next = 0;
	Format textureFormat = Format::RGB565;
	bool texturesMade = false;
	bool isReady = false;
	bool layered = false;
	vk::Format depthFormat = vk::Format::eUndefined;
	VideoLayer *layer = nullptr;

	// the last frame, as the core gave it
	std::vector<uint8_t> last;
	unsigned lastWidth = 0, lastHeight = 0;
	Format lastFormat = Format::RGB565;
	double waited = 0;
	uint32_t lastPageGiven = 0;
	// the submit thread (setAsyncSubmit)
	struct SubmitJob
	{
		vk::CommandBuffer cmds[2];
		uint32_t count;
		vk::Fence fence;
		int page;
	};
	bool asyncSubmit = false;
	std::thread submitThread;
	std::mutex submitMutex;
	std::condition_variable submitChanged;
	std::deque<SubmitJob> submitJobs;
	bool submitStop = false;
	double submitting = 0;
	void submitNow(const SubmitJob& job);
	void submitLoop();
};
