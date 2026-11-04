#include "video.h"
#include "log.h"

#include "frame_frag.h"
#include "frame_exact_frag.h"
#include "frame_vert.h"

#include <algorithm>
#include <chrono>
#include <cstring>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
struct FramePush
{
	float uvOffset[2];
	float uvScale[2];
	float uvMin[2];
	float uvMax[2];
};
}

uint32_t Video::memoryType(uint32_t bits, vk::MemoryPropertyFlags flags) const
{
	const vk::PhysicalDeviceMemoryProperties props = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < props.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
			return i;
	throw std::runtime_error("no memory type");
}

vk::UniquePipeline Video::makePipeline(const uint32_t *vert, size_t vertSize, const uint32_t *frag, size_t fragSize,
		vk::PipelineLayout layout, vk::PrimitiveTopology topology)
{
	const vk::Extent2D extent = screen.extent();
	vk::UniqueShaderModule v = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), vertSize, vert));
	vk::UniqueShaderModule f = device->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), fragSize, frag));
	vk::PipelineShaderStageCreateInfo stages[] = {
		{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, *v, "main" },
		{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, *f, "main" },
	};
	vk::PipelineVertexInputStateCreateInfo vertexInput;
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly(vk::PipelineInputAssemblyStateCreateFlags(), topology);
	vk::Viewport viewport(0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f);
	vk::Rect2D scissor({ 0, 0 }, extent);
	vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(), viewport, scissor);
	vk::PipelineRasterizationStateCreateInfo raster(vk::PipelineRasterizationStateCreateFlags(), false, false, vk::PolygonMode::eFill,
			vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);
	vk::PipelineMultisampleStateCreateInfo multisample;
	vk::PipelineColorBlendAttachmentState blendAttachment;
	blendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB
			| vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend(vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eCopy, blendAttachment);
	vk::PipelineDepthStencilStateCreateInfo depth;	// off
	vk::GraphicsPipelineCreateInfo info(vk::PipelineCreateFlags(), stages, &vertexInput, &inputAssembly, nullptr, &viewportState,
			&raster, &multisample, layered ? &depth : nullptr, &blend, nullptr, layout, *renderPass, 0);
	return device->createGraphicsPipelineUnique(nullptr, info).value;
}

bool Video::init(bool native, bool smooth, std::string& error, uint32_t maxWidth, uint32_t maxHeight, bool layered, uint32_t smallWidth)
{
	TexWidth = maxWidth;
	TexHeight = maxHeight;
	this->layered = layered;
	try
	{
		static vk::detail::DynamicLoader loader;
		auto getInstanceProcAddr = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (getInstanceProcAddr == nullptr)
		{
			error = "no vkGetInstanceProcAddr";
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);
		vk::ApplicationInfo app("Deadstation", 1, "Deadstation", 1, VK_API_VERSION_1_1);
		instance = vk::createInstanceUnique(vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app));
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
		const auto devices = instance->enumeratePhysicalDevices();
		if (devices.empty())
		{
			error = "no Vulkan device";
			return false;
		}
		physicalDevice = devices.front();
		const vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
		DS_LOG("Vulkan %d.%d.%d, %s", VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
				VK_API_VERSION_PATCH(props.apiVersion), props.deviceName.data());

		const auto families = physicalDevice.getQueueFamilyProperties();
		queueFamily = (uint32_t)(std::find_if(families.begin(), families.end(),
				[](const vk::QueueFamilyProperties& f) { return bool(f.queueFlags & vk::QueueFlagBits::eGraphics); }) - families.begin());
		if (queueFamily == families.size())
		{
			error = "no graphics queue";
			return false;
		}
		const float priority = 1.f;
		vk::DeviceQueueCreateInfo queueInfo(vk::DeviceQueueCreateFlags(), queueFamily, 1, &priority);
		const char *extensions[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME };
		device = physicalDevice.createDeviceUnique(vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queueInfo, nullptr, extensions));
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
		queue = device->getQueue(queueFamily, 0);

		if (!screen.init(*device, native, error, smallWidth))
			return false;
		const vk::Extent2D extent = screen.extent();

		// into a page, written out for the display engine to read once the fence has signalled
		vk::AttachmentDescription attachment(vk::AttachmentDescriptionFlags(), Panel::format(), vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
		vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
		vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, colorRef);
		vk::SubpassDependency dependency(0, vk::SubpassExternal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eHost,
				vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eHostRead);
		if (layered)
		{
			// a depth attachment for the layer (with a stencil, where it casts shadows), cleared each frame and never kept
			for (vk::Format f : { vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint, vk::Format::eD16Unorm, vk::Format::eX8D24UnormPack32,
						vk::Format::eD32Sfloat })
				if (physicalDevice.getFormatProperties(f).optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
				{
					depthFormat = f;
					break;
				}
			if (depthFormat == vk::Format::eUndefined)
			{
				error = "no depth format";
				return false;
			}
			vk::AttachmentDescription attachments[2] = { attachment,
				{ vk::AttachmentDescriptionFlags(), depthFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
					vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
					vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal } };
			vk::AttachmentReference depthRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
			vk::SubpassDescription withDepth(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, colorRef, nullptr,
					&depthRef);
			renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachments, withDepth,
					dependency));
		}
		else
			renderPass = device->createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachment, subpass, dependency));

		vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
		setLayout = device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), binding));
		vk::PushConstantRange framePush(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(FramePush));
		pipelineLayout = device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *setLayout, framePush));
		pipeline = makePipeline(frame_vert, sizeof(frame_vert), frame_frag, sizeof(frame_frag), *pipelineLayout, vk::PrimitiveTopology::eTriangleList);
		exactPipeline = makePipeline(frame_vert, sizeof(frame_vert), frame_exact_frag, sizeof(frame_exact_frag), *pipelineLayout, vk::PrimitiveTopology::eTriangleList);

		const vk::Filter filter = smooth ? vk::Filter::eLinear : vk::Filter::eNearest;
		sampler = device->createSamplerUnique(vk::SamplerCreateInfo(vk::SamplerCreateFlags(), filter, filter,
				vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
				vk::SamplerAddressMode::eClampToEdge));
		vk::DescriptorPoolSize poolSize(vk::DescriptorType::eCombinedImageSampler, 2 * Panel::Pages);
		descriptorPool = device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 2 * Panel::Pages, poolSize));
		commandPool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily));
		if (getenv("DEADSTATION_PERF") != nullptr && physicalDevice.getQueueFamilyProperties()[queueFamily].timestampValidBits > 0)
		{
			timestampPeriod = physicalDevice.getProperties().limits.timestampPeriod;
			timestamps = device->createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp,
					Panel::Pages * 3));
		}

		std::vector<vk::DescriptorSetLayout> layouts(2 * Panel::Pages, *setLayout);
		const auto sets = device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(*descriptorPool, layouts));
		auto commands = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, Panel::Pages));
		const vk::DeviceSize stagingSize = (vk::DeviceSize)TexWidth * TexHeight * 4;
		for (uint32_t i = 0; i < Panel::Pages; i++)
		{
			Page& p = pages[i];
			p.set = sets[i];
			p.externalSet = sets[Panel::Pages + i];
			p.externalView = VK_NULL_HANDLE;
			p.commands = std::move(commands[i]);
			p.fence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
			vk::ImageView view = screen.view(i);
			if (layered)
			{
				p.depth = device->createImageUnique(vk::ImageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, depthFormat,
						vk::Extent3D(extent.width, extent.height, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
						vk::ImageUsageFlagBits::eDepthStencilAttachment));
				const vk::MemoryRequirements dreq = device->getImageMemoryRequirements(*p.depth);
				p.depthMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(dreq.size,
						memoryType(dreq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)));
				device->bindImageMemory(*p.depth, *p.depthMemory, 0);
				p.depthView = device->createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), *p.depth, vk::ImageViewType::e2D,
						depthFormat, vk::ComponentMapping(), vk::ImageSubresourceRange(depthFormat == vk::Format::eD24UnormS8Uint ||
						depthFormat == vk::Format::eD32SfloatS8Uint ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil :
						vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)));
				vk::ImageView views[2] = { view, *p.depthView };
				p.framebuffer = device->createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *renderPass, views,
						extent.width, extent.height, 1));
			}
			else
				p.framebuffer = device->createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *renderPass, view,
						extent.width, extent.height, 1));
			p.staging = device->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), stagingSize, vk::BufferUsageFlagBits::eTransferSrc));
			const vk::MemoryRequirements req = device->getBufferMemoryRequirements(*p.staging);
			p.stagingMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(req.size,
					memoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)));
			device->bindBufferMemory(*p.staging, *p.stagingMemory, 0);
			p.mapped = (uint8_t *)device->mapMemory(*p.stagingMemory, 0, stagingSize);
		}
		texturesMade = false;
		makeTextures(lastFormat);
		next = 0;
		isReady = true;
		return true;
	}
	catch (const std::exception& e)
	{
		error = std::string("Vulkan: ") + e.what();
		return false;
	}
}

// the pages' textures in the core's pixel format (made again if it changes)
void Video::makeTextures(Format format)
{
	if (texturesMade)
		device->waitIdle();
	const vk::Format vkFormat = format == Format::RGB565 ? vk::Format::eR5G6B5UnormPack16 : vk::Format::eB8G8R8A8Unorm;
	for (Page& p : pages)
	{
		p.textureView.reset();
		p.texture.reset();
		p.textureMemory.reset();
		p.texture = device->createImageUnique(vk::ImageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, vkFormat,
				vk::Extent3D(TexWidth, TexHeight, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
		const vk::MemoryRequirements req = device->getImageMemoryRequirements(*p.texture);
		p.textureMemory = device->allocateMemoryUnique(vk::MemoryAllocateInfo(req.size,
				memoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)));
		device->bindImageMemory(*p.texture, *p.textureMemory, 0);
		p.textureView = device->createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), *p.texture, vk::ImageViewType::e2D,
				vkFormat, vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
		p.textureReady = false;
		vk::DescriptorImageInfo imageInfo(*sampler, *p.textureView, vk::ImageLayout::eShaderReadOnlyOptimal);
		device->updateDescriptorSets(vk::WriteDescriptorSet(p.set, 0, 0, vk::DescriptorType::eCombinedImageSampler, imageInfo), nullptr);
	}
	textureFormat = format;
	texturesMade = true;
}

void Video::frame(const void *data, unsigned width, unsigned height, size_t pitch, Format format)
{
	if (!isReady || data == nullptr || width == 0 || height == 0)
		return;
	width = std::min(width, TexWidth);
	height = std::min(height, TexHeight);
	// kept, tightly packed: the menu's backdrop, a save state's picture, the frame again
	const size_t bpp = format == Format::RGB565 ? 2 : 4;
	const size_t row = width * bpp;
	last.resize(row * height);
	for (unsigned y = 0; y < height; y++)
		memcpy(last.data() + y * row, (const uint8_t *)data + y * pitch, row);
	lastWidth = width;
	lastHeight = height;
	lastFormat = format;
	draw(width, height, format, true);
}

void Video::again()
{
	if (isReady && !last.empty())
		draw(lastWidth, lastHeight, lastFormat, true);
}

void Video::submitNow(const SubmitJob& job)
{
	const auto from = std::chrono::steady_clock::now();
	queue.submit(vk::SubmitInfo(nullptr, nullptr, vk::ArrayProxyNoTemporaries<const vk::CommandBuffer>(job.count, job.cmds)), job.fence);
	const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
	if (took > 0.05)
		DS_LOG("video: a submission took %.0f ms", took * 1000.0);
	if (job.page >= 0)
		screen.queue((uint32_t)job.page, job.fence);
	std::lock_guard<std::mutex> lock(submitMutex);
	submitting += took;
}

void Video::submitLoop()
{
	for (;;)
	{
		SubmitJob job;
		{
			std::unique_lock<std::mutex> lock(submitMutex);
			submitChanged.wait(lock, [this] { return submitStop || !submitJobs.empty(); });
			if (submitJobs.empty())
				return;
			job = submitJobs.front();
		}
		submitNow(job);
		{
			std::lock_guard<std::mutex> lock(submitMutex);
			submitJobs.pop_front();
		}
		submitChanged.notify_all();
	}
}

void Video::submit(const vk::CommandBuffer *cmds, uint32_t count, vk::Fence fence, int page)
{
	SubmitJob job = { { cmds[0], count > 1 ? cmds[1] : vk::CommandBuffer() }, count, fence, page };
	if (!asyncSubmit)
	{
		submitNow(job);
		return;
	}
	{
		std::lock_guard<std::mutex> lock(submitMutex);
		if (!submitThread.joinable())
		{
			submitStop = false;
			submitThread = std::thread([this] { submitLoop(); });
		}
		submitJobs.push_back(job);
	}
	submitChanged.notify_all();
}

void Video::submitDrain()
{
	std::unique_lock<std::mutex> lock(submitMutex);
	submitChanged.wait(lock, [this] { return submitJobs.empty(); });
}

double Video::takeSubmitting()
{
	std::lock_guard<std::mutex> lock(submitMutex);
	const double s = submitting;
	submitting = 0;
	return s;
}

vk::Fence Video::frameImage(vk::ImageView view, uint32_t imageWidth, uint32_t imageHeight, uint32_t x, uint32_t y,
		uint32_t width, uint32_t height, vk::CommandBuffer before)
{
	if (!isReady)
		return nullptr;
	const External e = { view, imageWidth, imageHeight, x, y };
	return draw(width, height, textureFormat, false, &e, before);
}

int Video::pageForFrame(uint32_t width, uint32_t height) const
{
	const vk::Extent2D extent = screen.extent();
	if (!isReady || layer != nullptr || width != extent.width || height != extent.height)
		return -1;
	return (int)next;
}

vk::Fence Video::framePage(vk::CommandBuffer before)
{
	if (!isReady)
		return nullptr;
	const vk::Extent2D extent = screen.extent();
	External e {};
	e.imageWidth = extent.width, e.imageHeight = extent.height, e.drawnThere = true;
	return draw(extent.width, extent.height, textureFormat, false, &e, before);
}

vk::Fence Video::draw(unsigned width, unsigned height, Format format, bool upload, const External *external,
		vk::CommandBuffer before)
{
	if (external == nullptr && format != textureFormat)
		makeTextures(format);
	const uint32_t i = next;
	next = (next + 1) % Panel::Pages;
	lastPageGiven = i;
	Page& p = pages[i];
	const auto waitFrom = std::chrono::steady_clock::now();
	// the frames handed to the submit thread first (the panel counts only those queued to it)
	submitDrain();
	screen.waitWritable(i);
	screen.waitDrawn(i);
	waited += std::chrono::duration<double>(std::chrono::steady_clock::now() - waitFrom).count();
	if (p.timed)
	{
		uint64_t t[3];
		if (device->getQueryPoolResults(*timestamps, i * 3, 3, sizeof(t), t, sizeof(uint64_t), vk::QueryResultFlagBits::e64) ==
				vk::Result::eSuccess)
		{
			gpuLayer += (double)(t[1] - t[0]) * timestampPeriod * 1e-9;
			gpuPage += (double)(t[2] - t[1]) * timestampPeriod * 1e-9;
		}
		p.timed = false;
	}
	device->resetFences(*p.fence);

	const size_t bpp = format == Format::RGB565 ? 2 : 4;
	if (upload && external == nullptr)
		memcpy(p.mapped, last.data(), (size_t)width * height * bpp);

	vk::CommandBuffer cmd = *p.commands;
	cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	const bool timed = timestamps && pagesSeen++ % TimedEvery == 0;
	if (timed)
	{
		cmd.resetQueryPool(*timestamps, i * 3, 3);
		// bottom of the pipe: after the work submitted before it (the module's commands), not while it runs
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps, i * 3);
	}
	const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
	if (external != nullptr && external->drawnThere)
	{
		// (the module's pass has drawn the page: nothing but its writes out to memory, for the panel)
		const vk::MemoryBarrier written(vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags(), written, nullptr, nullptr);
		if (timed)
		{
			cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps, i * 3 + 1);
			cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps, i * 3 + 2);
			p.timed = true;
		}
		cmd.end();
		if (before)
		{
			const vk::CommandBuffer both[2] = { before, cmd };
			submit(both, 2, *p.fence, (int)i);
		}
		else
			submit(&cmd, 1, *p.fence, (int)i);
		return *p.fence;
	}
	const auto toTransfer = [&](vk::Image image, bool ready) {
		vk::ImageMemoryBarrier b(ready ? vk::AccessFlagBits::eShaderRead : vk::AccessFlags(), vk::AccessFlagBits::eTransferWrite,
				ready ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
				vk::DependencyFlags(), nullptr, nullptr, b);
	};
	const auto toShader = [&](vk::Image image) {
		vk::ImageMemoryBarrier b(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(),
				nullptr, nullptr, b);
	};
	vk::DescriptorSet set = p.set;
	FramePush fp = { { 0, 0 }, { (float)width / TexWidth, (float)height / TexHeight }, { 0, 0 },
			{ (width - 0.5f) / TexWidth, (height - 0.5f) / TexHeight } };
	if (external == nullptr)
	{
		toTransfer(*p.texture, p.textureReady);
		vk::BufferImageCopy copy(0, width, height, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { 0, 0, 0 }, { width, height, 1 });
		cmd.copyBufferToImage(*p.staging, *p.texture, vk::ImageLayout::eTransferDstOptimal, copy);
		toShader(*p.texture);
		p.textureReady = true;
	}
	else
	{
		if (p.externalView != (VkImageView)external->view)
		{
			vk::DescriptorImageInfo info(*sampler, external->view, vk::ImageLayout::eGeneral);
			device->updateDescriptorSets(vk::WriteDescriptorSet(p.externalSet, 0, 0, vk::DescriptorType::eCombinedImageSampler, info), nullptr);
			p.externalView = external->view;
		}
		set = p.externalSet;
		const float iw = (float)external->imageWidth, ih = (float)external->imageHeight;
		fp = { { external->x / iw, external->y / ih }, { width / iw, height / ih }, { (external->x + 0.5f) / iw, (external->y + 0.5f) / ih },
				{ (external->x + width - 0.5f) / iw, (external->y + height - 0.5f) / ih } };
	}

	const vk::Extent2D extent = screen.extent();
	if (layer != nullptr)
		layer->prepare(cmd, i);
	if (timed)
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps, i * 3 + 1);
	const vk::ClearValue clears[2] = { vk::ClearColorValue(), vk::ClearDepthStencilValue(1.f, 0) };
	cmd.beginRenderPass(vk::RenderPassBeginInfo(*renderPass, *p.framebuffer, vk::Rect2D({ 0, 0 }, extent), layered ? 2u : 0u, clears),
			vk::SubpassContents::eInline);
	// (an image the page's own size, the whole of it: pixel for pixel, 2-3 ms of this GPU's on every frame otherwise)
	const bool exact = external != nullptr && external->x == 0 && external->y == 0 && width == external->imageWidth && height == external->imageHeight && width == extent.width && height == extent.height;
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, exact ? *exactPipeline : *pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, set, nullptr);
	cmd.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(fp), &fp);
	cmd.draw(3, 1, 0, 0);
	if (layer != nullptr)
		layer->draw(cmd, i, set, { { fp.uvScale[0], fp.uvScale[1] }, { fp.uvMax[0], fp.uvMax[1] }, { fp.uvOffset[0], fp.uvOffset[1] },
				{ fp.uvMin[0], fp.uvMin[1] } });
	cmd.endRenderPass();
	if (timed)
	{
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamps, i * 3 + 2);
		p.timed = true;
	}
	cmd.end();
	if (before)
	{
		const vk::CommandBuffer both[2] = { before, cmd };
		submit(both, 2, *p.fence, (int)i);
	}
	else
		submit(&cmd, 1, *p.fence, (int)i);
	return *p.fence;
}

VideoLayer::Target Video::layerTarget() const
{
	return { physicalDevice, *device, *renderPass, *setLayout, screen.extent(), Panel::Pages };
}

bool Video::lastFrame(std::vector<uint8_t>& rgb, int& width, int& height) const
{
	if (last.empty())
		return false;
	width = (int)lastWidth;
	height = (int)lastHeight;
	rgb.resize((size_t)width * height * 3);
	uint8_t *out = rgb.data();
	if (lastFormat == Format::RGB565)
	{
		const uint16_t *in = (const uint16_t *)last.data();
		for (size_t i = 0; i < (size_t)width * height; i++, out += 3)
		{
			const uint16_t v = in[i];
			out[0] = (uint8_t)(((v >> 11) & 31) * 255 / 31);
			out[1] = (uint8_t)(((v >> 5) & 63) * 255 / 63);
			out[2] = (uint8_t)((v & 31) * 255 / 31);
		}
	}
	else
	{
		const uint32_t *in = (const uint32_t *)last.data();
		for (size_t i = 0; i < (size_t)width * height; i++, out += 3)
		{
			out[0] = (uint8_t)(in[i] >> 16);
			out[1] = (uint8_t)(in[i] >> 8);
			out[2] = (uint8_t)in[i];
		}
	}
	return true;
}

void Video::term()
{
	if (!device)
		return;
	isReady = false;
	if (submitThread.joinable())
	{
		{
			std::lock_guard<std::mutex> lock(submitMutex);
			submitStop = true;
		}
		submitChanged.notify_all();
		submitThread.join();
	}
	screen.drain();
	device->waitIdle();
	screen.term();
	for (Page& p : pages)
	{
		if (p.mapped != nullptr)
			device->unmapMemory(*p.stagingMemory);
		p = Page();
	}
	timestamps.reset();
	commandPool.reset();
	descriptorPool.reset();
	sampler.reset();
	exactPipeline.reset();
	pipeline.reset();
	pipelineLayout.reset();
	setLayout.reset();
	renderPass.reset();
	device.reset();
	instance.reset();
}
