#include "scene_light.h"

#include "scene_cast_vert.h"
#include "scene_flash_floor_frag.h"
#include "scene_flash_vert.h"
#include "scene_light_frag.h"
#include "scene_light_vert.h"
#include "scene_shadow_vert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace
{
using deadstation::scene::Corner;

// scene_params.glsl's, std140
struct Params
{
	float axisU[4], axisV[4], axisS[4], scale[4], screen[4], frameUv[4], light[4];
	float flashRight[4], flashUp[4], flashAhead[4], flashAt[4];
	float lightAt[6][4], lightColour[6][4];
	float options[4];	// x: the models' sun shadows on themselves
	float frameAt[4];	// xy: the frame's uv offset; zw: its smallest uv
	float floorPlane[4], project[4];
};

constexpr float Bias = 40.f;	// the shadow map's slack, in view units (the game's depths come in steps of 32)
}

uint32_t SceneLight::memoryType(uint32_t bits, vk::MemoryPropertyFlags flags) const
{
	const vk::PhysicalDeviceMemoryProperties props = target.physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < props.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
			return i;
	throw std::runtime_error("no memory type");
}

bool SceneLight::init(const Target& t, std::string& error)
{
	target = t;
	vk::Device device = target.device;
	try
	{
		const vk::FormatFeatureFlags needs = vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage;
		for (vk::Format f : { vk::Format::eD16Unorm, vk::Format::eD32Sfloat, vk::Format::eX8D24UnormPack32 })
			if ((target.physicalDevice.getFormatProperties(f).optimalTilingFeatures & needs) == needs)
			{
				shadowFormat = f;
				break;
			}
		if (shadowFormat == vk::Format::eUndefined)
		{
			error = "no sampled depth format";
			return false;
		}

		// the shadow map: depth only, cleared, then read by the drawing's fragments
		vk::AttachmentDescription attachment(vk::AttachmentDescriptionFlags(), shadowFormat, vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
		vk::AttachmentReference depthRef(0, vk::ImageLayout::eDepthStencilAttachmentOptimal);
		vk::SubpassDescription subpass(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, nullptr, nullptr, nullptr, &depthRef);
		const vk::PipelineStageFlags tests = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
		vk::SubpassDependency dependencies[2] = {
			{ vk::SubpassExternal, 0, vk::PipelineStageFlagBits::eFragmentShader, tests, vk::AccessFlagBits::eShaderRead,
				vk::AccessFlagBits::eDepthStencilAttachmentWrite },
			{ 0, vk::SubpassExternal, tests, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eDepthStencilAttachmentWrite,
				vk::AccessFlagBits::eShaderRead },
		};
		shadowPass = device.createRenderPassUnique(vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachment, subpass, dependencies));
		// outside the map: lit (depth 1, the far end)
		// compared in the sampler (lit where the pixel's depth is at most the map's), 2 x 2 filtered by the hardware when
		// the format can be (one fetch for the 2 x 2 test the shader made in four)
		const vk::Filter filter = (target.physicalDevice.getFormatProperties(shadowFormat).optimalTilingFeatures &
				vk::FormatFeatureFlagBits::eSampledImageFilterLinear) ? vk::Filter::eLinear : vk::Filter::eNearest;
		shadowSampler = device.createSamplerUnique(vk::SamplerCreateInfo(vk::SamplerCreateFlags(), filter, filter,
				vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
				vk::SamplerAddressMode::eClampToBorder, 0.f, false, 1.f, true, vk::CompareOp::eLessOrEqual, 0.f, 0.f,
				vk::BorderColor::eFloatOpaqueWhite));

		vk::DescriptorSetLayoutBinding bindings[3] = {
			{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			{ 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
			{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
		};
		setLayout = device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), bindings));
		vk::DescriptorSetLayout sets[2] = { target.frameLayout, *setLayout };
		vk::PushConstantRange pass(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(int32_t));	// 1: the floor; or which flash
		layout = device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), sets, pass));
		vk::DescriptorPoolSize sizes[2] = { { vk::DescriptorType::eCombinedImageSampler, 2 * target.pages },
			{ vk::DescriptorType::eUniformBuffer, target.pages } };
		pool = device.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), target.pages, sizes));

		// the pipelines: both read the corners (scene.h's Corner), test and write depth, cull nothing
		vk::UniqueShaderModule shadowVert = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
				sizeof(scene_shadow_vert), scene_shadow_vert));
		vk::UniqueShaderModule flashVert = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
				sizeof(scene_flash_vert), scene_flash_vert));
		vk::UniqueShaderModule lightVert = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
				sizeof(scene_light_vert), scene_light_vert));
		vk::UniqueShaderModule lightFrag = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
				sizeof(scene_light_frag), scene_light_frag));
		vk::VertexInputBindingDescription binding(0, sizeof(Corner), vk::VertexInputRate::eVertex);
		vk::VertexInputAttributeDescription attributes[3] = {
			{ 0, 0, vk::Format::eR32G32Sfloat, offsetof(Corner, sx) },
			{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Corner, x) },
			{ 2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Corner, nx) },
		};
		vk::PipelineInputAssemblyStateCreateInfo assembly(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);
		vk::PipelineRasterizationStateCreateInfo raster(vk::PipelineRasterizationStateCreateFlags(), false, false, vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);
		vk::PipelineMultisampleStateCreateInfo multisample;
		vk::PipelineDepthStencilStateCreateInfo depth(vk::PipelineDepthStencilStateCreateFlags(), true, true, vk::CompareOp::eLess);
		vk::PipelineColorBlendAttachmentState blendAttachment;
		blendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB
				| vk::ColorComponentFlagBits::eA;

		for (int which = 0; which < 2; which++)
		{
			vk::PipelineShaderStageCreateInfo stage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
					which == 0 ? *shadowVert : *flashVert, "main");
			vk::PipelineVertexInputStateCreateInfo input(vk::PipelineVertexInputStateCreateFlags(), binding, attributes[1]);
			vk::Viewport viewport(0.f, 0.f, (float)ShadowSize, (float)ShadowSize, 0.f, 1.f);
			vk::Rect2D scissor({ 0, 0 }, { ShadowSize, ShadowSize });
			vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(), viewport, scissor);
			vk::PipelineColorBlendStateCreateInfo blend;
			vk::GraphicsPipelineCreateInfo info(vk::PipelineCreateFlags(), stage, &input, &assembly, nullptr, &viewportState, &raster,
					&multisample, &depth, &blend, nullptr, *layout, *shadowPass, 0);
			(which == 0 ? shadowPipeline : flashPipeline) = device.createGraphicsPipelineUnique(nullptr, info).value;
		}
		{
			vk::PipelineShaderStageCreateInfo stages[2] = {
				{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, *lightVert, "main" },
				{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, *lightFrag, "main" },
			};
			vk::PipelineVertexInputStateCreateInfo input(vk::PipelineVertexInputStateCreateFlags(), binding, attributes);
			vk::Viewport viewport(0.f, 0.f, (float)target.extent.width, (float)target.extent.height, 0.f, 1.f);
			vk::Rect2D scissor({ 0, 0 }, target.extent);
			vk::PipelineViewportStateCreateInfo viewportState(vk::PipelineViewportStateCreateFlags(), viewport, scissor);
			vk::PipelineColorBlendStateCreateInfo blend(vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eCopy, blendAttachment);
			vk::GraphicsPipelineCreateInfo info(vk::PipelineCreateFlags(), stages, &input, &assembly, nullptr, &viewportState, &raster,
					&multisample, &depth, &blend, nullptr, *layout, target.renderPass, 0);
			lightPipeline = device.createGraphicsPipelineUnique(nullptr, info).value;
			// the floor: tested against the models, written by nothing (everything on it lies in one plane)
			vk::PipelineDepthStencilStateCreateInfo floorDepth(vk::PipelineDepthStencilStateCreateFlags(), true, false, vk::CompareOp::eLessOrEqual);
			info.pDepthStencilState = &floorDepth;
			floorPipeline = device.createGraphicsPipelineUnique(nullptr, info).value;
			// a flash on the floor: added, outside what it casts (the stencil, its own reference)
			vk::UniqueShaderModule flashFloorFrag = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
					sizeof(scene_flash_floor_frag), scene_flash_floor_frag));
			vk::PipelineShaderStageCreateInfo addStages[2] = { stages[0],
				{ vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, *flashFloorFrag, "main" } };
			vk::PipelineColorBlendAttachmentState add(true, vk::BlendFactor::eOne, vk::BlendFactor::eOne, vk::BlendOp::eAdd, vk::BlendFactor::eZero,
					vk::BlendFactor::eOne, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB);
			vk::PipelineColorBlendStateCreateInfo addBlend(vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eCopy, add);
			vk::StencilOpState outside(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eNotEqual, 0xFF, 0, 0);
			vk::PipelineDepthStencilStateCreateInfo addDepth(vk::PipelineDepthStencilStateCreateFlags(), true, false, vk::CompareOp::eLessOrEqual, false, true,
					outside, outside);
			const vk::DynamicState reference = vk::DynamicState::eStencilReference;
			vk::PipelineDynamicStateCreateInfo dynamic(vk::PipelineDynamicStateCreateFlags(), reference);
			info.setStages(addStages);
			info.pDepthStencilState = &addDepth;
			info.pColorBlendState = &addBlend;
			info.pDynamicState = &dynamic;
			flashFloorPipeline = device.createGraphicsPipelineUnique(nullptr, info).value;
			// the models flattened from a flash: the stencil where they land and no model stands nearer; no colour
			vk::UniqueShaderModule castVert = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
					sizeof(scene_cast_vert), scene_cast_vert));
			vk::PipelineShaderStageCreateInfo castStage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, *castVert, "main");
			vk::PipelineVertexInputStateCreateInfo castInput(vk::PipelineVertexInputStateCreateFlags(), binding, attributes[1]);
			vk::StencilOpState mark(vk::StencilOp::eKeep, vk::StencilOp::eReplace, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0xFF, 0xFF, 0);
			vk::PipelineDepthStencilStateCreateInfo castDepth(vk::PipelineDepthStencilStateCreateFlags(), true, false, vk::CompareOp::eLessOrEqual, false, true,
					mark, mark);
			vk::PipelineColorBlendAttachmentState noColour;
			noColour.colorWriteMask = vk::ColorComponentFlags();
			vk::PipelineColorBlendStateCreateInfo castBlend(vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eCopy, noColour);
			info.setStages(castStage);
			info.pVertexInputState = &castInput;
			info.pDepthStencilState = &castDepth;
			info.pColorBlendState = &castBlend;
			castPipeline = device.createGraphicsPipelineUnique(nullptr, info).value;
		}

		// per page: the corners, the parameters, the shadow map and its framebuffer, their descriptor set
		std::vector<vk::DescriptorSetLayout> layouts(target.pages, *setLayout);
		const auto descriptorSets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(*pool, layouts));
		slots.resize(target.pages);
		const vk::MemoryPropertyFlags host = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
		for (uint32_t i = 0; i < target.pages; i++)
		{
			Slot& s = slots[i];
			s.vertices = device.createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), MaxCorners * sizeof(Corner),
					vk::BufferUsageFlagBits::eVertexBuffer));
			vk::MemoryRequirements req = device.getBufferMemoryRequirements(*s.vertices);
			s.verticesMemory = device.allocateMemoryUnique(vk::MemoryAllocateInfo(req.size, memoryType(req.memoryTypeBits, host)));
			device.bindBufferMemory(*s.vertices, *s.verticesMemory, 0);
			s.corners = (Corner *)device.mapMemory(*s.verticesMemory, 0, MaxCorners * sizeof(Corner));

			s.params = device.createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), sizeof(Params), vk::BufferUsageFlagBits::eUniformBuffer));
			req = device.getBufferMemoryRequirements(*s.params);
			s.paramsMemory = device.allocateMemoryUnique(vk::MemoryAllocateInfo(req.size, memoryType(req.memoryTypeBits, host)));
			device.bindBufferMemory(*s.params, *s.paramsMemory, 0);
			s.paramsMapped = device.mapMemory(*s.paramsMemory, 0, sizeof(Params));

			auto depthMap = [&](vk::UniqueImage& img, vk::UniqueDeviceMemory& mem, vk::UniqueImageView& view, vk::UniqueFramebuffer& fb) {
				img = device.createImageUnique(vk::ImageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, shadowFormat,
						vk::Extent3D(ShadowSize, ShadowSize, 1), 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
						vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled));
				const vk::MemoryRequirements r = device.getImageMemoryRequirements(*img);
				mem = device.allocateMemoryUnique(vk::MemoryAllocateInfo(r.size, memoryType(r.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)));
				device.bindImageMemory(*img, *mem, 0);
				view = device.createImageViewUnique(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), *img, vk::ImageViewType::e2D,
						shadowFormat, vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)));
				fb = device.createFramebufferUnique(vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *shadowPass, *view,
						ShadowSize, ShadowSize, 1));
			};
			depthMap(s.shadow, s.shadowMemory, s.shadowView, s.shadowFramebuffer);
			depthMap(s.flash, s.flashMemory, s.flashView, s.flashFramebuffer);

			s.set = descriptorSets[i];
			vk::DescriptorImageInfo image(*shadowSampler, *s.shadowView, vk::ImageLayout::eShaderReadOnlyOptimal);
			vk::DescriptorImageInfo flashImage(*shadowSampler, *s.flashView, vk::ImageLayout::eShaderReadOnlyOptimal);
			vk::DescriptorBufferInfo buffer(*s.params, 0, sizeof(Params));
			vk::WriteDescriptorSet writes[3] = {
				{ s.set, 0, 0, vk::DescriptorType::eCombinedImageSampler, image },
				{ s.set, 1, 0, vk::DescriptorType::eUniformBuffer, nullptr, buffer },
				{ s.set, 2, 0, vk::DescriptorType::eCombinedImageSampler, flashImage },
			};
			device.updateDescriptorSets(writes, nullptr);
		}
	}
	catch (const std::exception& e)
	{
		error = e.what();
		return false;
	}
	return true;
}

void SceneLight::prepare(vk::CommandBuffer cmd, uint32_t page)
{
	Slot& s = slots[page % slots.size()];
	s.count = s.floorCount = s.floorBase = 0;
	deadstation::scene::take(scene);
	const uint32_t count = std::min<uint32_t>((uint32_t)scene.corners.size() / 3 * 3, MaxCorners / 3 * 3);
	if (count == 0 || scene.width <= 0 || scene.height <= 0)
		return;
	std::memcpy(s.corners, scene.corners.data(), count * sizeof(Corner));
	const uint32_t floorCount = std::min<uint32_t>((uint32_t)scene.floor.size() / 3 * 3, (MaxCorners - count) / 3 * 3);
	std::memcpy(s.corners + count, scene.floor.data(), floorCount * sizeof(Corner));

	// the sun's axes: s toward it, a and b across its light; the models' extent along them
	const float *sun = scene.sun;
	float a[3] = { sun[1], -sun[0], 0.f };
	if (std::fabs(sun[2]) > 0.9f)
		a[0] = 0.f, a[1] = sun[2], a[2] = -sun[1];
	const float an = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	for (float& c : a)
		c /= an;
	const float b[3] = { sun[1] * a[2] - sun[2] * a[1], sun[2] * a[0] - sun[0] * a[2], sun[0] * a[1] - sun[1] * a[0] };
	float u0 = 1e30f, u1 = -1e30f, v0 = 1e30f, v1 = -1e30f, d0 = 1e30f, d1 = -1e30f, zFar = 1.f;
	for (uint32_t i = 0; i < count; i++)
	{
		const Corner& c = scene.corners[i];
		const float u = c.x * a[0] + c.y * a[1] + c.z * a[2], v = c.x * b[0] + c.y * b[1] + c.z * b[2];
		const float d = c.x * sun[0] + c.y * sun[1] + c.z * sun[2];
		u0 = std::min(u0, u), u1 = std::max(u1, u), v0 = std::min(v0, v), v1 = std::max(v1, v);
		d0 = std::min(d0, d), d1 = std::max(d1, d), zFar = std::max(zFar, c.z);
	}
	const float pad = 0.02f * std::max(u1 - u0, v1 - v0) + 1.f;
	u0 -= pad, u1 += pad, v0 -= pad, v1 += pad, d0 -= pad, d1 += pad;
	Params p = {};
	std::memcpy(p.axisU, a, sizeof(a));
	p.axisU[3] = -u0;
	std::memcpy(p.axisV, b, sizeof(b));
	p.axisV[3] = -v0;
	std::memcpy(p.axisS, sun, 3 * sizeof(float));
	p.axisS[3] = d1;
	p.scale[0] = 1.f / (u1 - u0), p.scale[1] = 1.f / (v1 - v0), p.scale[2] = 1.f / (d1 - d0);
	p.screen[0] = (float)target.extent.width, p.screen[1] = (float)target.extent.height;
	p.screen[2] = scene.width, p.screen[3] = scene.height;
	p.light[0] = scene.ambient, p.light[1] = scene.darkness, p.light[2] = zFar * 1.05f, p.light[3] = Bias * p.scale[2];
	for (uint32_t i = count; i < count + floorCount; i++)
		p.light[2] = std::max(p.light[2], s.corners[i].z * 1.05f);

	// the flashes; the brightest's view over the models (its shadow map): from it toward their middle, wide enough
	const uint32_t lights = std::min<uint32_t>((uint32_t)scene.lights.size(), deadstation::scene::MaxLights);
	p.flashAt[3] = (float)lights;
	for (uint32_t i = 0; i < lights; i++)
	{
		const deadstation::scene::Light& l = scene.lights[i];
		p.lightAt[i][0] = l.x, p.lightAt[i][1] = l.y, p.lightAt[i][2] = l.z, p.lightAt[i][3] = l.reach;
		p.lightColour[i][0] = l.r, p.lightColour[i][1] = l.g, p.lightColour[i][2] = l.b;
	}
	float mid[3] = { 0, 0, 0 };
	for (uint32_t i = 0; i < count; i++)
		mid[0] += scene.corners[i].x, mid[1] += scene.corners[i].y, mid[2] += scene.corners[i].z;
	for (float& m : mid)
		m /= (float)count;
	float radius = 1.f;
	for (uint32_t i = 0; i < count; i++)
	{
		const float dx = scene.corners[i].x - mid[0], dy = scene.corners[i].y - mid[1], dz = scene.corners[i].z - mid[2];
		radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz));
	}
	const float at[3] = { lights ? scene.lights[0].x : mid[0], lights ? scene.lights[0].y : mid[1] - 2 * radius, lights ? scene.lights[0].z : mid[2] };
	float ahead[3] = { mid[0] - at[0], mid[1] - at[1], mid[2] - at[2] };
	float dist = std::sqrt(ahead[0] * ahead[0] + ahead[1] * ahead[1] + ahead[2] * ahead[2]);
	if (dist < 1.f)
		ahead[0] = 0, ahead[1] = 0, ahead[2] = 1, dist = 1.f;
	for (float& c : ahead)
		c /= dist;
	const float pick[3] = { std::fabs(ahead[1]) < 0.9f ? 0.f : 1.f, std::fabs(ahead[1]) < 0.9f ? 1.f : 0.f, 0.f };
	float right[3] = { pick[1] * ahead[2] - pick[2] * ahead[1], pick[2] * ahead[0] - pick[0] * ahead[2], pick[0] * ahead[1] - pick[1] * ahead[0] };
	const float rn = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
	for (float& c : right)
		c /= rn;
	const float up[3] = { ahead[1] * right[2] - ahead[2] * right[1], ahead[2] * right[0] - ahead[0] * right[2], ahead[0] * right[1] - ahead[1] * right[0] };
	const float tanHalf = dist > radius * 1.05f ? std::min(5.67f, radius / std::sqrt(dist * dist - radius * radius)) : 5.67f;
	std::memcpy(p.flashRight, right, sizeof(right));
	p.flashRight[3] = tanHalf;
	std::memcpy(p.flashUp, up, sizeof(up));
	p.flashUp[3] = 20.f;
	std::memcpy(p.flashAhead, ahead, sizeof(ahead));
	p.flashAhead[3] = dist + radius + 100.f;
	std::memcpy(p.flashAt, at, sizeof(at));
	p.options[0] = scene.selfShadow ? 1.f : 0.f;
	std::memcpy(p.floorPlane, scene.floorPlane, sizeof(p.floorPlane));
	std::memcpy(p.project, scene.project, sizeof(p.project));
	std::memcpy(s.paramsMapped, &p, sizeof(p));
	s.count = count;
	s.floorCount = floorCount;
	s.floorBase = std::min(scene.floorBase, floorCount);
	s.lights = lights;

	const vk::ClearValue clear = vk::ClearDepthStencilValue(1.f, 0);
	cmd.beginRenderPass(vk::RenderPassBeginInfo(*shadowPass, *s.shadowFramebuffer, vk::Rect2D({ 0, 0 }, { ShadowSize, ShadowSize }), clear),
			vk::SubpassContents::eInline);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *layout, 1, s.set, nullptr);
	const vk::DeviceSize offset = 0;
	cmd.bindVertexBuffers(0, *s.vertices, offset);
	cmd.draw(count, 1, 0, 0);
	cmd.endRenderPass();
	// the brightest flash's (cleared even with none, so its map is always ready to be read)
	cmd.beginRenderPass(vk::RenderPassBeginInfo(*shadowPass, *s.flashFramebuffer, vk::Rect2D({ 0, 0 }, { ShadowSize, ShadowSize }), clear),
			vk::SubpassContents::eInline);
	if (lights > 0 && scene.flashMap)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *flashPipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *layout, 1, s.set, nullptr);
		cmd.bindVertexBuffers(0, *s.vertices, offset);
		cmd.draw(count, 1, 0, 0);
	}
	cmd.endRenderPass();
}

void SceneLight::draw(vk::CommandBuffer cmd, uint32_t page, vk::DescriptorSet frame, const Frame& f)
{
	Slot& s = slots[page % slots.size()];
	if (s.count == 0)
		return;
	Params *p = (Params *)s.paramsMapped;
	p->frameUv[0] = f.uvScale[0], p->frameUv[1] = f.uvScale[1], p->frameUv[2] = f.uvMax[0], p->frameUv[3] = f.uvMax[1];
	p->frameAt[0] = f.uvOffset[0], p->frameAt[1] = f.uvOffset[1], p->frameAt[2] = f.uvMin[0], p->frameAt[3] = f.uvMin[1];
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *lightPipeline);
	const vk::DescriptorSet sets[2] = { frame, s.set };
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *layout, 0, sets, nullptr);
	const vk::DeviceSize offset = 0;
	cmd.bindVertexBuffers(0, *s.vertices, offset);
	const vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	int32_t floor = 0;
	cmd.pushConstants(*layout, stages, 0, sizeof(floor), &floor);
	cmd.draw(s.count, 1, 0, 0);
	if (s.floorCount == 0)
		return;
	// the sun's shadows only round the models: most of a lamp-lit floor has none to draw
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *floorPipeline);
	floor = 1;
	cmd.pushConstants(*layout, stages, 0, sizeof(floor), &floor);
	cmd.draw(s.floorBase, 1, s.count, 0);
	// each flash: its planar shadows of the models marked, then its light added to the floor outside them. Each with its own
	// stencil value, so a pixel in shadow of one still takes the others' light: as many shadows as flashes.
	for (uint32_t i = 0; i < s.lights; i++)
	{
		const int32_t index = (int32_t)i;
		cmd.pushConstants(*layout, stages, 0, sizeof(index), &index);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *castPipeline);
		cmd.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, i + 1);
		cmd.draw(s.count, 1, 0, 0);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *flashFloorPipeline);
		cmd.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, i + 1);
		cmd.draw(s.floorCount - s.floorBase, 1, s.count + s.floorBase, 0);
	}
}
