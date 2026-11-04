// Per-pixel self-shadowing of a recomp module's models (scene.h) over the frames Deadstation presents
// (video.h's VideoLayer): each frame, the latest layer the game's plugin published is drawn from the sun into a
// shadow map (and from its brightest flash into another), then drawn where the game drew it, at its depth, over the
// frame: the frame's pixels of the models that the sun's light doesn't reach lose the sun's share of their light,
// and the models' and the floor's pixels round the flashes take their light (the brightest's blocked by the models).
#pragma once
#include "scene.h"
#include "video.h"

#include <string>
#include <vector>

class SceneLight : public VideoLayer
{
public:
	// target: the presenter's (Video::layerTarget, or a test's)
	bool init(const Target& target, std::string& error);
	void prepare(vk::CommandBuffer cmd, uint32_t page) override;
	void draw(vk::CommandBuffer cmd, uint32_t page, vk::DescriptorSet frame, const Frame& f) override;

	static constexpr uint32_t MaxCorners = 16384;
	static constexpr uint32_t ShadowSize = 512;

private:
	struct Slot
	{
		vk::UniqueBuffer vertices;
		vk::UniqueDeviceMemory verticesMemory;
		deadstation::scene::Corner *corners = nullptr;
		vk::UniqueBuffer params;
		vk::UniqueDeviceMemory paramsMemory;
		void *paramsMapped = nullptr;
		vk::UniqueImage shadow;
		vk::UniqueDeviceMemory shadowMemory;
		vk::UniqueImageView shadowView;
		vk::UniqueFramebuffer shadowFramebuffer;
		vk::UniqueImage flash;
		vk::UniqueDeviceMemory flashMemory;
		vk::UniqueImageView flashView;
		vk::UniqueFramebuffer flashFramebuffer;
		vk::DescriptorSet set;
		uint32_t count = 0;		// the models' corners this frame (0: nothing)
		uint32_t floorCount = 0;	// the floor's, after them
		uint32_t floorBase = 0;		// the first of those: where the sun's shadows fall; the rest, round the lights
		uint32_t lights = 0;		// the flashes this frame
	};

	uint32_t memoryType(uint32_t bits, vk::MemoryPropertyFlags flags) const;

	Target target;
	vk::Format shadowFormat = vk::Format::eUndefined;
	vk::UniqueRenderPass shadowPass;
	vk::UniqueSampler shadowSampler;
	vk::UniqueDescriptorSetLayout setLayout;
	vk::UniqueDescriptorPool pool;
	vk::UniquePipelineLayout layout;
	vk::UniquePipeline shadowPipeline;
	vk::UniquePipeline flashPipeline;
	vk::UniquePipeline lightPipeline;		// the models
	vk::UniquePipeline floorPipeline;		// the floor: tested against them, not written
	vk::UniquePipeline castPipeline;		// the models flattened from a flash: the stencil
	vk::UniquePipeline flashFloorPipeline;	// that flash's light on the floor, outside them
	std::vector<Slot> slots;
	deadstation::scene::Frame scene;
};
