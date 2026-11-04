// Vulkan through vulkan.hpp, its functions loaded at run time from the device's libvulkan.so.1
// (video.cpp holds the dispatcher's storage).
#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
