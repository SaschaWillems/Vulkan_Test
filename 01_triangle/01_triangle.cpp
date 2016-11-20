/*
* Vulkan Tutorial 01 - Complete setup and triangle rendering as single file using vulkan_hpp
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <exception>
#include <iostream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

#define ENABLE_VALIDATION true
#define USE_STAGING true

namespace debug
{
	// Function pointers, debug is an extension, so we need to load them manually
	PFN_vkCreateDebugReportCallbackEXT createDebugReportCallbackEXT;
	PFN_vkDestroyDebugReportCallbackEXT destroyDebugReportCallbackEXT;
	PFN_vkDebugReportMessageEXT debugReportMessageEXT;

	VkDebugReportCallbackEXT debugReportCallback;

	VkBool32 debugMessageCallback(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT objType,
		uint64_t srcObject,
		size_t location,
		int32_t msgCode,
		const char* pLayerPrefix,
		const char* pMsg,
		void* pUserData)
	{
		// Select prefix depending on flags passed to the callback
		// Note that multiple flags may be set for a single validation message
		std::string prefix("");

		// Error that may result in undefined behaviour
		if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
		{
			prefix += "ERROR:";
		};
		// Warnings may hint at unexpected / non-spec API usage
		if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
		{
			prefix += "WARNING:";
		};
		// May indicate sub-optimal usage of the API
		if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
		{
			prefix += "PERFORMANCE:";
		};
		// Informal messages that may become handy during debugging
		if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT)
		{
			prefix += "INFO:";
		}
		// Diagnostic info from the Vulkan loader and layers
		// Usually not helpful in terms of API usage, but may help to debug layer and loader problems 
		if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
		{
			prefix += "DEBUG:";
		}

		// Display message to default output (console if activated)
		std::cout << prefix << " [" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg << "\n";

		fflush(stdout);

		return VK_FALSE;
	}

	void setupDebugging(vk::Instance instance, vk::DebugReportFlagsEXT flags)
	{
		createDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
		destroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
		debugReportMessageEXT = (PFN_vkDebugReportMessageEXT)vkGetInstanceProcAddr(instance, "vkDebugReportMessageEXT");

		VkDebugReportCallbackCreateInfoEXT dbgCreateInfo = {};
		dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
		dbgCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;
		dbgCreateInfo.flags = flags.operator VkSubpassDescriptionFlags();

		VkResult err = createDebugReportCallbackEXT(instance, &dbgCreateInfo, nullptr, &debugReportCallback);
		assert(!err);
	}

	void freeDebugCallback(vk::Instance instance)
	{
		destroyDebugReportCallbackEXT(instance, debugReportCallback, nullptr);
	}
}

// Swapchain ===========================================================================================================================================

class SwapChain
{
private:
	vk::Instance instance;
	vk::Device device;
	vk::PhysicalDevice physicalDevice;
	vk::SurfaceKHR surface;
public:
	typedef struct _SwapChainBuffers {
		vk::Image image;
		vk::ImageView view;
	} SwapChainBuffer;
	vk::Format colorFormat;
	vk::ColorSpaceKHR colorSpace;
	vk::SwapchainKHR swapChain;
	std::vector<vk::Image> images;
	std::vector<SwapChainBuffer> buffers;
	uint32_t queueNodeIndex = UINT32_MAX;

	SwapChain(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device)
	{
		this->instance = instance;
		this->physicalDevice = physicalDevice;
		this->device = device;
	}

	~SwapChain()
	{
		if (swapChain)
		{
			for (auto& buffer : buffers)
			{
				device.destroyImageView(buffer.view);
			}
		}
		if (surface) // todo
		{
			device.destroySwapchainKHR(swapChain);
			instance.destroySurfaceKHR(surface);
		}
		//surface = VK_NULL_HANDLE; // todo
		//swapChain = VK_NULL_HANDLE; // todo
	}

	void createSurface(
#ifdef _WIN32
		void* platformHandle, void* platformWindow
#else
#ifdef __ANDROID__
		ANativeWindow* window
#else
#ifdef _DIRECT2DISPLAY
		uint32_t width, uint32_t height
#else
		xcb_connection_t* connection, xcb_window_t window
#endif
#endif
#endif
	)
	{

		// Create the os-specific surface
		vk::Result result;

#ifdef _WIN32
		vk::Win32SurfaceCreateInfoKHR surfaceCreateInfo;
		surfaceCreateInfo.hinstance = (HINSTANCE)platformHandle;
		surfaceCreateInfo.hwnd = (HWND)platformWindow;
		surface = instance.createWin32SurfaceKHR(surfaceCreateInfo);
#else
#ifdef __ANDROID__
		vk::AndroidSurfaceCreateInfoKHR surfaceCreateInfo;
		surfaceCreateInfo.window = window;
		surface = instance.createAndroidSurfaceKHR(surfaceCreateInfo);
#else
#if defined(_DIRECT2DISPLAY)
		createDirect2DisplaySurface(width, height);
#else
		vk::XcbSurfaceCreateInfoKHR surfaceCreateInfo;
		surfaceCreateInfo.connection = connection;
		surfaceCreateInfo.window = window;
		surface = instance.createXcbSurfaceKHR(surfaceCreateInfo);
#endif
#endif
#endif

		assert(surface);

		// Get available queue family properties
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
		queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// Iterate over each queue to learn whether it supports presenting:
		// Find a queue with present support
		// Will be used to present the swap chain images to the windowing system
		std::vector<vk::Bool32> supportsPresent(queueFamilyProperties.size());
		for (size_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			supportsPresent[i] = physicalDevice.getSurfaceSupportKHR(i, surface);
		}

		// Search for a graphics and a present queue in the array of queue families, try to find one that supports both
		uint32_t graphicsQueueNodeIndex = UINT32_MAX;
		uint32_t presentQueueNodeIndex = UINT32_MAX;
		for (size_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
			{
				if (graphicsQueueNodeIndex == UINT32_MAX)
				{
					graphicsQueueNodeIndex = i;
				}

				if (supportsPresent[i] == VK_TRUE)
				{
					graphicsQueueNodeIndex = i;
					presentQueueNodeIndex = i;
					break;
				}
			}
		}
		if (presentQueueNodeIndex == UINT32_MAX)
		{
			// If there's no queue that supports both present and graphics try to find a separate present queue
			for (size_t i = 0; i < queueFamilyProperties.size(); i++)
			{
				if (supportsPresent[i] == VK_TRUE)
				{
					presentQueueNodeIndex = i;
					break;
				}
			}
		}

		// Exit if either a graphics or a presenting queue hasn't been found
		if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX)
		{
			//vkTools::exitFatal("Could not find a graphics and/or presenting queue!", "Fatal error");
		}

		// todo : Add support for separate graphics and presenting queue
		if (graphicsQueueNodeIndex != presentQueueNodeIndex)
		{
			//vkTools::exitFatal("Separate graphics and presenting queues are not supported yet!", "Fatal error");
		}

		queueNodeIndex = graphicsQueueNodeIndex;

		// Get list of supported surface formats		
		std::vector<vk::SurfaceFormatKHR> surfaceFormats;
		surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);

		// If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
		// there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
		if ((surfaceFormats.size() == 1) && (surfaceFormats[0].format == vk::Format::eUndefined))
		{
			colorFormat = vk::Format::eB8G8R8A8Unorm;
		}
		else
		{
			// Always select the first available color format
			// If you need a specific format (e.g. SRGB) you'd need to
			// iterate over the list of available surface format and
			// check for it's presence
			colorFormat = surfaceFormats[0].format;
		}
		colorSpace = surfaceFormats[0].colorSpace;
	}

	void create(uint32_t *width, uint32_t *height, bool vsync = false)
	{
		vk::SwapchainKHR oldSwapchain = swapChain;

		// Get physical device surface properties and formats
		vk::SurfaceCapabilitiesKHR surfCaps = physicalDevice.getSurfaceCapabilitiesKHR(surface);

		// Get available present modes
		std::vector<vk::PresentModeKHR> presentModes;
		presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
		assert(presentModes.size() > 0);

		vk::Extent2D swapchainExtent;
		// If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
		if (surfCaps.currentExtent.width == (uint32_t)-1)
		{
			// If the surface size is undefined, the size is set to
			// the size of the images requested.
			swapchainExtent.width = *width;
			swapchainExtent.height = *height;
		}
		else
		{
			// If the surface size is defined, the swap chain size must match
			swapchainExtent = surfCaps.currentExtent;
			*width = surfCaps.currentExtent.width;
			*height = surfCaps.currentExtent.height;
		}


		// Select a present mode for the swapchain

		// The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
		// This mode waits for the vertical blank ("v-sync")
		vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

		// If v-sync is not requested, try to find a mailbox mode
		// It's the lowest latency non-tearing present mode available
		if (!vsync)
		{
			for (auto presentMode : presentModes)
			{
				if (presentMode == vk::PresentModeKHR::eMailbox)
				{
					swapchainPresentMode = vk::PresentModeKHR::eMailbox;
					break;
				}
				if ((swapchainPresentMode != vk::PresentModeKHR::eMailbox) && (presentMode == vk::PresentModeKHR::eImmediate))
				{
					swapchainPresentMode = vk::PresentModeKHR::eImmediate;
				}
			}
		}

		// Determine the number of images
		uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
		if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
		{
			desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
		}

		// Find the transformation of the surface, prefer a non-rotated transform
		vk::SurfaceTransformFlagBitsKHR preTransform;
		if (surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
		{
			preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
		}
		else
		{
			preTransform = surfCaps.currentTransform;
		}

		vk::SwapchainCreateInfoKHR swapchainCI;
		swapchainCI.surface = surface;
		swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
		swapchainCI.imageFormat = colorFormat;
		swapchainCI.imageColorSpace = colorSpace;
		swapchainCI.imageExtent = swapchainExtent;
		swapchainCI.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		swapchainCI.preTransform = preTransform;
		swapchainCI.imageArrayLayers = 1;
		swapchainCI.imageSharingMode = vk::SharingMode::eExclusive;
		swapchainCI.queueFamilyIndexCount = 0;
		swapchainCI.pQueueFamilyIndices = NULL;
		swapchainCI.presentMode = swapchainPresentMode;
		swapchainCI.oldSwapchain = oldSwapchain;
		swapchainCI.clipped = VK_TRUE;
		swapchainCI.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

		swapChain = device.createSwapchainKHR(swapchainCI);

		// If an existing sawp chain is re-created, destroy the old swap chain
		// This also cleans up all the presentable images
		if (oldSwapchain)
		{
			for (auto& buffer : buffers)
			{
				device.destroyImageView(buffer.view);
			}
			device.destroySwapchainKHR(oldSwapchain);
		}

		// Get the swap chain images
		images = device.getSwapchainImagesKHR(swapChain);

		// Get the swap chain buffers containing the image and imageview
		buffers.resize(images.size());
		for (size_t i = 0; i < buffers.size(); i++)
		{
			vk::ImageViewCreateInfo colorAttachmentView;
			colorAttachmentView.format = colorFormat;
			colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			colorAttachmentView.subresourceRange.baseMipLevel = 0;
			colorAttachmentView.subresourceRange.levelCount = 1;
			colorAttachmentView.subresourceRange.baseArrayLayer = 0;
			colorAttachmentView.subresourceRange.layerCount = 1;
			colorAttachmentView.viewType = vk::ImageViewType::e2D;

			buffers[i].image = images[i];

			colorAttachmentView.image = buffers[i].image;

			buffers[i].view = device.createImageView(colorAttachmentView);
		}
	}

	void acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t &imageIndex)
	{
		auto resultValue = device.acquireNextImageKHR(swapChain, UINT64_MAX, presentCompleteSemaphore, vk::Fence());
		imageIndex = resultValue.value;
	}

	void queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore = VK_NULL_HANDLE)
	{
		vk::PresentInfoKHR presentInfo;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain;
		presentInfo.pImageIndices = &imageIndex;
		if (waitSemaphore)
		{
			presentInfo.pWaitSemaphores = &waitSemaphore;
			presentInfo.waitSemaphoreCount = 1;
		}
		queue.presentKHR(presentInfo);
	}

#if defined(_DIRECT2DISPLAY)
	// todo
#endif 
};

// Example =============================================================================================================================================

class VulkanExample
{
public:
	// Windows/surface
	vk::Extent2D windowSize;
	SwapChain *swapChain;

	// OS specific 
#if defined(_WIN32)
	HWND window;
	HINSTANCE windowInstance;
#elif defined(__ANDROID__)
	android_app* androidApp;
	bool focused = false;
#elif defined(__linux__)
	struct {
		bool left = false;
		bool right = false;
		bool middle = false;
	} mouseButtons;
	bool quit = false;
	xcb_connection_t *connection;
	xcb_screen_t *screen;
	xcb_window_t window;
	xcb_intern_atom_reply_t *atom_wm_delete_window;
#endif


	// Camera
	float zoom;
	glm::vec3 rotation;

	vk::Instance instance;

	// Vulkan device
	vk::Device device;
	vk::PhysicalDevice physicalDevice;

	vk::RenderPass renderPass;
	std::vector<vk::Framebuffer> frameBuffers;
	uint32_t currentFrameBuffer = 0;

	struct
	{
		vk::Image image;
		vk::DeviceMemory memory;
		vk::ImageView view;
	} depthStencil;

	// Vertex buffer and attributes
	struct {
		vk::DeviceMemory memory;															// Handle to the device memory for this buffer
		vk::Buffer buffer;																// Handle to the Vulkan buffer object that the memory is bound to
		vk::PipelineVertexInputStateCreateInfo inputState;
		vk::VertexInputBindingDescription inputBinding;
		std::vector<vk::VertexInputAttributeDescription> inputAttributes;
	} vertices;

	// Index buffer
	struct 
	{
		vk::DeviceMemory memory;		
		vk::Buffer buffer;			
		uint32_t count;
	} indices;

	// Uniform block object
	struct {
		vk::DeviceMemory memory;		
		vk::Buffer buffer;			
		vk::DescriptorBufferInfo descriptor;
	}  uniformDataVS;

	// For simplicity we use the same uniform block layout as in the shader:
	//
	//	layout(set = 0, binding = 0) uniform UBO
	//	{
	//		mat4 projectionMatrix;
	//		mat4 modelMatrix;
	//		mat4 viewMatrix;
	//	} ubo;
	//
	// This way we can just memcopy the ubo data to the ubo
	// Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
	struct {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	} uboVS;

	// Graphics queue used to submit commands
	vk::Queue queue;

	vk::PipelineLayout pipelineLayout;
	vk::PipelineCache pipelineCache;
	vk::Pipeline pipeline;

	std::vector<vk::ShaderModule> shaderModules;

	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;

	// Command buffers are allocated from a (per-thread) pool
	vk::CommandPool commandPool;

	// Command buffers used to store rendering commands
	std::vector<vk::CommandBuffer> drawCmdBuffers;

	// Synchronization primitives
	// Synchronization is an important concept of Vulkan that OpenGL mostly hid away. Getting this right is crucial to using Vulkan.

	// Semaphores
	// Used to coordinate operations within the graphics queue and ensure correct command ordering
	vk::Semaphore presentCompleteSemaphore;
	vk::Semaphore renderCompleteSemaphore;

	// Fences
	// Used to check the completion of queue operations (e.g. command buffer execution)
	std::vector<vk::Fence> waitFences;

	VulkanExample(HINSTANCE hinstance, WNDPROC wndproc)
	{
		windowSize = { 1280, 720 };

		createWindow(hinstance, wndproc);

		zoom = -2.5f;

		createInstance(ENABLE_VALIDATION);
		
		// Get first physical device
		std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
		assert(physicalDevices.size() > 0);
		// Always use first physical device
		physicalDevice = physicalDevices[0];

		createDevice(physicalDevice);

		createSwapChain();

		createRenderPass();
		createFrameBuffers();

		// Command buffer pool
		vk::CommandPoolCreateInfo cmdPoolCreateInfo;
		cmdPoolCreateInfo.queueFamilyIndex = swapChain->queueNodeIndex;
		cmdPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		commandPool = device.createCommandPool(cmdPoolCreateInfo);

		prepareVertices(USE_STAGING);

		createUniformBuffers();

		createDescriptors();
		createGraphicsPipeline();

		createSynchronizationPrimitives();

		createCommandBuffers();

		renderLoop();
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 

		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyPipelineCache(pipelineCache);
		for (auto& module : shaderModules) {
			device.destroyShaderModule(module);
		}

		device.destroyBuffer(vertices.buffer);
		device.freeMemory(vertices.memory);

		device.destroyBuffer(indices.buffer);
		device.freeMemory(indices.memory);

		device.destroyBuffer(uniformDataVS.buffer);
		device.freeMemory(uniformDataVS.memory);

		device.destroySemaphore(presentCompleteSemaphore);
		device.destroySemaphore(renderCompleteSemaphore);

		for (auto& fence : waitFences) { 
			device.destroyFence(fence); 
		}

		device.destroyDescriptorSetLayout(descriptorSetLayout);
		device.destroyDescriptorPool(descriptorPool);

		for (auto& frameBuffer : frameBuffers) { 
			device.destroyFramebuffer(frameBuffer); 
		}

		device.destroyRenderPass(renderPass);
		device.destroyImage(depthStencil.image);
		device.destroyImageView(depthStencil.view);
		device.freeMemory(depthStencil.memory);

		// Destroying the command pool also releases allocated command buffers
		device.destroyCommandPool(commandPool);

		delete swapChain;

		device.destroy();

//		if (enableValidation)
		{
			debug::freeDebugCallback(instance);
		}

		instance.destroy();

#if defined(__linux)
		xcb_destroy_window(connection, window);
		xcb_disconnect(connection);
#endif
	}

	std::string getAssetPath()
	{
#if defined(__ANDROID__)
		return "";
#else
		return "./../../data/";
#endif
	}

#if defined(__ANDROID__)
	// Android shaders are stored as assets in the apk
	// So they need to be loaded via the asset manager
	VkShaderModule loadShader(AAssetManager* assetManager, const char *fileName, VkDevice device, VkShaderStageFlagBits stage)
	{
		// Load shader from compressed asset
		AAsset* asset = AAssetManager_open(assetManager, fileName, AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);
		assert(size > 0);

		char *shaderCode = new char[size];
		AAsset_read(asset, shaderCode, size);
		AAsset_close(asset);

		VkShaderModule shaderModule;
		VkShaderModuleCreateInfo moduleCreateInfo;
		moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleCreateInfo.pNext = NULL;
		moduleCreateInfo.codeSize = size;
		moduleCreateInfo.pCode = (uint32_t*)shaderCode;
		moduleCreateInfo.flags = 0;

		VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

		delete[] shaderCode;

		return shaderModule;
	}
#else
	vk::ShaderModule loadSPIRVShader(std::string fileName)
	{
		size_t size;

		FILE *fp = fopen(fileName.c_str(), "rb");
		assert(fp);

		fseek(fp, 0L, SEEK_END);
		size = ftell(fp);

		fseek(fp, 0L, SEEK_SET);

		//shaderCode = malloc(size);
		char *shaderCode = new char[size];
		size_t retval = fread(shaderCode, size, 1, fp);
		assert(retval == 1);
		assert(size > 0);

		fclose(fp);

		vk::ShaderModuleCreateInfo moduleCreateInfo;
		moduleCreateInfo.codeSize = size;
		moduleCreateInfo.pCode = (uint32_t*)shaderCode;

		vk::ShaderModule shaderModule = device.createShaderModule(moduleCreateInfo);

		delete[] shaderCode;

		return shaderModule;
	}
#endif

#ifdef _WIN32
	HWND createWindow(HINSTANCE hinstance, WNDPROC wndproc)
	{
		this->windowInstance = hinstance;

		bool fullscreen = false;
		/*
		for (auto arg : args)
		{
			if (arg == std::string("-fullscreen"))
			{
				fullscreen = true;
			}
		}
		*/

		WNDCLASSEX wndClass;

		wndClass.cbSize = sizeof(WNDCLASSEX);
		wndClass.style = CS_HREDRAW | CS_VREDRAW;
		wndClass.lpfnWndProc = wndproc;
		wndClass.cbClsExtra = 0;
		wndClass.cbWndExtra = 0;
		wndClass.hInstance = hinstance;
		wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
		wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		wndClass.lpszMenuName = NULL;
		wndClass.lpszClassName = "VULKAN_TUTORIAL";
		wndClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);

		if (!RegisterClassEx(&wndClass))
		{
			std::cout << "Could not register window class!\n";
			fflush(stdout);
			exit(1);
		}

		int screenWidth = GetSystemMetrics(SM_CXSCREEN);
		int screenHeight = GetSystemMetrics(SM_CYSCREEN);

		if (fullscreen)
		{
			DEVMODE dmScreenSettings;
			memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
			dmScreenSettings.dmSize = sizeof(dmScreenSettings);
			dmScreenSettings.dmPelsWidth = screenWidth;
			dmScreenSettings.dmPelsHeight = screenHeight;
			dmScreenSettings.dmBitsPerPel = 32;
			dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
		}

		DWORD dwExStyle;
		DWORD dwStyle;

		if (fullscreen)
		{
			dwExStyle = WS_EX_APPWINDOW;
			dwStyle = WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
		}
		else
		{
			dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
			dwStyle = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
		}

		RECT windowRect;
		windowRect.left = 0L;
		windowRect.top = 0L;
		windowRect.right = fullscreen ? (long)screenWidth : (long)windowSize.width;
		windowRect.bottom = fullscreen ? (long)screenHeight : (long)windowSize.height;

		AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);

		window = CreateWindowEx(0,
			"VULKAN_TUTORIAL",
			"VULKAN TUTORIAL 01",
			dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
			0,
			0,
			windowRect.right - windowRect.left,
			windowRect.bottom - windowRect.top,
			NULL,
			NULL,
			hinstance,
			NULL);

		if (!fullscreen)
		{
			// Center on screen
			uint32_t x = (GetSystemMetrics(SM_CXSCREEN) - windowRect.right) / 2;
			uint32_t y = (GetSystemMetrics(SM_CYSCREEN) - windowRect.bottom) / 2;
			SetWindowPos(window, 0, x, y, 0, 0, SWP_NOZORDER | SWP_NOSIZE);
		}

		if (!window)
		{
			printf("Could not create window!\n");
			fflush(stdout);
			return 0;
			exit(1);
		}

		ShowWindow(window, SW_SHOW);
		SetForegroundWindow(window);
		SetFocus(window);

		// Console
		AllocConsole();
		AttachConsole(GetCurrentProcessId());
		FILE *stream;
		freopen_s(&stream, "CONOUT$", "w+", stdout);
		SetConsoleTitle(TEXT("VULKAN_TUTORIAL"));

		return window;
	}
#endif

	void createInstance(bool enableValidation)
	{
		vk::ApplicationInfo appInfo;
		appInfo.pApplicationName = "Vulkan Tutorial 01";
		appInfo.pEngineName = "VK_TUTORIAL";
		appInfo.apiVersion = VK_API_VERSION_1_0;

		std::vector<const char*> enabledExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

		// Enable surface extensions depending on os
#if defined(_WIN32)
		enabledExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__ANDROID__)
		enabledExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
		enabledExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(__linux__)
		enabledExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif

		vk::InstanceCreateInfo instanceCreateInfo;
		instanceCreateInfo.pApplicationInfo = &appInfo;
		if (enabledExtensions.size() > 0)
		{
			if (enableValidation)
			{
				enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			}
			instanceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
			instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
		}
		if (enableValidation)
		{
			instanceCreateInfo.enabledLayerCount = 1;
			const char *validationLayerNames[] = { "VK_LAYER_LUNARG_standard_validation" };
			instanceCreateInfo.ppEnabledLayerNames = validationLayerNames;
		}

		instance = vk::createInstance(instanceCreateInfo);

		if (enableValidation)
		{
			debug::setupDebugging(instance, vk::DebugReportFlagBitsEXT::eError);
		}
	}

	void createDevice(vk::PhysicalDevice physicalDevice)
	{
		// Request one graphics queue

		// Get index of first queue family that supports graphics

		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		uint32_t graphicsQueueFamilyIndex = 0;
		for (size_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
			{
				graphicsQueueFamilyIndex = i;
				break;
			}
		}

		const float defaultQueuePriority(0.0f);

		vk::DeviceQueueCreateInfo queueCreatInfo;
		queueCreatInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
		queueCreatInfo.queueCount = 1;
		queueCreatInfo.pQueuePriorities = &defaultQueuePriority;

		// Create the logical device representation
		std::vector<const char*> deviceExtensions;
		deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		vk::DeviceCreateInfo deviceCreateInfo;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreatInfo;
		// No specific features used in this tutorial
		deviceCreateInfo.pEnabledFeatures = nullptr;					

		if (deviceExtensions.size() > 0)
		{
			deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
			deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		}

		device = physicalDevice.createDevice(deviceCreateInfo);	

		// Get a graphics queue from the device
		queue = device.getQueue(graphicsQueueFamilyIndex, 0);
	}

	void createSwapChain()
	{
		swapChain = new SwapChain(instance, physicalDevice, device);
		swapChain->createSurface(windowInstance, window);
		swapChain->create(&windowSize.width, &windowSize.height);
	}

	// This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visibile)
	// Upon success it will return the index of the memory type that fits our requestes memory properties
	// This is necessary as implementations can offer an arbitrary number of memory types with different
	// memory properties. 
	// You can check http://vulkan.gpuinfo.org/ for details on different memory configurations
	uint32_t getMemoryTypeIndex(uint32_t typeBits, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties deviceMemoryProperties = physicalDevice.getMemoryProperties();
		// Iterate over all memory types available for the device used in this example
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{						
					return i;
				}
			}
			typeBits >>= 1;
		}

		throw "Could not find a suitable memory type!";
	}

	// Create the Vulkan synchronization primitives used in this example
	void createSynchronizationPrimitives()
	{
		// Semaphores (Used for correct command ordering)
		vk::SemaphoreCreateInfo semaphoreCreateInfo;

		// Semaphore used to ensures that image presentation is complete before starting to submit again
		presentCompleteSemaphore = device.createSemaphore(semaphoreCreateInfo);

		// Semaphore used to ensures that all commands submitted have been finished before submitting the image to the queue
		renderCompleteSemaphore = device.createSemaphore(semaphoreCreateInfo);

		// Fences (Used to check draw command buffer completion)
		vk::FenceCreateInfo fenceCreateInfo;
		// Create in signaled state so we don't wait on first render of each command buffer
		fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;
		waitFences.resize(swapChain->images.size());
		for (auto& fence : waitFences)
		{
			fence = device.createFence(fenceCreateInfo);
		}
	}

	// Build separate command buffers for every framebuffer image
	void createCommandBuffers()
	{
		// One command buffer for each swap chain image
		vk::CommandBufferAllocateInfo cmdBuferAllocInfo;
		cmdBuferAllocInfo.commandPool = commandPool;
		cmdBuferAllocInfo.commandBufferCount = swapChain->images.size();
		cmdBuferAllocInfo.level = vk::CommandBufferLevel::ePrimary;
		
		drawCmdBuffers = device.allocateCommandBuffers(cmdBuferAllocInfo);

		vk::CommandBufferBeginInfo cmdBufferBeginInfo;

		// Set clear values for all framebuffer attachments with loadOp set to clear
		// We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.2f, 1.0f });
		clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0.0f);

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent = windowSize;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;
	
		for (size_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufferBeginInfo);
			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport(0.0f, 0.0f, windowSize.width, windowSize.height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, 1, &viewport);

			vk::Rect2D scissor(vk::Offset2D(0, 0), windowSize);
			drawCmdBuffers[i].setScissor(0, 1, &scissor);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

			VkDeviceSize offsets[1] = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(0, 1, &vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(indices.count, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();
			drawCmdBuffers[i].end();
		}
	}

	void renderFrame()
	{
		// Get next image in the swap chain (back/front buffer)
		swapChain->acquireNextImage(presentCompleteSemaphore, currentFrameBuffer);

		// Use a fence to wait until the command buffer has finished execution before using it again
		device.waitForFences(1, &waitFences[currentFrameBuffer], VK_TRUE, UINT64_MAX);
		device.resetFences(1, &waitFences[currentFrameBuffer]);

		vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submitInfo;
		submitInfo.pWaitDstStageMask = &waitStageMask;
		submitInfo.pWaitSemaphores = &presentCompleteSemaphore;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderCompleteSemaphore;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentFrameBuffer];
		submitInfo.commandBufferCount = 1;

		queue.submit(1, &submitInfo, waitFences[currentFrameBuffer]);

		swapChain->queuePresent(queue, currentFrameBuffer, renderCompleteSemaphore);
	}

	// Prepare vertex and index buffers for an indexed triangle
	// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
	void prepareVertices(bool useStagingBuffers)
	{
		struct Vertex 
		{
			float position[3];
			float color[3];
		};

		// Setup vertices
		std::vector<Vertex> vertexBuffer = 
		{
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
		};
		uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

		// Setup indices
		std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
		indices.count = static_cast<uint32_t>(indexBuffer.size());
		uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;

		void *data;

		//if (useStagingBuffers)
		{
			struct StagingBuffer {
				vk::DeviceMemory memory;
				vk::Buffer buffer;
			};

			struct {
				StagingBuffer vertices;
				StagingBuffer indices;
			} stagingBuffers;

			// Vertex buffer
			vk::BufferCreateInfo vertexBufferInfo;
			vertexBufferInfo.size = vertexBufferSize;
			vertexBufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			// Copy to host visible staging buffer
			stagingBuffers.vertices.buffer = device.createBuffer(vertexBufferInfo);
			memReqs = device.getBufferMemoryRequirements(stagingBuffers.vertices.buffer);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			stagingBuffers.vertices.memory = device.allocateMemory(memAlloc);
			// Map and copy
			data = device.mapMemory(stagingBuffers.vertices.memory, 0, VK_WHOLE_SIZE);
			memcpy(data, vertexBuffer.data(), vertexBufferSize);
			device.unmapMemory(stagingBuffers.vertices.memory);
			device.bindBufferMemory(stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0);
			// Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
			vertexBufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
			vertices.buffer= device.createBuffer(vertexBufferInfo);
			memReqs = device.getBufferMemoryRequirements(vertices.buffer);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
			vertices.memory = device.allocateMemory(memAlloc);
			device.bindBufferMemory(vertices.buffer, vertices.memory, 0);

			// Index buffer
			vk::BufferCreateInfo indexbufferInfo;
			indexbufferInfo.size = indexBufferSize;
			indexbufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			// Copy to host visible staging buffer
			stagingBuffers.indices.buffer = device.createBuffer(indexbufferInfo);
			memReqs = device.getBufferMemoryRequirements(stagingBuffers.indices.buffer);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			stagingBuffers.indices.memory = device.allocateMemory(memAlloc);
			// Map and copy
			data = device.mapMemory(stagingBuffers.indices.memory, 0, VK_WHOLE_SIZE);
			memcpy(data, indexBuffer.data(), indexBufferSize);
			device.unmapMemory(stagingBuffers.indices.memory);
			device.bindBufferMemory(stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0);
			// Create a device local buffer to which the (host local) index data will be copied and which will be used for rendering
			indexbufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
			indices.buffer = device.createBuffer(indexbufferInfo);
			memReqs = device.getBufferMemoryRequirements(indices.buffer);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
			indices.memory = device.allocateMemory(memAlloc);
			device.bindBufferMemory(indices.buffer, indices.memory, 0);

			// Buffer copies have to be submitted to a queue, so we need a command buffer for them
			// Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies

			vk::CommandBufferAllocateInfo cmdBuferAllocInfo;
			cmdBuferAllocInfo.commandPool = commandPool;
			cmdBuferAllocInfo.commandBufferCount = 1;
			cmdBuferAllocInfo.level = vk::CommandBufferLevel::ePrimary;
			vk::CommandBuffer copyCmd = device.allocateCommandBuffers(cmdBuferAllocInfo)[0];

			copyCmd.begin(vk::CommandBufferBeginInfo());

			// Put buffer region copies into command buffer
			VkBufferCopy copyRegion = {};

			// Vertex buffer
			copyRegion.size = vertexBufferSize;
			vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, vertices.buffer, 1, &copyRegion);
			// Index buffer
			copyRegion.size = indexBufferSize;
			vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buffer,	1, &copyRegion);

			// End command buffer and submit
			copyCmd.end();

			vk::SubmitInfo submitInfo;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &copyCmd;

			queue.submit(submitInfo, VK_NULL_HANDLE);
			queue.waitIdle();

			device.freeCommandBuffers(commandPool, copyCmd);

			// Destroy staging buffers
			device.destroyBuffer(stagingBuffers.vertices.buffer);
			device.freeMemory(stagingBuffers.vertices.memory);
			device.destroyBuffer(stagingBuffers.indices.buffer);
			device.freeMemory(stagingBuffers.indices.memory);
		}
		/*
		else
		{
			// Don't use staging
			// Create host-visible buffers only and use these for rendering. This is not advised and will usually result in lower rendering performance

			// Vertex buffer
			VkBufferCreateInfo vertexBufferInfo = {};
			vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			vertexBufferInfo.size = vertexBufferSize;
			vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

			// Copy vertex data to a buffer visible to the host
			VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfo, nullptr, &vertices.buffer));
			vkGetBufferMemoryRequirements(device, vertices.buffer, &memReqs);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &vertices.memory));
			VK_CHECK_RESULT(vkMapMemory(device, vertices.memory, 0, memAlloc.allocationSize, 0, &data));
			memcpy(data, vertexBuffer.data(), vertexBufferSize);
			vkUnmapMemory(device, vertices.memory);
			VK_CHECK_RESULT(vkBindBufferMemory(device, vertices.buffer, vertices.memory, 0));

			// Index buffer
			VkBufferCreateInfo indexbufferInfo = {};
			indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			indexbufferInfo.size = indexBufferSize;
			indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

			// Copy index data to a buffer visible to the host
			VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferInfo, nullptr, &indices.buffer));
			vkGetBufferMemoryRequirements(device, indices.buffer, &memReqs);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &indices.memory));
			VK_CHECK_RESULT(vkMapMemory(device, indices.memory, 0, indexBufferSize, 0, &data));
			memcpy(data, indexBuffer.data(), indexBufferSize);
			vkUnmapMemory(device, indices.memory);
			VK_CHECK_RESULT(vkBindBufferMemory(device, indices.buffer, indices.memory, 0));
		}
		*/

		// Vertex input binding
		vertices.inputBinding.binding = 0;				
		vertices.inputBinding.stride = sizeof(Vertex);
		vertices.inputBinding.inputRate = vk::VertexInputRate::eVertex;

		// Inpute attribute binding describe shader attribute locations and memory layouts
		// These match the following shader layout (see triangle.vert):
		//	layout (location = 0) in vec3 inPos;
		//	layout (location = 1) in vec3 inColor;

		vertices.inputAttributes.push_back(vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)));
		vertices.inputAttributes.push_back(vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)));

		// Assign to the vertex input state used for pipeline creation
		vertices.inputState.vertexBindingDescriptionCount = 1;
		vertices.inputState.pVertexBindingDescriptions = &vertices.inputBinding;
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.inputAttributes.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.inputAttributes.data();
	}

	// Create descriptor sets, layouts, etc.
	void createDescriptors()
	{
		// List of descriptor types that will be used by this tutorial
		std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
		descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1));

		// Create the global descriptor pool, all descriptors usedin this example are allocated from this pool
		vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
		descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
		descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();
		descriptorPoolCreateInfo.maxSets = 1;

		descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

		// Descriptor set layout

		// Binding 0: Uniform buffer (Vertex shader)
		std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
		layoutBindings.push_back(vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex));

		vk::DescriptorSetLayoutCreateInfo descriptorLayout;
		descriptorLayout.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		descriptorLayout.pBindings = layoutBindings.data();

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		// Descriptor sets

		vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;
		descriptorSetAllocInfo.descriptorPool = descriptorPool;
		descriptorSetAllocInfo.descriptorSetCount = 1;
		descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;

		descriptorSet = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		writeDescriptorSets.push_back(vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor));

		device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);
	}

	// Create the frame buffers
	void createFrameBuffers()
	{
		// Create depth and stencil buffer attachments for the frame buffers

		// Create an optimal image used as the depth stencil attachment
		vk::ImageCreateInfo image;
		image.imageType = vk::ImageType::e2D;
		image.format = vk::Format::eD24UnormS8Uint;
		// Use example's height and width
		image.extent = { windowSize.width, windowSize.height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		image.initialLayout = vk::ImageLayout::eUndefined;
		depthStencil.image = device.createImage(image);

		// Allocate memory for the image (device local) and bind it to our image
		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;
		memReqs = device.getImageMemoryRequirements(depthStencil.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		depthStencil.memory = device.allocateMemory(memAlloc);
		device.bindImageMemory(depthStencil.image, depthStencil.memory, 0);

		// Create a view for the depth stencil image
		// Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
		// This allows for multiple views of one image with differing ranges (e.g. for different layers)
		vk::ImageViewCreateInfo depthStencilView;
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = vk::Format::eD24UnormS8Uint;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = depthStencil.image;
		depthStencil.view = device.createImageView(depthStencilView);

		// Create a frame buffer for every image in the swapchain
		frameBuffers.resize(swapChain->images.size());
		for (size_t i = 0; i < frameBuffers.size(); i++)
		{
			std::array<vk::ImageView, 2> attachments;										
			attachments[0] = swapChain->buffers[i].view;								// Color attachment is the view of the swapchain image			
			attachments[1] = depthStencil.view;											// Depth/Stencil attachment is the same for all frame buffers			

			vk::FramebufferCreateInfo frameBufferCreateInfo;
			// All frame buffers use the same renderpass setup
			frameBufferCreateInfo.renderPass = renderPass;
			frameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCreateInfo.pAttachments = attachments.data();
			frameBufferCreateInfo.width = windowSize.width;
			frameBufferCreateInfo.height = windowSize.height;
			frameBufferCreateInfo.layers = 1;
			// Create the framebuffer
			frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
		}
	}

	// Create the render pass
	void createRenderPass()
	{
		// This example will use a single render pass with one subpass

		// Descriptors for the attachments used by this renderpass
		std::array<vk::AttachmentDescription, 2> attachments;

		// Color attachment
		attachments[0].format = swapChain->colorFormat;
		attachments[0].samples = vk::SampleCountFlagBits::e1;							// We don't use multi sampling in this example
		attachments[0].loadOp = vk::AttachmentLoadOp::eClear;							// Clear this attachment at the start of the render pass
		attachments[0].storeOp = vk::AttachmentStoreOp::eStore;							// Keep it's contents after the render pass is finished (for displaying it)
		attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;					// We don't use stencil, so don't care for load
		attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;				// Same for store
		attachments[0].initialLayout = vk::ImageLayout::eUndefined;						// Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;					// Layout to which the attachment is transitioned when the render pass is finished
																						// As we want to present the color buffer to the swapchain, we transition to PRESENT_KHR	
		// Depth attachment
		attachments[1].format = vk::Format::eD24UnormS8Uint;								
		attachments[1].samples = vk::SampleCountFlagBits::e1;
		attachments[1].loadOp = vk::AttachmentLoadOp::eClear;							// Clear depth at start of first subpass
		attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;						// We don't need depth after render pass has finished (DONT_CARE may result in better performance)
		attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;					// No stencil
		attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;				// No Stencil
		attachments[1].initialLayout = vk::ImageLayout::eUndefined;						// Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;	// Transition to depth/stencil attachment

		// Setup attachment references
		vk::AttachmentReference colorReference;
		colorReference.attachment = 0;													// Attachment 0 is color
		colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;				// Attachment layout used as color during the subpass

		vk::AttachmentReference depthReference;
		depthReference.attachment = 1;													// Attachment 1 is color
		depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;		// Attachment used as depth/stemcil used during the subpass

		// Setup a single subpass reference
		vk::SubpassDescription subpassDescription;
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;			
		subpassDescription.colorAttachmentCount = 1;									// Subpass uses one color attachment
		subpassDescription.pColorAttachments = &colorReference;							// Reference to the color attachment in slot 0
		subpassDescription.pDepthStencilAttachment = &depthReference;					// Reference to the depth attachment in slot 1
		subpassDescription.inputAttachmentCount = 0;									// Input attachments can be used to sample from contents of a previous subpass
		subpassDescription.pInputAttachments = nullptr;									// (Input attachments not used by this example)
		subpassDescription.preserveAttachmentCount = 0;									// Preserved attachments can be used to loop (and preserve) attachments through subpasses
		subpassDescription.pPreserveAttachments = nullptr;								// (Preserve attachments not used by this example)
		subpassDescription.pResolveAttachments = nullptr;								// Resolve attachments are resolved at the end of a sub pass and can be used for e.g. multi sampling

		// Setup subpass dependencies
		// These will add the implicit ttachment layout transitionss specified by the attachment descriptions
		// The actual usage layout is preserved through the layout specified in the attachment reference		
		// Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
		// srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
		// Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
		std::array<vk::SubpassDependency, 2> dependencies;

		// First dependency at the start of the renderpass
		// Does the transition from final to initial layout 
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;								// Producer of the dependency 
		dependencies[0].dstSubpass = 0;													// Consumer is our single subpass that will wait for the execution depdendency
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Second dependency at the end the renderpass
		// Does the transition from the initial to the final layout
		dependencies[1].srcSubpass = 0;													// Producer of the dependency is our single subpass
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;								// Consumer are all commands outside of the renderpass
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Create the actual renderpass
		vk::RenderPassCreateInfo renderPassInfo;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());		// Number of attachments used by this render pass
		renderPassInfo.pAttachments = attachments.data();								// Descriptions of the attachments used by the render pass
		renderPassInfo.subpassCount = 1;												// We only use one subpass in this example
		renderPassInfo.pSubpasses = &subpassDescription;								// Description of that subpass
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());	// Number of subpass dependencies
		renderPassInfo.pDependencies = dependencies.data();								// Subpass dependencies used by the render pass

		renderPass = device.createRenderPass(renderPassInfo);
	}

	void createGraphicsPipeline()
	{
		// Pipeline cache
		pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

		// Pipeline layout based on default descriptor set layout
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		pPipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Create the graphics pipeline used in this example
		// Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
		// A pipeline is then stored and hashed on the GPU making pipeline changes very fast
		// Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
		pipelineCreateInfo.layout = pipelineLayout;
		pipelineCreateInfo.renderPass = renderPass;

		// Construct the differnent states making up the pipeline
		
		// Input assembly state describes how primitives are assembled
		// This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState;
		inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;

		// Rasterization state
		vk::PipelineRasterizationStateCreateInfo rasterizationState;
		rasterizationState.polygonMode = vk::PolygonMode::eFill;
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		rasterizationState.frontFace = vk::FrontFace::eCounterClockwise;
		rasterizationState.depthClampEnable = VK_FALSE;
		rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		rasterizationState.depthBiasEnable = VK_FALSE;
		rasterizationState.lineWidth = 1.0f;

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used
		vk::PipelineColorBlendAttachmentState blendAttachmentState;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		blendAttachmentState.blendEnable = VK_FALSE;
		vk::PipelineColorBlendStateCreateInfo colorBlendState;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = &blendAttachmentState;

		// Viewport state sets the number of viewports and scissor used in this pipeline
		// Note: This is actually overriden by the dynamic states (see below)
		vk::PipelineViewportStateCreateInfo viewportState;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Enable dynamic states
		// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
		// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
		// For this example we will set the viewport and scissor using dynamic states
		std::vector<vk::DynamicState> dynamicStateEnables;
		dynamicStateEnables.push_back(vk::DynamicState::eViewport);
		dynamicStateEnables.push_back(vk::DynamicState::eScissor);
		vk::PipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.pDynamicStates = dynamicStateEnables.data();
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

		// Depth and stencil state containing depth and stencil compare and test operations
		// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
		vk::PipelineDepthStencilStateCreateInfo depthStencilState;
		depthStencilState.depthTestEnable = VK_TRUE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
		depthStencilState.depthBoundsTestEnable = VK_FALSE;
		depthStencilState.back.failOp = vk::StencilOp::eKeep;
		depthStencilState.back.passOp = vk::StencilOp::eKeep;
		depthStencilState.back.compareOp = vk::CompareOp::eAlways;
		depthStencilState.stencilTestEnable = VK_FALSE;
		depthStencilState.front = depthStencilState.back;

		// Multi sampling state
		// This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
		vk::PipelineMultisampleStateCreateInfo multisampleState;
		multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;
		multisampleState.pSampleMask = nullptr;

		// Load shaders
		// Vulkan loads it's shaders from an immediate binary representation called SPIR-V
		// Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderModules.push_back(loadSPIRVShader(getAssetPath() + "shaders/01_triangle/triangle.vert.spv"));
		shaderModules.push_back(loadSPIRVShader(getAssetPath() + "shaders/01_triangle/triangle.frag.spv"));

		shaderStages[0].module = shaderModules[0];
		shaderStages[0].stage = vk::ShaderStageFlagBits::eVertex;
		shaderStages[0].pName = "main";

		shaderStages[1].module = shaderModules[1];
		shaderStages[1].stage = vk::ShaderStageFlagBits::eFragment;
		shaderStages[1].pName = "main";

		// Assign the pipeline states to the pipeline creation info structure
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.renderPass = renderPass;
		pipelineCreateInfo.pDynamicState = &dynamicState;

		// Create rendering pipeline using the specified states
		pipeline = device.createGraphicsPipeline(pipelineCache, pipelineCreateInfo);
	}

	void createUniformBuffers()
	{
		// Prepare and initialize a uniform buffer block containing shader uniforms

		// Vertex shader uniform buffer block
		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;

		vk::BufferCreateInfo bufferCreateInfo;
		bufferCreateInfo.size = sizeof(uboVS);
		bufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

		// Create a new buffer
		uniformDataVS.buffer = device.createBuffer(bufferCreateInfo);
		memReqs = device.getBufferMemoryRequirements(uniformDataVS.buffer);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		uniformDataVS.memory = device.allocateMemory(memAlloc);
		device.bindBufferMemory(uniformDataVS.buffer, uniformDataVS.memory, 0);
		
		// Store information in the uniform's descriptor that is used by the descriptor set
		uniformDataVS.descriptor.buffer = uniformDataVS.buffer;
		uniformDataVS.descriptor.offset = 0;
		uniformDataVS.descriptor.range = sizeof(uboVS);

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Update matrices
		uboVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)windowSize.width / (float)windowSize.height, 0.1f, 256.0f);

		uboVS.viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.modelMatrix = glm::mat4();
		uboVS.modelMatrix = glm::rotate(uboVS.modelMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.modelMatrix = glm::rotate(uboVS.modelMatrix, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.modelMatrix = glm::rotate(uboVS.modelMatrix, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		// Map uniform buffer and update it
		void *pData = device.mapMemory(uniformDataVS.memory, 0, VK_WHOLE_SIZE);
		memcpy(pData, &uboVS, sizeof(uboVS));
		device.unmapMemory(uniformDataVS.memory);
	}

	void renderLoop()
	{
		//destWidth = width;
		//destHeight = height;
#if defined(_WIN32)
		MSG msg;
		while (TRUE)
		{
			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			if (msg.message == WM_QUIT)
			{
				break;
			}

			renderFrame();
		}
#elif defined(__ANDROID__)
		while (1)
		{
			int ident;
			int events;
			struct android_poll_source* source;
			bool destroy = false;

			focused = true;

			while ((ident = ALooper_pollAll(focused ? 0 : -1, NULL, &events, (void**)&source)) >= 0)
			{
				if (source != NULL)
				{
					source->process(androidApp, source);
				}
				if (androidApp->destroyRequested != 0)
				{
					LOGD("Android app destroy requested");
					destroy = true;
					break;
				}
			}

			// App destruction requested
			// Exit loop, example will be destroyed in application main
			if (destroy)
			{
				break;
			}

			// Render frame
			if (prepared)
			{
				renderFrame();
			}
		}
#elif defined(__linux__)
		xcb_flush(connection);
		while (!quit)
		{
			xcb_generic_event_t *event;
			while ((event = xcb_poll_for_event(connection)))
			{
				handleEvent(event);
				free(event);
			}
			renderFrame();
		}
#endif
		// Flush device to make sure all resources can be freed if application is about to close
		vkDeviceWaitIdle(device);
	}

	void viewChanged()
	{
		// This function is called by the base example class each time the view is changed by user input
		updateUniformBuffers();
	}
};

// Main entry point
#if defined(_WIN32)
// Windows entry point
#define VULKAN_EXAMPLE_MAIN()
VulkanExample *vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (vulkanExample != NULL)
	{
	//	vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
	}

	switch (uMsg)
	{
	case WM_CLOSE:
		DestroyWindow(hWnd);
		PostQuitMessage(0);
		break;
	}

	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow)
{
	vulkanExample = new VulkanExample(hInstance, WndProc);
	delete(vulkanExample);																			
	return 0;																						
}																									
#elif defined(__ANDROID__)
// Android entry point
// A note on app_dummy(): This is required as the compiler may otherwise remove the main entry point of the application
#define VULKAN_EXAMPLE_MAIN()																		
VulkanExample *vulkanExample;																		
void android_main(android_app* state)																
{																									
	app_dummy();																					
	vulkanExample = new VulkanExample();															
	state->userData = vulkanExample;																
	state->onAppCmd = VulkanExample::handleAppCommand;												
	state->onInputEvent = VulkanExample::handleAppInput;											
	vulkanExample->androidApp = state;																
	vulkanExample->renderLoop();																	
	delete(vulkanExample);																			
}
#elif defined(_DIRECT2DISPLAY)
// Linux entry point with direct to display wsi
// todo: extract command line arguments
#define VULKAN_EXAMPLE_MAIN()																		
VulkanExample *vulkanExample;																		
static void handleEvent()                                											
{																									
}																									
int main(const int argc, const char *argv[])													    
{																									
	vulkanExample = new VulkanExample();															
	vulkanExample->initSwapchain();																	
	vulkanExample->prepare();																		
	vulkanExample->renderLoop();																	
	delete(vulkanExample);																			
	return 0;																						
}
#elif defined(__linux__)
// Linux entry point
// todo: extract command line arguments
#define VULKAN_EXAMPLE_MAIN()																		
VulkanExample *vulkanExample;																		
static void handleEvent(const xcb_generic_event_t *event)											
{																									
	if (vulkanExample != NULL)																		
	{																								
		vulkanExample->handleEvent(event);															
	}																								
}																									
int main(const int argc, const char *argv[])													    
{																									
	vulkanExample = new VulkanExample();															
	vulkanExample->setupWindow();					 												
	vulkanExample->initSwapchain();																	
	vulkanExample->prepare();																		
	vulkanExample->renderLoop();																	
	delete(vulkanExample);																			
	return 0;																						
}
#endif
