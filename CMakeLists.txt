cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME VULKAN_TUTORIALS)

project(${NAME})

include_directories("include")
include_directories("external/glm")
include_directories("external/gli")

OPTION(USE_D2D_WSI "Build the project using Direct to Display swapchain" OFF)

# Use FindVulkan module added with CMAKE 3.7
if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan REQUIRED)
endif()

IF(WIN32)
	IF (NOT Vulkan_FOUND)
		find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/vulkan)
		IF (Vulkan_LIBRARY)
			set(Vulkan_FOUND ON)
		ENDIF()
	ENDIF()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
ELSE(WIN32)
	IF (NOT Vulkan_FOUND)
		find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
		IF (Vulkan_LIBRARY)
			set(Vulkan_FOUND ON)
		ENDIF()
	ENDIF()
	find_package(Threads REQUIRED)
IF(USE_D2D_WSI)
	MESSAGE("Using direct to display extension...")
	add_definitions(-D_DIRECT2DISPLAY)
ELSE(USE_D2D_WSI)
	find_package(XCB REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
ENDIF(USE_D2D_WSI)
	# Todo : android?
ENDIF(WIN32)

IF (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
	message(STATUS ${Vulkan_LIBRARY})
ENDIF()

# Set preprocessor defines
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-std=c++11)

file(GLOB SOURCE *.cpp base/*.cpp)

# Function for building single project
function(GENERATE_PROJECT PROJECT_NAME)
	file(GLOB SOURCE *.cpp base/*.cpp ${PROJECT_NAME}/*.cpp)
	if(WIN32)
		add_executable(${PROJECT_NAME} WIN32 ${PROJECT_NAME}/${PROJECT_NAME}.cpp ${SOURCE})
		target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${PROJECT_NAME} ${PROJECT_NAME}/${PROJECT_NAME}.cpp ${SOURCE})
		target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
	endif(WIN32)
endfunction(GENERATE_PROJECT)

# Build all projects
function(GENERATE_PROJECTS)
	foreach(PROJ ${PROJECTS})
		GENERATE_PROJECT(${PROJECTS})
	endforeach(PROJ)
endfunction(GENERATE_PROJECTS)

# Compiler specific stuff
IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF(MSVC)

IF(WIN32)
	# Nothing here (yet)
ELSE(WIN32)
	link_libraries(${XCB_LIBRARIES} ${Vulkan_LIBRARY})
ENDIF(WIN32)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/")

set(PROJECTS
	01_triangle
)

GENERATE_PROJECTS()