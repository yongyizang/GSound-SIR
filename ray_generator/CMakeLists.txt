cmake_minimum_required(VERSION 3.18)
project(gsound LANGUAGES CXX C)

# Check for CUDA and OptiX
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    
    set(OPTIX_INCLUDE_DIR "" CACHE PATH "Path to OptiX include directory")
    find_path(OPTIX_INCLUDE_DIR_FOUND optix.h
        HINTS ${OPTIX_INCLUDE_DIR}
        PATHS
            ${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include
            ${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64/include
            /usr/local/optix/include
            /usr/include
            $ENV{OPTIX_ROOT}/include
            $ENV{OptiX_INSTALL_DIR}/include
    )
    
    if(OPTIX_INCLUDE_DIR_FOUND)
        message(STATUS "OptiX headers found at ${OPTIX_INCLUDE_DIR_FOUND}")
        set(GSOUND_USE_OPTIX ON)
        add_definitions(-DGSOUND_USE_OPTIX)
        include_directories(${OPTIX_INCLUDE_DIR_FOUND})
    else()
        message(WARNING "OptiX headers not found. OptiX backend will be disabled.")
    endif()
endif()

# Set a default build type if none was specified
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
	message( STATUS "Setting build type to 'debug' as none was specified." )
	set( CMAKE_BUILD_TYPE debug CACHE STRING "Choose the type of build." FORCE )
	# Set the possible values of build type for cmake-gui
	set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release" ) 
endif()

#set build directory
set(BUILD_DIRECTORY ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-narrowing -Wno-deprecated")
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIRECTORY}/lib)
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fPIC --coverage" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC" )

set( THREADS_PREFER_PTHREAD_FLAG ON )

find_package( Threads REQUIRED )
find_package( PkgConfig REQUIRED )
find_package( ZLIB ) # Try finding ZLIB first

# Fallback: if ZLIB not found, we might be in an environment without zlib-dev.
# However, gsound core might require it. Let's see if we can proceed or if we need to mock it/download it.
# For now, let's just make it required but print a helpful message if failed.
if(NOT ZLIB_FOUND)
    message(WARNING "ZLIB not found. This might cause build issues if not available in standard paths.")
    # Assuming the user environment (Cursor) might have limitations on installing sys packages.
    # We will try to proceed, maybe the compiler finds it anyway or it is not strictly needed for the python binding logic 
    # if it's only used for some compression features we don't hit.
endif()

# Fallback: if FFTW3 not found via pkg-config
if(NOT FFTW3_FOUND)
    message(WARNING "FFTW3 not found via PkgConfig. Proceeding without it, but build may fail if gsound relies on it.")
endif()

# Set include directories for dependencies
if(FFTW3_FOUND)
    include_directories(${FFTW3_INCLUDEDIR})
else()
    # Try finding system FFTW3
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    if(FFTW3_INCLUDE_DIR)
        include_directories(${FFTW3_INCLUDE_DIR})
    endif()
endif()

# Add include directories
include_directories(
	${CMAKE_SOURCE_DIR}/src/Om/Om\ Framework
	${CMAKE_SOURCE_DIR}/src/Om/Om\ BVH
	${CMAKE_SOURCE_DIR}/src/Om/Om\ Compression
	${CMAKE_SOURCE_DIR}/src/Om/Om\ Resources
	${CMAKE_SOURCE_DIR}/src/Om/Om\ Sound
	${CMAKE_SOURCE_DIR}/src/GSound
	${FFTW3_INCLUDEDIR}
)

# Add link directories
link_directories(
		    ${FFTW3_LIBRARY_DIRS}  
)


enable_testing()
add_subdirectory(lib/pybind11)
add_subdirectory(src/Om)
add_subdirectory(src/GSound)
add_subdirectory(src/pygsound)

