#include "OptixContext.h"

#ifdef GSOUND_USE_OPTIX

// Define the OptiX function table - must be included in exactly one compilation unit
#include <optix_function_table_definition.h>

GSOUND_NAMESPACE_START

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    ::std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << ::std::endl;
}

OptixContext::~OptixContext() {
    // Don't destroy context during program exit - can cause issues with 
    // CUDA driver shutdown order. The OS will clean up resources.
    // if (context) optixDeviceContextDestroy(context);
}

void OptixContext::init() {
    if (initialized) return;

    // Initialize CUDA first
    cudaError_t cudaErr = cudaFree(0);
    if (cudaErr != cudaSuccess) {
        ::std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaErr) << ::std::endl;
        return;
    }
    
    // Get CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        ::std::cerr << "No CUDA devices found" << ::std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    ::std::cout << "Using CUDA device: " << prop.name << ::std::endl;

    // Initialize OptiX
    OptixResult optixRes = optixInit();
    if (optixRes != OPTIX_SUCCESS) {
        ::std::cerr << "optixInit() failed with error code: " << optixRes << ::std::endl;
        ::std::cerr << "OptiX requires NVIDIA driver 525.60.13 or newer." << ::std::endl;
        return;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0; // Use current context
    optixRes = optixDeviceContextCreate(cuCtx, &options, &context);
    if (optixRes != OPTIX_SUCCESS) {
        ::std::cerr << "optixDeviceContextCreate() failed with error code: " << optixRes << ::std::endl;
        return;
    }
    
    initialized = true;
    ::std::cout << "OptiX Context Initialized successfully." << ::std::endl;
}

GSOUND_NAMESPACE_END

#endif
