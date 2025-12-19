#pragma once
#include "../gsound/gsConfig.h"

#ifdef GSOUND_USE_OPTIX

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>

#define OPTIX_CHECK( call )                                                    \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr << "Optix call '" << #call << "' failed: " #call " ("    \
                      << __FILE__ << ":" << __LINE__ << ")\n";                 \
            exit( 1 );                                                         \
        }                                                                      \
    }

#define CUDA_CHECK( call )                                                     \
    {                                                                          \
        cudaError_t res = call;                                                \
        if( res != cudaSuccess )                                               \
        {                                                                      \
            std::cerr << "CUDA call '" << #call << "' failed: " #call " ("     \
                      << __FILE__ << ":" << __LINE__ << ")\n";                 \
            exit( 1 );                                                         \
        }                                                                      \
    }

GSOUND_NAMESPACE_START

class OptixContext {
public:
    static OptixContext& getInstance() {
        static OptixContext instance;
        return instance;
    }

    void init();
    OptixDeviceContext getContext() const { return context; }
    
private:
    OptixContext() = default;
    ~OptixContext();
    
    // Prevent copy
    OptixContext(const OptixContext&) = delete;
    OptixContext& operator=(const OptixContext&) = delete;

    OptixDeviceContext context = nullptr;
    bool initialized = false;
};

GSOUND_NAMESPACE_END

#endif
