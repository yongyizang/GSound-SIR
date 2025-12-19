#include "Device.hpp"
#include <sstream>

#ifdef GSOUND_USE_OPTIX
#include <cuda_runtime.h>
#endif

bool DeviceConfig::isGPUAvailable() const {
#ifdef GSOUND_USE_OPTIX
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

DeviceType DeviceConfig::getEffectiveDevice() const {
    if (m_device == DeviceType::AUTO) {
        return isGPUAvailable() ? DeviceType::GPU : DeviceType::CPU;
    }
    return m_device;
}

std::string DeviceConfig::getDeviceInfo() const {
    std::ostringstream oss;
    
    oss << "Device Configuration:\n";
    oss << "  Selected: ";
    switch (m_device) {
        case DeviceType::CPU: oss << "CPU"; break;
        case DeviceType::GPU: oss << "GPU"; break;
        case DeviceType::AUTO: oss << "AUTO"; break;
    }
    oss << "\n";
    
    oss << "  Effective: ";
    DeviceType effective = getEffectiveDevice();
    switch (effective) {
        case DeviceType::CPU: oss << "CPU"; break;
        case DeviceType::GPU: oss << "GPU"; break;
        default: oss << "Unknown"; break;
    }
    oss << "\n";
    
#ifdef GSOUND_USE_OPTIX
    oss << "  GPU Available: " << (isGPUAvailable() ? "Yes" : "No") << "\n";
    
    if (isGPUAvailable()) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        oss << "  GPU Name: " << prop.name << "\n";
        oss << "  GPU Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    }
#else
    oss << "  OptiX Support: Not compiled\n";
#endif
    
    return oss.str();
}
