#ifndef INC_DEVICE_HPP
#define INC_DEVICE_HPP

#include "Python.hpp"
#include <string>

namespace py = pybind11;

/**
 * Device type for ray tracing computation.
 */
enum class DeviceType {
    CPU = 0,    // Use CPU ray tracing
    GPU = 1,    // Use GPU (OptiX) ray tracing
    AUTO = 2    // Auto-select (GPU if available, else CPU)
};

/**
 * Global device configuration singleton.
 */
class DeviceConfig {
public:
    static DeviceConfig& getInstance() {
        static DeviceConfig instance;
        return instance;
    }

    void setDevice(DeviceType device) { m_device = device; }
    DeviceType getDevice() const { return m_device; }
    
    // Check if GPU is available
    bool isGPUAvailable() const;
    
    // Get the effective device (resolves AUTO)
    DeviceType getEffectiveDevice() const;
    
    // Get device info string
    std::string getDeviceInfo() const;

private:
    DeviceConfig() : m_device(DeviceType::CPU) {}
    DeviceConfig(const DeviceConfig&) = delete;
    DeviceConfig& operator=(const DeviceConfig&) = delete;
    
    DeviceType m_device;
};

#endif // INC_DEVICE_HPP
