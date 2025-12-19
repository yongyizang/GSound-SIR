#include "BatchProcessor.hpp"
#include "SoundSource.hpp"
#include "Listener.hpp"
#include <iostream>
#include <algorithm>

BatchProcessor::BatchProcessor()
    : m_mesh(nullptr), m_context(nullptr)
{
}

void BatchProcessor::setMesh(SoundMesh& mesh) {
    m_mesh = &mesh;
}

void BatchProcessor::setContext(Context& context) {
    m_context = &context;
}

py::dict BatchProcessor::processOne(const SceneConfig& config, bool use_gpu) {
    // Create a fresh scene for this configuration
    Scene scene;
    scene.setMesh(*m_mesh);
    
    // Create sources and listeners
    std::vector<SoundSource> sources;
    std::vector<Listener> listeners;
    
    for (const auto& pos : config.sources) {
        SoundSource src(pos);
        src.setRadius(config.src_radius);
        src.setPower(config.src_power);
        sources.push_back(src);
    }
    
    for (const auto& pos : config.listeners) {
        Listener lis(pos);
        lis.setRadius(config.lis_radius);
        listeners.push_back(lis);
    }
    
    return scene.getPathData(sources, listeners, *m_context, 
                            config.energy_percentage, config.max_rays, use_gpu);
}

py::list BatchProcessor::processBatch(
    const std::vector<SceneConfig>& configs,
    DeviceType device,
    size_t num_workers)
{
    if (!m_mesh || !m_context) {
        throw std::runtime_error("BatchProcessor: mesh and context must be set before processing");
    }
    
    size_t batch_size = configs.size();
    py::list results(batch_size);
    
    // Determine effective device
    DeviceType effective_device = device;
    if (device == DeviceType::AUTO) {
        effective_device = DeviceConfig::getInstance().getEffectiveDevice();
    }
    
    bool use_gpu = (effective_device == DeviceType::GPU);
    
    if (use_gpu) {
        // GPU processing: sequential with fresh scenes for deterministic results
        // Each config gets a fresh scene/propagator to ensure identical results to CPU
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = processOne(configs[i], true);
        }
    } else {
        // CPU processing: sequential for safety
        // Note: Parallel processing is disabled due to GIL/propagator state issues.
        // The internal GSound propagator uses threading for ray tracing anyway,
        // so external parallelism would cause thread contention.
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = processOne(configs[i], false);
        }
    }
    
    return results;
}

py::list BatchProcessor::processBatchSimple(
    SoundMesh& mesh,
    const std::vector<std::vector<std::vector<float>>>& source_positions,
    const std::vector<std::vector<std::vector<float>>>& listener_positions,
    Context& context,
    DeviceType device,
    size_t num_workers,
    float energy_percentage,
    size_t max_rays)
{
    size_t batch_size = source_positions.size();
    
    if (batch_size != listener_positions.size()) {
        throw std::runtime_error("source_positions and listener_positions must have same length");
    }
    
    // Build configs
    std::vector<SceneConfig> configs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        configs[i].sources = source_positions[i];
        configs[i].listeners = listener_positions[i];
        configs[i].energy_percentage = energy_percentage;
        configs[i].max_rays = max_rays;
    }
    
    BatchProcessor processor;
    processor.setMesh(mesh);
    processor.setContext(context);
    
    return processor.processBatch(configs, device, num_workers);
}
