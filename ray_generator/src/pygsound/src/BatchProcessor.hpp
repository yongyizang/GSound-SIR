#ifndef INC_BATCH_PROCESSOR_HPP
#define INC_BATCH_PROCESSOR_HPP

#include "Python.hpp"
#include "Scene.hpp"
#include "SoundMesh.hpp"
#include "Context.hpp"
#include "Device.hpp"
#include <vector>
#include <thread>
#include <future>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * Configuration for a single scene in a batch.
 */
struct SceneConfig {
    std::vector<std::vector<float>> sources;
    std::vector<std::vector<float>> listeners;
    float src_radius = 0.01f;
    float src_power = 1.0f;
    float lis_radius = 0.01f;
    float energy_percentage = 100.0f;
    size_t max_rays = 0;
};

/**
 * Batch processor for running multiple ray tracing simulations.
 * 
 * This class enables efficient batch processing of multiple scene configurations,
 * either on CPU (using thread parallelism) or GPU. Each scene in the batch can
 * have different source/listener positions but shares the same mesh geometry.
 * 
 * GPU batch processing note: Ray tracing is not GRAM-intensive for typical scenes.
 * A simple box room uses < 1MB of GPU memory for acceleration structures.
 * Multiple simulations are processed sequentially on GPU but benefit from
 * the GPU's higher ray throughput.
 */
class BatchProcessor {
public:
    BatchProcessor();
    ~BatchProcessor() = default;
    
    /**
     * Set the mesh geometry for all scenes in the batch.
     * All scene configurations in the batch will use this same mesh.
     */
    void setMesh(SoundMesh& mesh);
    
    /**
     * Set the context (ray counts, depth, etc.) for all simulations.
     */
    void setContext(Context& context);
    
    /**
     * Process a batch of scene configurations.
     * 
     * @param configs List of SceneConfig objects defining source/listener positions
     * @param device Device to use (CPU, GPU, or AUTO)
     * @param num_workers Number of parallel workers for CPU processing (0 = auto)
     * @return List of path data dictionaries, one per config
     */
    py::list processBatch(
        const std::vector<SceneConfig>& configs,
        DeviceType device = DeviceType::AUTO,
        size_t num_workers = 0
    );
    
    /**
     * Simplified batch processing with list of source/listener position pairs.
     * 
     * @param source_positions List of source positions for each simulation
     * @param listener_positions List of listener positions for each simulation
     * @param context Simulation context
     * @param device Device to use
     * @param num_workers Number of parallel workers
     * @return List of path data dictionaries
     */
    static py::list processBatchSimple(
        SoundMesh& mesh,
        const std::vector<std::vector<std::vector<float>>>& source_positions,
        const std::vector<std::vector<std::vector<float>>>& listener_positions,
        Context& context,
        DeviceType device = DeviceType::AUTO,
        size_t num_workers = 0,
        float energy_percentage = 100.0f,
        size_t max_rays = 0
    );

private:
    SoundMesh* m_mesh;
    Context* m_context;
    
    py::dict processOne(const SceneConfig& config, bool use_gpu);
};

#endif // INC_BATCH_PROCESSOR_HPP
