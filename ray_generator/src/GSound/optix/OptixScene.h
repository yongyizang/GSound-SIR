#pragma once
#include "../gsound/gsConfig.h"

#ifdef GSOUND_USE_OPTIX

#include "OptixContext.h"
#include "../gsound/gsSoundScene.h"
#include <map>

GSOUND_NAMESPACE_START

class OptixScene {
public:
    OptixScene();
    ~OptixScene();

    // Build or update the acceleration structure for the given scene
    void build(const SoundScene& scene);
    
    OptixTraversableHandle getTraversable() const { return iasHandle; }
    
    // Helper to get vertex buffer for a mesh (needed for hit shaders)
    void* getVertexBuffer(const SoundMesh* mesh) const;
    void* getIndexBuffer(const SoundMesh* mesh) const;

    // Retrieve OptixPipeline and SBT (initialized lazily or explicitly)
    OptixPipeline getPipeline() const { return pipeline; }
    const OptixShaderBindingTable* getSBT() const { return &sbt; }
    void initPipeline(); // Should be called after building IAS if needed, or lazy

private:
    void buildGAS(const SoundMesh* mesh);
    void buildIAS(const SoundScene& scene);
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void buildSBT();
    void cleanup();

    OptixTraversableHandle iasHandle = 0;
    void* d_ias_output_buffer = nullptr;

    // Pipeline Objects
    OptixModule module = nullptr;
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};

    // SBT Buffers (device pointers)
    void* d_raygenRecords = nullptr;
    void* d_missRecords = nullptr;
    void* d_hitgroupRecords = nullptr;

    // Map mesh pointer to its GAS handle and buffer
    struct MeshGAS {
        OptixTraversableHandle handle;
        void* d_buffer;
        void* d_vertices; // float3 array
        void* d_indices;  // uint3 array
        size_t vertexCount;
        size_t triangleCount;
    };
    std::map<const SoundMesh*, MeshGAS> meshGASCache;
};

GSOUND_NAMESPACE_END

#endif
