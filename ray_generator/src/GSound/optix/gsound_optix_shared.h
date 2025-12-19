#pragma once

#include <cuda_runtime.h>
#include <optix.h>

struct RayGenData
{
    // No data needed
};

struct HitGroupData
{
    // Pointer to vertex buffer
    float3* vertices;
    // Pointer to index buffer
    uint3* indices;
};

struct MissData
{
    // Background color or similar
    float3 bgColor;
};

struct LaunchParams
{
    OptixTraversableHandle traversable;
    
    // Ray generation parameters
    // We can launch 1D or 2D. 
    // If 1D: idx.x is the ray index.
    
    int numRays;
    float3 origin; // Single listener for now
    
    // Output buffers (device pointers)
    float* out_distances;
    float3* out_directions; // Hit points or directions?
    float* out_intensities; 
    int* out_hit_indices;   // Which triangle/object was hit
    
    // Random seed for diffuse/Monte Carlo
    unsigned int seed;
};
