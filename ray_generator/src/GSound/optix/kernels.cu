#include <optix_stubs.h>
#include "gsound_optix_shared.h"

extern "C" {
__constant__ LaunchParams params;
}

// PCG random number generator
static __forceinline__ __device__ unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float rnd(unsigned int& seed)
{
    seed = pcg_hash(seed);
    return (float)seed / (float)UINT_MAX;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int rayIdx = idx.x;
    if(rayIdx >= params.numRays) return;

    unsigned int seed = params.seed + rayIdx;

    // Generate uniform direction on sphere
    float u = rnd(seed);
    float v = rnd(seed);
    
    float theta = 2.0f * 3.14159265f * u;
    float phi = acosf(2.0f * v - 1.0f);
    
    float3 direction;
    direction.x = sinf(phi) * cosf(theta);
    direction.y = sinf(phi) * sinf(theta);
    direction.z = cosf(phi);

    float3 origin = params.origin;

    unsigned int p0 = 0; // distance (float as uint)
    unsigned int p1 = 0; // hit object/primitive index

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,              // tmin
        1e16f,               // tmax
        0.0f,                // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        1,                   // SBT stride
        0,                   // missSBTIndex
        p0, p1
    );

    params.out_distances[rayIdx] = __uint_as_float(p0);
    params.out_directions[rayIdx] = direction;
    params.out_hit_indices[rayIdx] = (int)p1;
}

extern "C" __global__ void __closesthit__ch()
{
    float t_hit = optixGetRayTmax();
    unsigned int primitiveId = optixGetPrimitiveIndex();
    
    optixSetPayload_0( __float_as_uint(t_hit) );
    optixSetPayload_1( primitiveId );
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0( __float_as_uint(-1.0f) );
    optixSetPayload_1( (unsigned int)-1 );
}
