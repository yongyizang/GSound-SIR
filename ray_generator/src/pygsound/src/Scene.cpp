#include "SoundSource.hpp"
#include "Scene.hpp"
#include "SoundMesh.hpp"
#include "Listener.hpp"
#include "Context.hpp"
#include <iostream>
#include <pybind11/numpy.h>

namespace omm = om::math;
namespace omt = om::time;

Scene::Scene()
{
	m_scene.addObject( &m_soundObject );
}

void
Scene::setMesh( SoundMesh &_mesh )
{
	m_soundObject.setMesh( &_mesh.m_mesh );
	m_soundObject.setTransform( omm::Transform3f( omm::Vector3f( 0.0f, 0.0f, 0.0f ) ) );
}

py::dict
Scene::computeIR( std::vector<SoundSource> &_sources, std::vector<Listener> &_listeners, Context &_context )
{
    int n_src = _sources.size();
    int n_lis = _listeners.size();

    for (SoundSource& p : _sources){
        m_scene.addSource(&p.m_source);
    }
    for (Listener& p : _listeners){
        m_scene.addListener(&p.m_listener);
    }

    if (m_scene.getObjectCount() == 0){
        std::cerr << "object count is zero, cannot propagate sound!" << std::endl;
    }

    propagator.propagateSound(m_scene, _context.internalPropReq(), sceneIR);

    py::list IRPairs(n_src);
    auto rate = _context.getSampleRate();
    for (int i_src = 0; i_src < n_src; ++i_src){
        py::list srcSamples(n_src);
        for (int i_lis = 0; i_lis < n_lis; ++i_lis){
            const gs::SoundSourceIR& sourceIR = sceneIR.getListenerIR(i_lis).getSourceIR(i_src);
            gs::ImpulseResponse result;
            result.setIR(sourceIR, *m_scene.getListener(i_lis), _context.internalIRReq());
            auto numOfChannels = int(result.getChannelCount());

            py::list samples;
            for (int ch = 0; ch < numOfChannels; ch++)
            {
                auto *sample_ch = result.getChannel(ch);
                std::vector<float> samples_ch(sample_ch, sample_ch+result.getLengthInSamples());
                samples.append(samples_ch);
            }
            srcSamples[i_lis] = samples;
        }
        IRPairs[i_src] = srcSamples;
    }

    m_scene.clearSources();
    m_scene.clearListeners();

    py::dict ret;
    ret["rate"] = rate;
    ret["samples"] = IRPairs;   // index by [i_src, i_lis, i_channel]

    return ret;
}

py::dict
Scene::computeIR( std::vector<std::vector<float>> &_sources, std::vector<std::vector<float>> &_listeners, Context &_context,
                        float src_radius, float src_power, float lis_radius)
{
    int n_src = _sources.size();
    int n_lis = _listeners.size();

    // listener propagation is most expensive, so swap them for computation if there are more listeners
    bool swapBuffer = false;
    auto src_pos = std::ref(_sources);
    auto lis_pos = std::ref(_listeners);
    if (n_src < n_lis){
        swapBuffer = true;
        src_pos = std::ref(_listeners);
        lis_pos = std::ref(_sources);
        std::swap(n_src, n_lis);
    }

    std::vector<SoundSource> sources;
    std::vector<Listener> listeners;

    for (auto p : src_pos.get()){
        auto source = new SoundSource(p);
        source->setRadius(src_radius);
        source->setPower(src_power);
        sources.push_back( *source );
    }
    for (auto p : lis_pos.get()){
        auto listener = new Listener(p);
        listener->setRadius(lis_radius);
        listeners.push_back( *listener );
    }

    py::dict _ret = computeIR(sources, listeners, _context);
    py::list IRPairs = _ret["samples"];

    py::dict ret;
    ret["rate"] = _ret["rate"];

    // swap the IR buffer if sources and listeners have been swapped
    if (swapBuffer){
        py::list swapIRPairs;
        for (int i_lis = 0; i_lis < n_lis; ++i_lis) {
            py::list srcSamples;
            for (int i_src = 0; i_src < n_src; ++i_src){
                srcSamples.append(IRPairs[i_src].cast<py::list>()[i_lis]);
            }
            swapIRPairs.append(srcSamples);
        }
        ret["samples"] = swapIRPairs;
    }else{
        ret["samples"] = IRPairs;   // index by [i_src, i_lis, i_channel]
    }

	return ret;
}

py::dict Scene::getPathData(std::vector<SoundSource> &_sources, std::vector<Listener> &_listeners, 
                           Context &_context, float energyPercentage, size_t maxRays, bool use_gpu)
{
    int n_src = _sources.size();
    int n_lis = _listeners.size();

    // Add sources and listeners to scene
    for (SoundSource& p : _sources) {
        m_scene.addSource(&p.m_source);
    }
    for (Listener& p : _listeners) {
        m_scene.addListener(&p.m_listener);
    }

    if (m_scene.getObjectCount() == 0) {
        std::cerr << "object count is zero, cannot get path data!" << std::endl;
        return py::dict();
    }

    // Propagate sound to get paths
    if (use_gpu) {
#ifdef GSOUND_USE_OPTIX
        propagator.propagateSoundOptix(m_scene, _context.internalPropReq(), sceneIR);
#else
        std::cerr << "[Warning] OptiX not available, falling back to CPU." << std::endl;
        propagator.propagateSound(m_scene, _context.internalPropReq(), sceneIR);
#endif
    } else {
        propagator.propagateSound(m_scene, _context.internalPropReq(), sceneIR);
    }

    using om::Size;
    using om::Index;
    using om::UInt32;
    using gsound::Real;
    using om::util::ArrayList;

    py::list pathDataList(n_lis);
    
    for (int i_lis = 0; i_lis < n_lis; ++i_lis) {
        const auto& listenerIR = sceneIR.getListenerIR(i_lis);
        
        Size pathCount = 0;
        Size numBands = 0;
        ArrayList<Index> sourcesPerPath;
        ArrayList<UInt32> pathTypes;
        ArrayList<Real> distances;
        ArrayList<Real> listenerDirections;
        ArrayList<Real> sourceDirections;
        ArrayList<Real> relativeSpeeds;
        ArrayList<Real> speedsOfSound;
        ArrayList<Real> intensities;

        propagator.getPathDataArrays(
            listenerIR,
            pathCount, sourcesPerPath, pathTypes, distances,
            listenerDirections, sourceDirections, relativeSpeeds, 
            speedsOfSound, intensities, numBands
        );

        // Calculate total energy for each path and create index mapping
        std::vector<std::pair<Real, Index>> pathEnergies(pathCount);
        for (Index i = 0; i < pathCount; ++i) {
            Real totalEnergy = 0;
            for (Size band = 0; band < numBands; ++band) {
                totalEnergy += intensities[i * numBands + band];
            }
            pathEnergies[i] = {totalEnergy, i};
        }

        std::sort(pathEnergies.begin(), pathEnergies.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        Real totalEnergy = 0;
        for (const auto& p : pathEnergies) {
            totalEnergy += p.first;
        }

        Size keepPaths = pathCount;
        if (energyPercentage < 100.0f) {
            Real targetEnergy = totalEnergy * (energyPercentage / 100.0f);
            Real accumulatedEnergy = 0;
            for (Size i = 0; i < pathCount; ++i) {
                accumulatedEnergy += pathEnergies[i].first;
                if (accumulatedEnergy >= targetEnergy) {
                    keepPaths = i + 1;
                    break;
                }
            }
        }
        if (maxRays > 0 && maxRays < keepPaths) {
            keepPaths = maxRays;
        }

        ArrayList<Index> filteredSources(keepPaths);
        ArrayList<UInt32> filteredTypes(keepPaths);
        ArrayList<Real> filteredDistances(keepPaths);
        ArrayList<Real> filteredListenerDirs(keepPaths * 3);
        ArrayList<Real> filteredSourceDirs(keepPaths * 3);
        ArrayList<Real> filteredRelativeSpeeds(keepPaths);
        ArrayList<Real> filteredSpeedsOfSound(keepPaths);
        ArrayList<Real> filteredIntensities(keepPaths * numBands);

        for (Size i = 0; i < keepPaths; ++i) {
            Index originalIdx = pathEnergies[i].second;
            
            filteredSources[i] = sourcesPerPath[originalIdx];
            filteredTypes[i] = pathTypes[originalIdx];
            filteredDistances[i] = distances[originalIdx];
            filteredRelativeSpeeds[i] = relativeSpeeds[originalIdx];
            filteredSpeedsOfSound[i] = speedsOfSound[originalIdx];

            for (Size j = 0; j < 3; ++j) {
                filteredListenerDirs[i * 3 + j] = listenerDirections[originalIdx * 3 + j];
                filteredSourceDirs[i * 3 + j] = sourceDirections[originalIdx * 3 + j];
            }
            
            for (Size band = 0; band < numBands; ++band) {
                filteredIntensities[i * numBands + band] = intensities[originalIdx * numBands + band];
            }
        }

        std::vector<ssize_t> dir_shape = {keepPaths, 3};
        std::vector<ssize_t> intensity_shape = {keepPaths, numBands};

        py::dict pathData;
        pathData["source_indices"] = py::array_t<Index>({keepPaths}, filteredSources.getPointer());
        pathData["path_types"] = py::array_t<UInt32>({keepPaths}, filteredTypes.getPointer());
        pathData["distances"] = py::array_t<Real>({keepPaths}, filteredDistances.getPointer());
        pathData["listener_directions"] = py::array_t<Real>(dir_shape, filteredListenerDirs.getPointer());
        pathData["source_directions"] = py::array_t<Real>(dir_shape, filteredSourceDirs.getPointer());
        pathData["relative_speeds"] = py::array_t<Real>({keepPaths}, filteredRelativeSpeeds.getPointer());
        pathData["speeds_of_sound"] = py::array_t<Real>({keepPaths}, filteredSpeedsOfSound.getPointer());
        pathData["intensities"] = py::array_t<Real>(intensity_shape, filteredIntensities.getPointer());
        pathData["num_paths"] = py::int_(keepPaths);
        pathData["num_bands"] = py::int_(numBands);
        pathData["total_energy"] = py::float_(totalEnergy);
        pathData["kept_energy_percentage"] = py::float_(100.0f * 
            std::accumulate(pathEnergies.begin(), pathEnergies.begin() + keepPaths, 0.0f,
                          [](Real sum, const auto& p) { return sum + p.first; }) / totalEnergy);

        pathDataList[i_lis] = pathData;
    }

    m_scene.clearSources();
    m_scene.clearListeners();

    py::dict ret;
    ret["path_data"] = pathDataList;
    return ret;
}

py::dict Scene::getPathData(std::vector<std::vector<float>> &_sources, 
                           std::vector<std::vector<float>> &_listeners, 
                           Context &_context,
                           float src_radius, float src_power, float lis_radius,
                           float energyPercentage, size_t maxRays, bool use_gpu)
{
    int n_src = _sources.size();
    int n_lis = _listeners.size();

    bool swapBuffer = false;
    auto src_pos = std::ref(_sources);
    auto lis_pos = std::ref(_listeners);
    if (n_src < n_lis) {
        swapBuffer = true;
        src_pos = std::ref(_listeners);
        lis_pos = std::ref(_sources);
        std::swap(n_src, n_lis);
    }

    std::vector<SoundSource> sources;
    std::vector<Listener> listeners;

    for (auto p : src_pos.get()) {
        auto source = new SoundSource(p);
        source->setRadius(src_radius);
        source->setPower(src_power);
        sources.push_back(*source);
    }

    for (auto p : lis_pos.get()) {
        auto listener = new Listener(p);
        listener->setRadius(lis_radius);
        listeners.push_back(*listener);
    }

    py::dict _ret = getPathData(sources, listeners, _context, energyPercentage, maxRays, use_gpu);
    py::list pathDataList = _ret["path_data"];

    if (swapBuffer) {
        py::list swappedPathData;
        for (int i_lis = 0; i_lis < n_lis; ++i_lis) {
            py::list srcData;
            for (int i_src = 0; i_src < n_src; ++i_src) {
                srcData.append(pathDataList[i_src].cast<py::list>()[i_lis]);
            }
            swappedPathData.append(srcData);
        }
        _ret["path_data"] = swappedPathData;
    }

    return _ret;
}