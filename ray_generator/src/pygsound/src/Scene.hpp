#ifndef INC_SCENE_HPP
#define INC_SCENE_HPP

#include "Python.hpp"
#include <gsound/gsSoundScene.h>
#include <gsound/gsSoundObject.h>
#include <gsound/gsSoundPropagator.h>
#include <gsound/gsImpulseResponse.h>
#include <pybind11/stl.h>

namespace gs = gsound;
namespace py = pybind11;

class SoundMesh;
class SoundSource;
class Listener;
class Context;

class Scene
{
public:

	Scene();

	void setMesh( SoundMesh &_mesh );

    py::dict computeIR( std::vector<SoundSource> &_sources, std::vector<Listener> &_listeners, Context &_context );
    py::dict computeIR( std::vector<std::vector<float>> &_sources, std::vector<std::vector<float>> &_listeners, Context &_context,
                            float src_radius = 0.01, float src_power = 1.0, float lis_radius = 0.01);

	py::dict getPathData(std::vector<SoundSource> &_sources, 
						std::vector<Listener> &_listeners, 
						Context &_context,
						float energy_percentage = 100.0f,
						size_t max_rays = 0);
	py::dict getPathData(std::vector<std::vector<float>> &_sources, 
						std::vector<std::vector<float>> &_listeners, 
						Context &_context,
						float src_radius = 0.01f, 
						float src_power = 1.0f, 
						float lis_radius = 0.01f,
						float energy_percentage = 100.0f,
						size_t max_rays = 0);

public:

	gs::SoundScene  m_scene;
	gs::SoundObject m_soundObject;
	gs::SoundPropagator propagator;
	gs::SoundSceneIR sceneIR;

};

#endif  // INC_SCENE_HPP
