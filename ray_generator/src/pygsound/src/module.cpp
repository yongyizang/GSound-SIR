
#include "Python.hpp"

#include <memory>
#include <string>
#include <sstream>
#include <fftw3.h>

#include <om/omSound.h>
#include <om/omMath.h>

#include "Context.hpp"
#include "SoundMesh.hpp"
#include "Scene.hpp"
#include "SoundSource.hpp"
#include "Listener.hpp"
#include "MicrophoneArrays.hpp"
#include "Device.hpp"
#include "BatchProcessor.hpp"


namespace py = pybind11;
namespace oms = om::sound;
namespace omm = om::math;


PYBIND11_MODULE(pygsound, ps)
{
	fftw_init_threads();
	fftw_plan_with_nthreads( om::CPU::getCount() );

	// Device type enum for CPU/GPU selection
	py::enum_<DeviceType>(ps, "Device")
		.value("CPU", DeviceType::CPU, "Use CPU ray tracing")
		.value("GPU", DeviceType::GPU, "Use GPU (OptiX) ray tracing")
		.value("AUTO", DeviceType::AUTO, "Auto-select (GPU if available, else CPU)")
		.export_values();

	// Global device configuration functions
	ps.def("set_device", [](DeviceType device) {
		DeviceConfig::getInstance().setDevice(device);
	}, "Set the global device for ray tracing", py::arg("device"));

	ps.def("get_device", []() {
		return DeviceConfig::getInstance().getDevice();
	}, "Get the current global device setting");

	ps.def("get_effective_device", []() {
		return DeviceConfig::getInstance().getEffectiveDevice();
	}, "Get the effective device (resolves AUTO to CPU or GPU)");

	ps.def("is_gpu_available", []() {
		return DeviceConfig::getInstance().isGPUAvailable();
	}, "Check if GPU (OptiX) is available");

	ps.def("device_info", []() {
		return DeviceConfig::getInstance().getDeviceInfo();
	}, "Get device configuration info string");

	py::class_< Context, std::shared_ptr< Context > >( ps, "Context")
            .def(py::init<>())
	        .def_property( "specular_count", &Context::getSpecularCount, &Context::setSpecularCount )
            .def_property( "specular_depth", &Context::getSpecularDepth, &Context::setSpecularDepth )
            .def_property( "diffuse_count", &Context::getDiffuseCount, &Context::setDiffuseCount )
            .def_property( "diffuse_depth", &Context::getDiffuseDepth, &Context::setDiffuseDepth )
            .def_property( "threads_count", &Context::getThreadsCount, &Context::setThreadsCount )
            .def_property( "sample_rate", &Context::getSampleRate, &Context::setSampleRate )
            .def_property( "channel_type", &Context::getChannelLayout, &Context::setChannelLayout )
            .def_property( "normalize", &Context::getNormalize, &Context::setNormalize );

	py::class_< SoundMesh, std::shared_ptr< SoundMesh > >( ps, "SoundMesh" )
            .def(py::init<>());

	ps.def( "loadobj", &SoundMesh::loadObj, "A function to load mesh and materials",
            py::arg("_path"), py::arg("_forceabsorp") = -1.0, py::arg("_forcescatter") = -1.0 );
    ps.def( "createbox", py::overload_cast<float, float, float, float, float>(&SoundMesh::createBox),
            "A function to create a simple shoebox mesh", py::arg("_width"), py::arg("_length"), py::arg("_height"),
            py::arg("_absorp") = 0.5, py::arg("_scatter") = 0.1 );
    ps.def( "createbox", py::overload_cast<float, float, float, std::vector<float>, float>(&SoundMesh::createBox),
            "A function to create a simple shoebox mesh", py::arg("_width"), py::arg("_length"), py::arg("_height"),
            py::arg("_absorp"), py::arg("_scatter") = 0.1 );

      py::class_< Scene, std::shared_ptr< Scene > >( ps, "Scene" )
            .def(py::init<>())
            .def( "setMesh", &Scene::setMesh )
            .def( "computeIR", py::overload_cast<std::vector<SoundSource>&, std::vector<Listener>&, Context&>(&Scene::computeIR),
                  "A function to calculate IRs based on pre-defined sources and listeners", 
                  py::arg("_sources"), py::arg("_listeners"), py::arg("_context"))
            .def( "computeIR", py::overload_cast<std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, Context&, float, float, float>(&Scene::computeIR),
                  "A function to calculate IRs based on source and listener locations", 
                  py::arg("_sources"), py::arg("_listeners"), py::arg("_context"), 
                  py::arg("src_radius") = 0.01, py::arg("src_power") = 1.0, py::arg("lis_radius") = 0.01)
            .def( "getPathData", py::overload_cast<std::vector<SoundSource>&, std::vector<Listener>&, Context&, float, size_t, bool>(&Scene::getPathData),
                  "A function to get detailed path data based on pre-defined sources and listeners",
                  py::arg("_sources"), py::arg("_listeners"), py::arg("_context"),
                  py::arg("energy_percentage") = 100.0f, py::arg("max_rays") = 0, py::arg("use_gpu") = false)
            .def( "getPathData", py::overload_cast<std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, Context&, float, float, float, float, size_t, bool>(&Scene::getPathData),
                  "A function to get detailed path data based on source and listener locations",
                  py::arg("_sources"), py::arg("_listeners"), py::arg("_context"),
                  py::arg("src_radius") = 0.01, py::arg("src_power") = 1.0, py::arg("lis_radius") = 0.01,
                  py::arg("energy_percentage") = 100.0f, py::arg("max_rays") = 0, py::arg("use_gpu") = false);


	py::class_< SoundSource, std::shared_ptr< SoundSource > >( ps, "Source" )
            .def( py::init<std::vector<float>>() )
			.def_property( "pos", &SoundSource::getPosition, &SoundSource::setPosition )
			.def_property( "radius", &SoundSource::getRadius, &SoundSource::setRadius  )
			.def_property( "power", &SoundSource::getPower, &SoundSource::setPower  );

	py::class_< Listener, std::shared_ptr< Listener > >( ps, "Listener" )
            .def( py::init<std::vector<float>>() )
			.def_property( "pos", &Listener::getPosition, &Listener::setPosition )
			.def_property( "radius", &Listener::getRadius, &Listener::setRadius  );


	py::enum_<gs::SoundPathFlags::Flag>(ps, "PathType")
			.value("direct", gs::SoundPathFlags::DIRECT)
			.value("transmission", gs::SoundPathFlags::TRANSMISSION)
			.value("specular", gs::SoundPathFlags::SPECULAR)
			.value("diffuse", gs::SoundPathFlags::DIFFUSE)
			.value("diffraction", gs::SoundPathFlags::DIFFRACTION)
			.value("undefined", gs::SoundPathFlags::UNDEFINED)
			.export_values();

	py::enum_< oms::ChannelLayout::Type >( ps, "ChannelLayoutType" )
			.value( "mono", oms::ChannelLayout::MONO )
			.value( "stereo", oms::ChannelLayout::STEREO )
			.value( "binaural", oms::ChannelLayout::BINAURAL )
			.value( "quad", oms::ChannelLayout::QUAD )
			.value( "surround_4", oms::ChannelLayout::SURROUND_4 )
			.value( "surround_5_1", oms::ChannelLayout::SURROUND_5_1 )
			.value( "surround_7_1", oms::ChannelLayout::SURROUND_7_1 )
			.value( "custom", oms::ChannelLayout::CUSTOM )
			.value( "undefined", oms::ChannelLayout::UNDEFINED )
            .export_values();

	// SceneConfig for batch processing
	py::class_<SceneConfig>(ps, "SceneConfig")
		.def(py::init<>())
		.def_readwrite("sources", &SceneConfig::sources)
		.def_readwrite("listeners", &SceneConfig::listeners)
		.def_readwrite("src_radius", &SceneConfig::src_radius)
		.def_readwrite("src_power", &SceneConfig::src_power)
		.def_readwrite("lis_radius", &SceneConfig::lis_radius)
		.def_readwrite("energy_percentage", &SceneConfig::energy_percentage)
		.def_readwrite("max_rays", &SceneConfig::max_rays);

	// BatchProcessor class
	py::class_<BatchProcessor, std::shared_ptr<BatchProcessor>>(ps, "BatchProcessor",
		"Batch processor for running multiple ray tracing simulations efficiently.\n\n"
		"GPU batch processing note: Ray tracing is not GRAM-intensive for typical scenes.\n"
		"A simple box room uses < 1MB of GPU memory for acceleration structures.")
		.def(py::init<>())
		.def("set_mesh", &BatchProcessor::setMesh, "Set the mesh geometry for all scenes")
		.def("set_context", &BatchProcessor::setContext, "Set the simulation context")
		.def("process_batch", &BatchProcessor::processBatch,
			"Process a batch of scene configurations",
			py::arg("configs"),
			py::arg("device") = DeviceType::AUTO,
			py::arg("num_workers") = 0);

	// Convenience function for simple batch processing
	ps.def("batch_process", &BatchProcessor::processBatchSimple,
		"Process multiple scene configurations in batch.\n\n"
		"Args:\n"
		"    mesh: The mesh geometry shared by all scenes\n"
		"    source_positions: List of source position lists for each simulation\n"
		"    listener_positions: List of listener position lists for each simulation\n"
		"    context: Simulation context\n"
		"    device: Device to use (CPU, GPU, or AUTO)\n"
		"    num_workers: Number of parallel workers for CPU (0 = auto)\n"
		"    energy_percentage: Percentage of energy to keep\n"
		"    max_rays: Maximum rays to return (0 = unlimited)\n\n"
		"Returns:\n"
		"    List of path data dictionaries, one per configuration",
		py::arg("mesh"),
		py::arg("source_positions"),
		py::arg("listener_positions"),
		py::arg("context"),
		py::arg("device") = DeviceType::AUTO,
		py::arg("num_workers") = 0,
		py::arg("energy_percentage") = 100.0f,
		py::arg("max_rays") = 0);
}
