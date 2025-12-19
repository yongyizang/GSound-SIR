"""
Test for batch processing and device configuration APIs.
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, 'src/pygsound')
import pygsound as ps


class TestDeviceConfiguration(unittest.TestCase):
    """Test device configuration APIs."""
    
    def test_device_enum(self):
        """Test that device enum values exist."""
        self.assertEqual(ps.Device.CPU.value, 0)
        self.assertEqual(ps.Device.GPU.value, 1)
        self.assertEqual(ps.Device.AUTO.value, 2)
    
    def test_set_get_device(self):
        """Test setting and getting device."""
        original = ps.get_device()
        
        ps.set_device(ps.Device.CPU)
        self.assertEqual(ps.get_device(), ps.Device.CPU)
        
        ps.set_device(ps.Device.GPU)
        self.assertEqual(ps.get_device(), ps.Device.GPU)
        
        ps.set_device(ps.Device.AUTO)
        self.assertEqual(ps.get_device(), ps.Device.AUTO)
        
        # Restore original
        ps.set_device(original)
    
    def test_gpu_available(self):
        """Test GPU availability check."""
        available = ps.is_gpu_available()
        self.assertIsInstance(available, bool)
        print(f"GPU Available: {available}")
    
    def test_effective_device(self):
        """Test effective device resolution."""
        effective = ps.get_effective_device()
        self.assertIn(effective, [ps.Device.CPU, ps.Device.GPU])
        print(f"Effective Device: {effective}")
    
    def test_device_info(self):
        """Test device info string."""
        info = ps.device_info()
        self.assertIsInstance(info, str)
        self.assertIn("Device Configuration", info)
        print(info)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing APIs."""
    
    def setUp(self):
        """Set up mesh and context for tests."""
        self.mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
        self.ctx = ps.Context()
        self.ctx.diffuse_count = 500
        self.ctx.specular_count = 100  # Must set this to avoid hanging
        self.ctx.threads_count = 1  # Single thread for deterministic results
    
    def test_batch_process_multiple_configs(self):
        """Test batch processing with multiple source/listener configurations."""
        # Create batch of 5 different configurations
        source_positions = [
            [[2.0, 2.0, 2.0]],  # Config 1
            [[3.0, 3.0, 3.0]],  # Config 2
            [[4.0, 4.0, 4.0]],  # Config 3
            [[2.0, 5.0, 2.0]],  # Config 4
            [[5.0, 2.0, 5.0]],  # Config 5
        ]
        
        listener_positions = [
            [[8.0, 8.0, 8.0]],  # Config 1
            [[7.0, 7.0, 7.0]],  # Config 2
            [[6.0, 6.0, 6.0]],  # Config 3
            [[8.0, 5.0, 8.0]],  # Config 4
            [[5.0, 8.0, 5.0]],  # Config 5
        ]
        
        print("\n=== Batch Processing Test ===")
        
        # Process batch on CPU
        results = ps.batch_process(
            self.mesh,
            source_positions,
            listener_positions,
            self.ctx,
            device=ps.Device.CPU,
            num_workers=1
        )
        
        self.assertEqual(len(results), 5)
        
        for i, result in enumerate(results):
            path_data = result["path_data"][0]
            print(f"Config {i+1}: {path_data['num_paths']} paths, energy: {path_data['total_energy']:.6f}")
            self.assertGreater(path_data['num_paths'], 0)
            self.assertGreater(path_data['total_energy'], 0)
    
    def test_batch_processor_class(self):
        """Test BatchProcessor class directly."""
        processor = ps.BatchProcessor()
        processor.set_mesh(self.mesh)
        processor.set_context(self.ctx)
        
        # Create configs
        configs = []
        for i in range(3):
            cfg = ps.SceneConfig()
            cfg.sources = [[2.0 + i, 2.0 + i, 2.0 + i]]
            cfg.listeners = [[8.0 - i, 8.0 - i, 8.0 - i]]
            cfg.energy_percentage = 100.0
            configs.append(cfg)
        
        results = processor.process_batch(configs, device=ps.Device.CPU)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("path_data", result)
    
    def test_batch_cpu_gpu_first_config_identical(self):
        """Test that the first config in batch processing produces identical CPU/GPU results.
        
        Note: Subsequent configs may differ slightly due to shared context state (internal caches).
        For fully identical results, use fresh contexts per config like the rigorous test does.
        """
        if not ps.is_gpu_available():
            self.skipTest("GPU not available")
        
        # Single config to test
        source_positions = [[[2.0, 2.0, 2.0]]]
        listener_positions = [[[8.0, 8.0, 8.0]]]
        
        print("\n=== CPU vs GPU Batch Comparison (Single Config) ===")
        
        # Create fresh mesh and context for CPU batch
        cpu_mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
        cpu_ctx = ps.Context()
        cpu_ctx.diffuse_count = 500
        cpu_ctx.specular_count = 100
        cpu_ctx.threads_count = 1
        
        cpu_results = ps.batch_process(
            cpu_mesh,
            source_positions,
            listener_positions,
            cpu_ctx,
            device=ps.Device.CPU
        )
        
        # Create fresh mesh and context for GPU batch
        gpu_mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
        gpu_ctx = ps.Context()
        gpu_ctx.diffuse_count = 500
        gpu_ctx.specular_count = 100
        gpu_ctx.threads_count = 1
        
        gpu_results = ps.batch_process(
            gpu_mesh,
            source_positions,
            listener_positions,
            gpu_ctx,
            device=ps.Device.GPU
        )
        
        self.assertEqual(len(cpu_results), len(gpu_results))
        
        cpu_data = cpu_results[0]["path_data"][0]
        gpu_data = gpu_results[0]["path_data"][0]
        
        print(f"CPU: {cpu_data['num_paths']} paths, energy: {cpu_data['total_energy']:.6f}")
        print(f"GPU: {gpu_data['num_paths']} paths, energy: {gpu_data['total_energy']:.6f}")
        
        self.assertEqual(cpu_data['num_paths'], gpu_data['num_paths'])
        np.testing.assert_allclose(
            cpu_data['total_energy'], 
            gpu_data['total_energy'],
            rtol=1e-10
        )
        
        print("CPU and GPU results are identical!")


class TestUseGpuParameter(unittest.TestCase):
    """Test that use_gpu parameter still works alongside new Device API."""
    
    def test_use_gpu_parameter(self):
        """Test that use_gpu=True/False still works."""
        mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
        ctx = ps.Context()
        ctx.diffuse_count = 500
        ctx.specular_count = 100  # Must set this to avoid hanging
        ctx.threads_count = 1
        
        scene = ps.Scene()
        scene.setMesh(mesh)
        
        # Test with use_gpu=False
        result_cpu = scene.getPathData(
            [[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]], ctx,
            use_gpu=False
        )
        
        self.assertIn("path_data", result_cpu)
        print(f"use_gpu=False: {result_cpu['path_data'][0]['num_paths']} paths")
        
        # Test with use_gpu=True
        result_gpu = scene.getPathData(
            [[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]], ctx,
            use_gpu=True
        )
        
        self.assertIn("path_data", result_gpu)
        print(f"use_gpu=True: {result_gpu['path_data'][0]['num_paths']} paths")


if __name__ == '__main__':
    unittest.main(verbosity=2)
