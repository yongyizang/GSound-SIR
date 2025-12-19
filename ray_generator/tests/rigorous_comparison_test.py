"""
Rigorous test to verify GPU ray tracing produces identical results to CPU.
This test:
1. Creates fresh scene instances for fair comparison (same random seed state)
2. Compares all path data fields, not just energy
3. Tests multiple scenarios
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, 'src/pygsound')
import pygsound as ps


def create_fresh_scene_and_run(room_dim, src_loc, lis_loc, diffuse_count, specular_count, use_gpu=False):
    """Create a completely fresh scene and run propagation."""
    mesh = ps.createbox(room_dim[0], room_dim[1], room_dim[2], 0.5, 0.5)
    ctx = ps.Context()
    ctx.diffuse_count = diffuse_count
    ctx.specular_count = specular_count
    ctx.threads_count = 1  # Single thread for deterministic results
    
    scene = ps.Scene()
    scene.setMesh(mesh)
    
    result = scene.getPathData(
        [src_loc], [lis_loc], ctx,
        energy_percentage=100.0, max_rays=0,
        use_gpu=use_gpu
    )
    
    return result["path_data"][0]


class RigorousComparisonTest(unittest.TestCase):
    """Rigorous tests comparing CPU and GPU ray tracing results."""
    
    def test_identical_results_simple_box(self):
        """Test that GPU produces identical results to CPU for a simple box room."""
        room_dim = [10, 10, 10]
        src_loc = [2.0, 2.0, 2.0]
        lis_loc = [8.0, 8.0, 8.0]
        
        print("\n=== Test: Simple Box Room ===")
        print(f"Room: {room_dim}, Source: {src_loc}, Listener: {lis_loc}")
        
        # Run CPU version with fresh scene
        print("Running CPU...")
        cpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc, 
            diffuse_count=500, specular_count=100,
            use_gpu=False
        )
        
        # Run GPU version with fresh scene
        print("Running GPU...")
        gpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc,
            diffuse_count=500, specular_count=100,
            use_gpu=True
        )
        
        self._compare_results(cpu_result, gpu_result, "Simple Box")
    
    def test_identical_results_rectangular_room(self):
        """Test with a non-cubic room."""
        room_dim = [15, 8, 4]
        src_loc = [2.0, 4.0, 2.0]
        lis_loc = [13.0, 4.0, 2.0]
        
        print("\n=== Test: Rectangular Room ===")
        print(f"Room: {room_dim}, Source: {src_loc}, Listener: {lis_loc}")
        
        cpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc,
            diffuse_count=500, specular_count=100,
            use_gpu=False
        )
        
        gpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc,
            diffuse_count=500, specular_count=100,
            use_gpu=True
        )
        
        self._compare_results(cpu_result, gpu_result, "Rectangular Room")
    
    def test_identical_results_source_near_wall(self):
        """Test with source near a wall."""
        room_dim = [10, 10, 10]
        src_loc = [0.5, 5.0, 5.0]  # Near wall
        lis_loc = [9.5, 5.0, 5.0]  # Opposite side
        
        print("\n=== Test: Source Near Wall ===")
        print(f"Room: {room_dim}, Source: {src_loc}, Listener: {lis_loc}")
        
        cpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc,
            diffuse_count=500, specular_count=100,
            use_gpu=False
        )
        
        gpu_result = create_fresh_scene_and_run(
            room_dim, src_loc, lis_loc,
            diffuse_count=500, specular_count=100,
            use_gpu=True
        )
        
        self._compare_results(cpu_result, gpu_result, "Source Near Wall")
    
    def test_consistency_multiple_runs(self):
        """Test that multiple GPU runs produce identical results."""
        room_dim = [10, 10, 10]
        src_loc = [3.0, 3.0, 3.0]
        lis_loc = [7.0, 7.0, 7.0]
        
        print("\n=== Test: Consistency Across Multiple Runs ===")
        
        results = []
        for i in range(3):
            result = create_fresh_scene_and_run(
                room_dim, src_loc, lis_loc,
                diffuse_count=500, specular_count=100,
                use_gpu=True
            )
            results.append(result)
            print(f"Run {i+1}: {result['num_paths']} paths, energy: {result['total_energy']:.6f}")
        
        # All runs should produce identical results (since they start with same state)
        for i in range(1, len(results)):
            self.assertEqual(results[0]['num_paths'], results[i]['num_paths'],
                           f"Path count differs between run 1 and run {i+1}")
            np.testing.assert_allclose(
                results[0]['total_energy'], results[i]['total_energy'],
                rtol=1e-10,
                err_msg=f"Energy differs between run 1 and run {i+1}"
            )
        print("All runs produced identical results!")
    
    def _compare_results(self, cpu, gpu, test_name):
        """Compare all fields between CPU and GPU results."""
        print(f"\n--- Comparing {test_name} ---")
        
        # 1. Compare path counts
        print(f"Path count: CPU={cpu['num_paths']}, GPU={gpu['num_paths']}")
        self.assertEqual(cpu['num_paths'], gpu['num_paths'],
                        f"Path counts differ: CPU={cpu['num_paths']}, GPU={gpu['num_paths']}")
        
        # 2. Compare total energy
        energy_diff = abs(cpu['total_energy'] - gpu['total_energy'])
        print(f"Total energy: CPU={cpu['total_energy']:.10f}, GPU={gpu['total_energy']:.10f}")
        print(f"Energy difference: {energy_diff:.2e}")
        np.testing.assert_allclose(
            cpu['total_energy'], gpu['total_energy'],
            rtol=1e-10,
            err_msg="Total energy differs"
        )
        
        # 3. Compare distances
        cpu_dist = np.array(cpu['distances'])
        gpu_dist = np.array(gpu['distances'])
        print(f"Distances shape: CPU={cpu_dist.shape}, GPU={gpu_dist.shape}")
        self.assertEqual(cpu_dist.shape, gpu_dist.shape, "Distance array shapes differ")
        max_dist_diff = np.max(np.abs(cpu_dist - gpu_dist)) if len(cpu_dist) > 0 else 0
        print(f"Max distance difference: {max_dist_diff:.2e}")
        np.testing.assert_allclose(cpu_dist, gpu_dist, rtol=1e-10,
                                  err_msg="Distances differ")
        
        # 4. Compare intensities
        cpu_int = np.array(cpu['intensities'])
        gpu_int = np.array(gpu['intensities'])
        print(f"Intensities shape: CPU={cpu_int.shape}, GPU={gpu_int.shape}")
        self.assertEqual(cpu_int.shape, gpu_int.shape, "Intensity array shapes differ")
        max_int_diff = np.max(np.abs(cpu_int - gpu_int)) if cpu_int.size > 0 else 0
        print(f"Max intensity difference: {max_int_diff:.2e}")
        np.testing.assert_allclose(cpu_int, gpu_int, rtol=1e-10,
                                  err_msg="Intensities differ")
        
        # 5. Compare path types
        cpu_types = np.array(cpu['path_types'])
        gpu_types = np.array(gpu['path_types'])
        print(f"Path types match: {np.array_equal(cpu_types, gpu_types)}")
        np.testing.assert_array_equal(cpu_types, gpu_types,
                                     err_msg="Path types differ")
        
        # 6. Compare listener directions
        cpu_lis_dir = np.array(cpu['listener_directions'])
        gpu_lis_dir = np.array(gpu['listener_directions'])
        print(f"Listener directions shape: CPU={cpu_lis_dir.shape}, GPU={gpu_lis_dir.shape}")
        max_lis_dir_diff = np.max(np.abs(cpu_lis_dir - gpu_lis_dir)) if cpu_lis_dir.size > 0 else 0
        print(f"Max listener direction difference: {max_lis_dir_diff:.2e}")
        np.testing.assert_allclose(cpu_lis_dir, gpu_lis_dir, rtol=1e-10,
                                  err_msg="Listener directions differ")
        
        # 7. Compare source directions
        cpu_src_dir = np.array(cpu['source_directions'])
        gpu_src_dir = np.array(gpu['source_directions'])
        print(f"Source directions shape: CPU={cpu_src_dir.shape}, GPU={gpu_src_dir.shape}")
        max_src_dir_diff = np.max(np.abs(cpu_src_dir - gpu_src_dir)) if cpu_src_dir.size > 0 else 0
        print(f"Max source direction difference: {max_src_dir_diff:.2e}")
        np.testing.assert_allclose(cpu_src_dir, gpu_src_dir, rtol=1e-10,
                                  err_msg="Source directions differ")
        
        print(f"\n✓ {test_name}: ALL COMPARISONS PASSED!")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
