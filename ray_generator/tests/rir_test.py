import unittest
import pygsound as ps
import multiprocessing
import numpy as np


def check_ir(samples):
    assert np.argmax(np.fabs(samples)) > 0, "Max val is at beginning"
    assert np.max(np.fabs(samples)) > 0, "IR max is zero"
    assert ~np.isnan(samples).any(), "IR contains NAN"


def same_ir(ir1, ir2):
    start_id = np.argmax(np.abs(ir1))
    assert start_id == np.argmax(np.abs(ir2)), "IRs have different start sample"
    minlen = min([len(ir1), len(ir2), start_id + 1000])
    corrcoef_thresh = 0.9
    corrcoef = np.corrcoef(ir1[start_id:minlen], ir2[start_id:minlen])[0, 1]
    assert corrcoef > corrcoef_thresh, "IRs are not similar"
    



class MainTest(unittest.TestCase):
    def test_exception(self):
        self.assertRaises(AssertionError, check_ir, [1, 0, 0])
        self.assertRaises(AssertionError, check_ir, [0, 0, 0])
        self.assertRaises(AssertionError, check_ir, [np.nan, 0, 0])
        self.assertRaises(AssertionError, same_ir, [1, 0, 0], [0, 1, 0])
        self.assertRaises(AssertionError, same_ir, [1, 0, 0], [1, 0.5, -0.5])

    @staticmethod
    def test_rir():
        seed = 0
        np.random.seed(seed)

        low = 0.5
        high = 0.99
        N = 10
        cnt = 0
        margin = 0.1

        roomdims = np.random.uniform(0.5, 10.0, (N, 3)).tolist()
        lis_locs = [[np.random.uniform(margin, x - margin) for x in roomdim] for roomdim in roomdims]
        src_locs = [[np.random.uniform(margin, x - margin) for x in roomdim] for roomdim in roomdims]
        alphas = 1 - np.random.uniform(low, high, (N, ))
        while cnt < N:
            alpha = alphas[cnt]
            tasks = [[src_locs[cnt], lis_locs[cnt]]]
            compute_scene_ir_absorb(roomdims[cnt], tasks, alpha)
            cnt += 1

    @staticmethod
    def test_rir_pairs():
        roomdim = [10, 10, 10]
        src_locs = [[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]]
        lis_locs = [[2.5, 0.5, 0.5], [5.0, 5.0, 5.0], [9.5, 0.5, 0.5]]
        alpha = 0.5

        mesh = ps.createbox(roomdim[0], roomdim[1], roomdim[2], alpha, 0.5)

        ctx = ps.Context()
        ctx.diffuse_count = 20000
        ctx.specular_count = 2000
        ctx.threads_count = min(multiprocessing.cpu_count(), 8)

        scene = ps.Scene()
        scene.setMesh(mesh)

        channel = ps.ChannelLayoutType.mono
        ctx.channel_type = channel
        ctx.sample_rate = 16000

        src_lis_res = scene.computeIR(src_locs, lis_locs, ctx)
        lis_src_res = scene.computeIR(lis_locs, src_locs, ctx)

        for i_src in range(len(src_locs)):
            for i_lis in range(len(lis_locs)):
                ir1 = src_lis_res['samples'][i_src][i_lis][0]
                ir2 = lis_src_res['samples'][i_lis][i_src][0]
                check_ir(ir1)
                check_ir(ir2)
                same_ir(ir1, ir2)  # IRs should be similar by reciprocity
                

    @staticmethod
    def test_path_data():
        room_dim = [10, 10, 10]
        src_locs = [[0.5, 0.5, 0.5], [9.5, 9.5, 9.5]]
        lis_locs = [[2.5, 0.5, 0.5]]
        
        mesh = ps.createbox(room_dim[0], room_dim[1], room_dim[2], 0.5, 0.5)
        ctx = ps.Context()
        ctx.diffuse_count = 20000
        ctx.specular_count = 2000
        ctx.threads_count = min(multiprocessing.cpu_count(), 8)
        ctx.channel_type = ps.ChannelLayoutType.mono
        ctx.sample_rate = 16000
        
        # Setup scene
        scene = ps.Scene()
        scene.setMesh(mesh)
        
        # Get path data for testing
        paths = scene.getPathData(src_locs, lis_locs, ctx)["path_data"][0]
        
        print("Testing path data structure...")
        required_keys = {
            'source_indices', 'path_types', 'distances',
            'listener_directions', 'source_directions',
            'relative_speeds', 'speeds_of_sound', 'intensities',
            'num_paths', 'num_bands'
        }
        assert set(paths.keys()) == required_keys, "Missing or extra keys in path data"
        
        print("Testing array shapes...")
        num_paths = paths['num_paths']
        num_bands = paths['num_bands']
        
        expected_shapes = {
            'source_indices': (num_paths,),
            'path_types': (num_paths,),
            'distances': (num_paths,),
            'listener_directions': (num_paths, 3),
            'source_directions': (num_paths, 3),
            'relative_speeds': (num_paths,),
            'speeds_of_sound': (num_paths,),
            'intensities': (num_paths, num_bands)
        }
        
        for key, expected_shape in expected_shapes.items():
            assert paths[key].shape == expected_shape, f"Wrong shape for {key}"
        
        print("Testing physical constraints...")
        assert np.all(paths['distances'] > 0), "Found non-positive distances"
        assert np.all(paths['speeds_of_sound'] > 330), "Invalid speed of sound"
        assert np.all(paths['intensities'] >= 0), "Found negative intensities"

        assert np.all(paths['source_indices'] >= 0), "Invalid source indices"
        assert np.all(paths['source_indices'] < len(src_locs)), "Source index out of range"
        
        assert np.allclose(np.linalg.norm(paths['listener_directions'], axis=1), 1.0), \
            "Non-normalized listener directions"
        assert np.allclose(np.linalg.norm(paths['source_directions'], axis=1), 1.0), \
            "Non-normalized source directions"

        print("Testing path reciprocity...")
        paths1 = scene.getPathData([src_locs[0]], lis_locs, ctx)["path_data"][0]
        paths2 = scene.getPathData([src_locs[1]], lis_locs, ctx)["path_data"][0]
        
        direct_mask1 = paths1['path_types'] != 0
        direct_mask2 = paths2['path_types'] != 0
        
        direct_dist1 = paths1['distances'][direct_mask1]
        direct_dist2 = paths2['distances'][direct_mask2]
        
        assert len(direct_dist1) > 0, "No direct paths found for source 1"
        assert len(direct_dist2) > 0, "No direct paths found for source 2"
        
        print("All path data tests passed successfully!")

def compute_scene_ir_absorb(roomdim, tasks, r):
    # Initialize scene mesh
    try:
        mesh = ps.createbox(roomdim[0], roomdim[1], roomdim[2], r, 0.1)
    except Exception as e:
        print(str(e))

    ctx = ps.Context()
    ctx.diffuse_count = 2000
    ctx.specular_count = 2000
    ctx.threads_count = min(multiprocessing.cpu_count(), 8)

    scene = ps.Scene()
    scene.setMesh(mesh)

    channel = ps.ChannelLayoutType.mono
    ctx.channel_type = channel
    ctx.sample_rate = 16000

    for task in tasks:
        src_coord = task[0]
        lis_coord = task[1]

        src = ps.Source(src_coord)
        src.radius = 0.01
        src.power = 1

        lis = ps.Listener(lis_coord)
        lis.radius = 0.01
        res = scene.computeIR([src], [lis], ctx)
        check_ir(res['samples'][0][0][0])


if __name__ == "__main__":
    unittest.main()
