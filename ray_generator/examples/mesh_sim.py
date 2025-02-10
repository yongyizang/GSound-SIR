import numpy as np
import pygsound as ps
import json

def dict_to_json(d): # for demonstration purposes only, only displays the shape of numpy arrays
    # convert all numpy arrays to lists
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.shape
        elif isinstance(v, dict):
            d[k] = dict_to_json(v)
            
    return json.dumps(d, indent=4)

def main():
    # Simulation using .obj file (and an optional .mtl file)
    ctx = ps.Context()
    ctx.diffuse_count = 20000
    ctx.specular_count = 2000
    ctx.channel_type = ps.ChannelLayoutType.stereo
    
    mesh1 = ps.loadobj("cube.obj")
    scene = ps.Scene()
    scene.setMesh(mesh1)

    src_coord = [1, 1, 0.5]
    lis_coord = [5, 3, 0.5]

    src = ps.Source(src_coord)
    src.radius = 0.01
    src.power = 1.0

    lis = ps.Listener(lis_coord)
    lis.radius = 0.01

    # res = scene.computeIR([src], [lis], ctx)    # you may pass lists of sources and listeners to get N_src x N_lis IRs
    # audio_data = np.array(res['samples'][0][0])     # the IRs are indexed by [i_src, i_lis, i_channel]
    # with WaveWriter('test1.wav', channels=audio_data.shape[0], samplerate=int(res['rate'])) as w1:
    #     w1.write(audio_data)
    #     print("IR using .obj input written to test1.wav.")
    
    res = scene.getPathData([src], [lis], ctx)["path_data"]
    # get the first listener's path data
    path_data = res[0]
    path_data = dict_to_json(path_data)
    # dump as a .json file
    with open("path_data_obj.json", "w") as f:
        f.write(path_data)

    # Simulation using a shoebox definition
    mesh2 = ps.createbox(10, 6, 2, 0.5, 0.1)
    scene = ps.Scene()
    scene.setMesh(mesh2)

    # res = scene.computeIR([src_coord], [lis_coord], ctx)    # use default source and listener settings if you only pass coordinates
    # audio_data = np.array(res['samples'][0][0])
    # with WaveWriter('test2.wav', channels=audio_data.shape[0], samplerate=int(res['rate'])) as w2:
    #     w2.write(audio_data)
    #     print("IR using shoebox input written to test2.wav.")
    
    res = scene.getPathData([src_coord], [lis_coord], ctx)["path_data"]
    path_data = res[0]
    path_data = dict_to_json(path_data)
    with open("path_data_shoebox.json", "w") as f:
        f.write(path_data)

if __name__ == '__main__':
    main()
