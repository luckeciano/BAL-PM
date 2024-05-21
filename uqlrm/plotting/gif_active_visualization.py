import argparse
import os
from PIL import Image

def make_gif():
    frames = []
    path = './animations/active_batch_visualization_per_batch/batchstate'
    exp = 'batchstate'
    for idx in range(30):
        filename = os.path.join(path, f'test_batch_visualization_{exp}_red_{idx}.png')
        frames.append(Image.open(filename))
    frame_one = frames[0]
    gif_path = os.path.join(path,  f"{exp}.gif")
    frame_one.save(gif_path, format="GIF", append_images=frames, save_all=True, duration=500, loop=0)



make_gif()