import imageio
import argparse
import os
from PIL import Image

def make_gif(args):
    frames = []
    ckpts = [1] + list(range(args.min_ckpt, args.max_ckpt, args.steps_ckpt))
    postfix = f"{args.experiment_postfix}_" if args.experiment_postfix is not None else ""

    for ckpt in ckpts:
        filename = os.path.join('./images', args.experiment_name, args.experiment_prefix, f"{postfix}{args.experiment_name}_ckpt_{ckpt}.jpg")
        frames.append(Image.open(filename))
    frame_one = frames[0]
    gif_path = os.path.join('./animations', f"{args.experiment_prefix}_{postfix}_{args.experiment_name}.gif")
    frame_one.save(gif_path, format="GIF", append_images=frames, save_all=True, duration=500, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate gif.')
    parser.add_argument('experiment_prefix', type=str, default='scratch/lucelo/sft/results/', help='Name of the experiment')
    parser.add_argument('experiment_postfix', type=str, default=None, help='Name of the experiment')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('min_ckpt', type=int, default=1, help='Number of ensemble components')
    parser.add_argument('max_ckpt', type=int, help='Number of ensemble components')
    parser.add_argument('steps_ckpt', type=int, help='Number of ensemble components')
    args = parser.parse_args()
    make_gif(args)