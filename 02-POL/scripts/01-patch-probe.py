from even import PatchAnalyzer, PatchPlotter, PathoImage
from roma import finder

import cv2
import numpy as np
import os



# 1. read image
data_dir = r'../../data/02-POL/patch_pair/300'

image_paths = finder.walk(data_dir, pattern='*.png')

# 2. convert to PathoImage
path_images = [PathoImage(p) for p in image_paths]

# 3. create PatchAnalyzer() and show
pa = PatchAnalyzer()
pa.objects = path_images
# pa.labels = [f'{a} {b}' for a, b in zip(types, subjects)]
pa.add_plotter(PatchPlotter('bin'))
# pa.add_plotter(PatchPlotter('rh'))

pa.show()
