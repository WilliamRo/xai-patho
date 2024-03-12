from even import PatchAnalyzer, PatchPlotter, PathoImage
from roma import finder

import cv2
import numpy as np
import os



# 1. read image
data_dir = r'../../data/02-POL/video/video_patch_512'

image_paths = finder.walk(data_dir, pattern='*.png')
images = np.array([cv2.imread(p, cv2.COLOR_BGR2RGB) for p in image_paths])
subjects = [os.path.basename(p).split('_')[1] for p in image_paths]
types = [os.path.basename(p).split('_')[0] for p in image_paths]

# 2. convert to PathoImage
path_images = [PathoImage(x[0], x[1], x[2]) for x in
               zip(images, subjects, types)]

# 3. create PatchAnalyzer() and show
pa = PatchAnalyzer()
pa.objects = path_images
pa.labels = [f'{a} {b}' for a, b in zip(types, subjects)]
pa.add_plotter(PatchPlotter('bin'))
# pa.add_plotter(PatchPlotter('rh'))

pa.show()
