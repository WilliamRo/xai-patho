from even.patho_img import PathoImage
from even.patch_analyzers.patch_plotter import PatchPlotter
from pictor import Pictor

import cv2
import glob
import os
import numpy as np


class PatchAnalyzer(Pictor):

  def __init__(self, fig_size=(5, 5)):
    super().__init__(title='PatchAnalyzer', figure_size=fig_size)
    self.retina = self.add_plotter(PatchPlotter())



if __name__ == '__main__':
  # 1. read image
  data_dir = r'../../data/02-POL/video/video_patch_512'
  image_paths = glob.glob(os.path.join(data_dir, '*.png'))
  images = np.array([cv2.imread(p, cv2.COLOR_BGR2RGB) for p in image_paths])
  subjects = [os.path.basename(p).split('_')[1] for p in image_paths]
  types = [os.path.basename(p).split('_')[0] for p in image_paths]
  # images = np.random.randint(0, 255, size=(10, 100, 100))

  # 2. convert to PathoImage
  path_images = [PathoImage(x[0], x[1], x[2]) for x in zip(images, subjects, types)]

  # 3. create PatchAnalyzer() and show
  pa = PatchAnalyzer()
  pa.objects = path_images
  pa.labels = [f'{a} {b}' for a,b in zip(types, subjects)]
  pa.add_plotter(PatchPlotter('bin'))
  pa.add_plotter(PatchPlotter('rh'))

  pa.show()
