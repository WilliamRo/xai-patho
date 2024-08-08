from roma import Nomear
from scipy.ndimage import label

import cv2
import numpy as np



class Immage(Nomear):

  def __init__(self, img: np.ndarray):
    self.img = img

  # region: Intermediate Results

  @Nomear.property()
  def red_channel(self):
    return self.img[:, :, 0]

  @Nomear.property()
  def mask(self):
    MAX_VALUE = 50
    mask = self.red_channel < MAX_VALUE
    return mask

  @Nomear.property()
  def locations(self):
    MIN_SIZE = 200

    locs = []

    labeled_mask, num_features = label(self.mask)
    for i in range(1, num_features + 1):
      loc = np.where(labeled_mask == i)

      # Eliminate small regions
      if loc[0].size < MIN_SIZE: continue

      locs.append((loc[0].mean(), loc[1].mean()))

    return locs

  # endregion: Intermediate Results

  # region: MISC

  def __getitem__(self, item):
    if item == 'raw': return self.img
    else: return self.__getattribute__(item)


  @classmethod
  def read_as_immage_list(cls, img_paths):
    immages = []

    for img_path in img_paths:
      img = cv2.imread(img_path)
      immage = cls(img)
      immages.append(immage)

    return immages

  # endregion: MISC
