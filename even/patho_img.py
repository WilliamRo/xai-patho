from roma import Nomear

import cv2
import numpy as np
import os


class PathoImage(Nomear):

  def __init__(self, path):
    """
        path (str): fullpath to patch
    """
    self.path = path
    self.im = cv2.imread(path, cv2.COLOR_BGR2RGB)

    basename = os.path.splitext(os.path.basename(path))[0]
    classification, subject, number, dataset_name = basename.split('_')
    self.subject = '_'.join([subject, number])
    self.classification = classification
    self.data_name = dataset_name


  # region: Properties

  @Nomear.property()
  def binary_image(self):
    gray_image = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
    return binary_image

  @Nomear.property()
  def reinhard(self):
    import slideflow as sf
    normalizer = sf.norm.autoselect('reinhard', backend='opencv')
    img = normalizer.transform(self.im)
    return img

  # endregion: Properties

  # region: Public Methods

  def get_lego(self, lego_key):
    if lego_key in ('raw', ): return self.im
    if lego_key in ('bin', ): return self.binary_image
    if lego_key in ('rh', ): return self.reinhard

    return None

  # endregion: Public Methods



if __name__ == '__main__':
  pass
