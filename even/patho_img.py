from roma import Nomear

import cv2
import numpy as np
import slideflow as sf


class PathoImage(Nomear):

  def __init__(self, im: np.ndarray, subject=None, type=None):
    self.im = im
    self.subject = subject
    self.type = type


  @Nomear.property()
  def binary_image(self):
    gray_image = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
    return binary_image

  @Nomear.property()
  def reinhard(self):
    normalizer = sf.norm.autoselect('reinhard', backend='opencv')
    img = normalizer.transform(self.im)
    return img


  def get_lego(self, lego_key):
    if lego_key in ('raw', ): return self.im
    if lego_key in ('bin', ): return self.binary_image
    if lego_key in ('rh', ): return self.reinhard

    return None



if __name__ == '__main__':
  pass
