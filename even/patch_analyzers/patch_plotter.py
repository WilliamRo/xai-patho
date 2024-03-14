from even.patho_img import PathoImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.plotters import Retina

import matplotlib.pyplot as plt
import numpy as np



class PatchPlotter(Retina):

  def __init__(self, lego='raw'):
    super().__init__()
    self.lego_key = lego

    self.new_settable_attr('patch_pair', False, bool, 'patch pair')


  def imshow(self, ax: plt.Axes, x: PathoImage, fig: plt.Figure, label: str):
    if self.get('patch_pair'):
      classes = ['wsi', 'video']
      origin_dn = x.data_name
      new_dn = classes[1 - classes.index(origin_dn)]
      path = x.path.replace(origin_dn, new_dn)
      x = PathoImage(path)
    patho_image = x
    x = patho_image.get_lego(self.lego_key)

    # Clear axes before drawing, and hide axes
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      self.show_text('No image found', ax)
      return
    x = self._check_image(x)

    # Process x if preprocessor is provided
    if callable(self.preprocessor):
      x: np.ndarray = self.preprocessor(x)

    # Do 2D DFT if required
    if self.get('k_space'):
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.get('log'): x: np.ndarray = np.log(x + 1e-10)

    # show title if provided
    if label is not None and self.get('title'): ax.set_title(label)

    # Do auto-scale if required
    if self.get('auto_scale'): x = (x - np.mean(x)) / np.std(x)

    # Show histogram if required
    if self.get('histogram'):
      x = np.ravel(x)
      ax.hist(x=x, bins=50)
      ax.set_axis_on()
      return

    # Show image
    im = ax.imshow(x, cmap=self.get('cmap'), alpha=self.get('alpha'),
                   interpolation=self.get('interpolation'),
                   vmin=self.get('vmin'), vmax=self.get('vmax'))

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

  def register_shortcuts(self):
    self.register_a_shortcut(
      'p', lambda: self.flip('patch_pair'), 'Turn on/off patch pair'
    )