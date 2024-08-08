from pictor.plotters import Plotter
from roma import console
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .immage import Immage

import matplotlib.pyplot as plt



class IMMViewer(Plotter):

  def __init__(self, key, cmap='RdBu', pictor=None):
    super().__init__(self.plot, pictor=pictor)
    self.key = key
    self.cmap = cmap

    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('show_locations', False, bool,
                           'Option to show target locations')


  @property
  def zoom_rectangles(self):
    return self.pictor.get_from_pocket('zoom_rectangles', None)


  def plot(self, x: Immage, ax: plt.Axes, fig):
    immage = x
    x = immage[self.key]
    dim = len(x.shape)

    if dim == 3: ax.imshow(x)
    else:
      im = ax.imshow(x, cmap=self.cmap)

      if self.get('color_bar'):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    shape = 'x'.join(map(str, x.shape))
    title = f'{self.key} ({shape})'
    ax.set_title(title)

    # Hide axis
    ax.axis('off')

    # Show locations if required
    if self.get('show_locations'):
      for i, j in immage.locations: ax.plot(j, i, 'go', markersize=5)

    # Zoom to rectangle if necessary
    if self.zoom_rectangles is not None:
      xlim, ylim = self.zoom_rectangles
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)


  def register_shortcuts(self):
    self.register_a_shortcut(
      'c', lambda: self.flip('color_bar'), 'Toggle color bar')
    self.register_a_shortcut(
      'Return', lambda: self.flip('show_locations'), 'Toggle show_locations')
    self.register_a_shortcut('space', self.toggle_froze_lim, 'Toggle froze lim')


  def toggle_froze_lim(self):
    if self.zoom_rectangles is None:
      ax = self.pictor.canvas.axes2D
      xlim, ylim = ax.get_xlim(), ax.get_ylim()
      self.pictor.put_into_pocket('zoom_rectangles', (xlim, ylim))
      console.show_status(f'Froze rectangle: {xlim}, {ylim}')
    else:
      self.pictor.get_from_pocket('zoom_rectangles', put_back=False)
      console.show_status('Unfroze rectangle')

    self.refresh()

