from imm.immage import Immage
from imm.imm_viewer import IMMViewer

from pictor import Pictor
from roma import console
from roma import finder
from roma import io



# -----------------------------------------------------------------------------
# (1) Configure
# -----------------------------------------------------------------------------
data_path = r'../data'
img_paths = finder.walk(data_path, pattern='*.jpg')[:5]

# -----------------------------------------------------------------------------
# (2) Load images and wrap them into Immage objects
# -----------------------------------------------------------------------------
immages = Immage.read_as_immage_list(img_paths)

# -----------------------------------------------------------------------------
# (3) Show immages in viewer
# -----------------------------------------------------------------------------
p = Pictor(figure_size=(9, 6), toolbar=True)
p.objects = immages

# Add plotters
keys = ['raw', 'red_channel', 'mask']
for k in keys: p.add_plotter(IMMViewer(k))

p.show()
