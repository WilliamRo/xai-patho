from roma import console
from roma import io
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



# ---------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# ---------------------------------------------------------------------------
console.section('Data preparation')

data_path = r'../data/radiomics-111x851.omix'
omix_train = Omix.load(data_path)

data_path = r'../data/radiomics-67x851.omix'
omix_test = Omix.load(data_path)
# ---------------------------------------------------------------------------
# TODO
# ---------------------------------------------------------------------------
# omix_train.show_in_explorer()

omix_test.show_in_explorer()
