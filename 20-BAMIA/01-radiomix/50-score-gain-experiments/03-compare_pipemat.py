from roma import console
from roma import io
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



# ---------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# ---------------------------------------------------------------------------
console.section('Data preparation')

data_path = r'../data/radiomics-111x851.omix'
omix_refit = Omix.load(data_path)

data_path = r'../data/radiomics-67x851.omix'
omix_test = Omix.load(data_path)

SAVE_PATH_1 = r'../data/pipline_flat.omix'
SAVE_PATH_2 = r'../data/pipline_nested.omix'
# ---------------------------------------------------------------------------
# Generate pipeline-1
# ---------------------------------------------------------------------------
# omix_flat = Omix.load(SAVE_PATH_1)
# pi_flat = Pipeline(omix_flat, ignore_warnings=1, save_models=1)
# pi_flat.report()
# pi_flat.plot_matrix()

# pkg = pi_flat.evaluate_best_pipeline(omix)
# pkg.report()

# ---------------------------------------------------------------------------
# Generate pipeline-2
# ---------------------------------------------------------------------------
omix_nested = Omix.load(SAVE_PATH_2)
pi_nested = Pipeline(omix_nested, ignore_warnings=1, save_models=1)
# pi_nested.report()
# pi_nested.plot_matrix(omix_test=omix_test)
# pi_nested.plot_matrix(omix_test=omix_test, omix_refit=omix_refit, verbose=1)

