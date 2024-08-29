from roma import console
from roma import io
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



# ---------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# ---------------------------------------------------------------------------
data_path = r'../data/radiomics-111x851.omix'

SAVE_PATH_1 = r'../data/pipline_flat.omix'
SAVE_PATH_2 = r'../data/pipline_nested.omix'
# ---------------------------------------------------------------------------
# Generate pipeline-1
# ---------------------------------------------------------------------------
console.section('Generating pipeline-1 (dr-flat)')

omix = Omix.load(data_path)

pi = Pipeline(omix, ignore_warnings=1, save_models=1)
M = 1
pi.create_sub_space('*', repeats=M)
pi.create_sub_space('pca', k=20, repeats=M)
# pi.create_sub_space('lasso', repeats=M)
N = 2
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1, verbose=0)
pi.fit_traverse_spaces('dt', repeats=N, show_progress=1, verbose=0)
# pi.fit_traverse_spaces('xgb', repeats=N, nested=1, show_progress=1, verbose=0)

pi.plot_matrix()

SAVE_PATH_1 = r'../data/0724-pipline_flat.omix'
omix.save(SAVE_PATH_1)
# ---------------------------------------------------------------------------
# Generate pipeline-2
# ---------------------------------------------------------------------------
exit()

console.section('Generating pipeline-2 (dr-nested)')

omix = Omix.load(data_path)

pi = Pipeline(omix, ignore_warnings=1, save_models=1)
M = 3
pi.create_sub_space('lasso', repeats=M, nested=1)
pi.create_sub_space('pca', k=16, repeats=M, nested=1)
pi.create_sub_space('mrmr', k=16, repeats=M, nested=1)
pi.create_sub_space('rfe', k=16, repeats=M, nested=1)
pi.create_sub_space('pval', k=16, repeats=M, nested=1)
N = 3
pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1, verbose=0)
pi.fit_traverse_spaces('svm', repeats=N, nested=1, show_progress=1, verbose=0)
pi.fit_traverse_spaces('dt', repeats=N, nested=1, show_progress=1, verbose=0)
pi.fit_traverse_spaces('xgb', repeats=N, nested=1, show_progress=1, verbose=0)

omix.save(SAVE_PATH_2)




