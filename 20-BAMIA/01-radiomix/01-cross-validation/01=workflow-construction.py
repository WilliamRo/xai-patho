from roma import console
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage

import numpy as np



# -----------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# -----------------------------------------------------------------------------
console.section('Data preparation')

data_path = r'../data/radiomics-111x851.omix'

omix = Omix.load(data_path)

omix_internal, omix_external = omix.split(1, 1, shuffle=False)
omix_internal.report()

X = omix_internal.features
y = omix_internal.targets
# -----------------------------------------------------------------------------
# TODO: Import packages and construct machine learning workflow
# -----------------------------------------------------------------------------
console.section('Learning on internal data')

# TODO: attention, this reference is sub-optimal

pi = Pipeline(omix_internal, ignore_warnings=1, save_models=1)
M = 2
pi.create_sub_space('*', repeats=M, show_progress=1, nested=1)
pi.create_sub_space('pval', k=16, repeats=M, show_progress=1, nested=1)
# pi.create_sub_space('lasso', repeats=M, show_progress=1, nested=1)

N = 2
pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1, verbose=0)
pi.fit_traverse_spaces('rf', repeats=N, nested=1, show_progress=1, verbose=0)
# pi.fit_traverse_spaces('svm', repeats=N, nested=1, show_progress=1, verbose=0)

pi.plot_matrix()
# -----------------------------------------------------------------------------
# Validating model on external data
# -----------------------------------------------------------------------------
pkg = pi.evaluate_best_pipeline(omix_external, rank=1, omix_refit=omix_internal)
pkg.report()
