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
omix = Omix.load(data_path)

SAVE_PATH_2 = r'../data/pipline_nested.omix'
# ---------------------------------------------------------------------------
# Load nested pipeline
# ---------------------------------------------------------------------------
omix_nested = Omix.load(SAVE_PATH_2)
pi_nested = Pipeline(omix_nested, ignore_warnings=1, save_models=1)

pkg = pi_nested.evaluate_best_pipeline(omix, rank=1, omix_refit=omix_refit)
pkg.report()

# ---------------------------------------------------------------------------
# Predict probabilities
# ---------------------------------------------------------------------------
X = omix.features
probs = pkg.predict_proba(X)
console.show_info(f'Probabilities ({omix.target_labels}):')
for i, p in enumerate(probs):
  pid = omix.sample_labels[i]
  target = omix.targets[i]
  tlb = omix.target_labels[target]
  console.supplement(f'{pid}({tlb}): {p}', level=2)

