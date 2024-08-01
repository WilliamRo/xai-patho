from roma import console
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import FitPackage

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

# Construct your machine learning



# Wrap your predictor into a function
def predict_probabilities(_X):
  assert len(_X.shape) == 2
  prob_1 = np.array([0] * _X.shape[0])
  prob_0 = 1. - prob_1
  return np.stack([prob_0, prob_1], axis=1)
# -----------------------------------------------------------------------------
# Validating model on external data
# -----------------------------------------------------------------------------
console.section('Validating on external data')

prob = predict_probabilities(omix_external.features)
predictions = np.argmax(prob, axis=1)
package = FitPackage.pack(predictions, prob, omix_external)
package.report()
