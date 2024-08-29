from roma import console
from roma import io
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import FitPackage

import numpy as np
import importlib.util
import sys



# -----------------------------------------------------------------------------
# Import Predictor definition
# -----------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("module.name", "./02=train_and_save_model.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
Predictor = foo.Predictor

# -----------------------------------------------------------------------------
# Load predictor and data (Do not modify codes in this section)
# -----------------------------------------------------------------------------
console.section('Data preparation')

pred_path = r'../data/rad_111x851.pred'
pred: Predictor = io.load_file(pred_path, verbose=True)

data_path = r'../data/radiomics-67x851.omix'
omix = Omix.load(data_path)
# -----------------------------------------------------------------------------
# Test predictor
# -----------------------------------------------------------------------------
console.section('Test predictor')

prob = pred.predict_probabilities(omix.features)
predictions = np.argmax(prob, axis=1)
package = FitPackage.pack(predictions, prob, omix)
package.report()
