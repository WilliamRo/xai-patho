from roma import console
from roma import io
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



class Predictor:
  def __init__(self, items): self.items = items

  # ---------------------------------------------------------------------------
  # TODO: Define a function to predict probabilities
  # ---------------------------------------------------------------------------
  def predict_probabilities(self, _X):
    assert len(_X.shape) == 2

    pi = self.items[0]
    dr, pkg = pi.get_best_pipeline()

    dummy_targets = [0] * len(_X)
    omix = Omix(_X, targets=dummy_targets)
    omix_reduced = dr.reduce_dimension(omix)
    prob = pkg.predict_proba(omix_reduced.features)

    return prob



if __name__ == '__main__':
  # ---------------------------------------------------------------------------
  # Load data (Do not modify codes in this section)
  # ---------------------------------------------------------------------------
  console.section('Data preparation')

  data_path = r'../data/radiomics-111x851.omix'
  omix = Omix.load(data_path)

  X, Y = omix.features, omix.targets
  # ---------------------------------------------------------------------------
  # TODO: Import packages and construct machine learning workflow
  # ---------------------------------------------------------------------------
  console.section('Learning on internal data')

  pi = Pipeline(omix, ignore_warnings=1, save_models=1)
  M = 2
  pi.create_sub_space('pca', k=20, repeats=M, show_progress=1)
  pi.create_sub_space('lasso', repeats=M, show_progress=1)
  N = 2
  pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1, verbose=0)

  pi.report()

  # Wrap your predictor into a function

  pred = Predictor((pi,))
  # ---------------------------------------------------------------------------
  # Save probability predictor
  # ---------------------------------------------------------------------------
  pred_path = r'../data/rad_111x851.pred'
  io.save_file(pred, pred_path, verbose=True)



