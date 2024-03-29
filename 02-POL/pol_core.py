import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import Classifier
from pol.pol_config import POLConfig as Hub

import pol_du as du



# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, 'data')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.9

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.input_shape = [100, 100, 3]

th.num_classes = 3
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5
th.shuffle = True


th.print_cycle = 2
th.updates_per_round = 50
th.validation_per_round = 5
th.val_batch_size = 1
th.eval_batch_size = 1

th.export_tensors_upon_validation = True

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True


def activate():
  if 'deactivate' in th.developer_code: return

  # Load data
  train_set, val_set, test_set = du.load_data()

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th)
  else:
    # Evaluate on test set
    model.evaluate_image_sets(train_set, val_set, test_set,
                              visualize_last_false_set=False)

  # End
  model.shutdown()
  console.end()


if __name__ == "__main__":
  print(ROOT)