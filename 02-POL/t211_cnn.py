import pol_core as core
import pol_mu as m
from tframe import tf
from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn'
id = 1
def model():
  th = core.th
  model = m.get_container(flatten=False)

  for i, c in enumerate(core.th.archi_string.split('-')):
    if c == 'p':
      model.add(m.mu.MaxPool2D(pool_size=2, strides=2))
    elif c == 'd':
      # c = int(c.replace('d', ''))
      if th.dropout > 0: model.add(m.mu.Dropout(1. - th.dropout))
      # model.add(m.mu.Dense(num_neurons=c))
      # model.add(m.mu.Activation('relu'))
    elif c == 'f':
      # Add flatten layer
      model.add(m.mu.Flatten())
    else:
      c = int(c)
      model.add(m.mu.Conv2D(
        filters=c, kernel_size=th.kernel_size, use_bias=False,
        activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))

  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)

def main(_):
  console.start('{} on polyp task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'wsi:16:300'  # wsi/video
  th.input_shape = [100, 100, 3]

  th.num_classes = 2
  th.val_size = 2
  th.test_size = 2

  th.centralize_data = False
  th.augmentation = True
  th.aug_config = 'rotate'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  prefix = '{}_'.format(date_string())
  suffix = ''
  th.visible_gpu_id = 1

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '8-p-16-p-32-p-64-p-128-p'
  th.kernel_size = 3
  th.dropout = 0.5
  th.activation = 'relu'
  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 16

  th.optimizer = 'adam'
  th.learning_rate = 0.0008

  th.patience = 10

  th.folds_i = 1
  th.folds_k = 5
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = True
  # th.rehearse = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = suffix

  th.gather_classification_results = True
  th.evaluate_test_set = True

  th.mark = prefix + '{}({}){}'.format(model_name, th.num_layers, tail)
  th.gather_summ_name = prefix + summ_name + tail +  '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

