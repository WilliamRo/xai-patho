from pol_core import th
from tframe import Classifier
from tframe import context
from tframe import console
from tframe import mu, tf
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from tframe.layers.hyper.hyper_base import HyperBase
from pol_core import th
from tframe import pedia

import numpy as np



def get_container(flatten=False, fully_conv=False):
  model = Classifier(mark=th.mark)
  shape = [int(th.data_config.split(':')[-1])] * 2 + [3]
  shape = th.input_shape
  if fully_conv: shape = [None, None, 3]
  model.add(mu.Input(sample_shape=shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean, sigma=255.))
  if flatten: model.add(mu.Flatten())
  # model.add(Reshape2Average(batch_size=th.batch_size))

  return model


def finalize(model, fully_conv=False):
  assert isinstance(model, Classifier)

  if fully_conv:
    model.add(mu.Conv2D(filters=th.num_classes, kernel_size=1, use_bias=False))
    # model.add(mu.GlobalAveragePooling2D())
    model.add(AveragePoolingLayer())
    model.add(mu.Activation('softmax'))
  else:
    # model.add(PatchPooling_v1(batch_size=th.batch_size))
    model.add(mu.Dense(num_neurons=th.num_classes, use_bias=False, prune_frac=0.05))
    # model.add(mu.BatchNormalization())
    model.add(mu.Activation('softmax'))


  metrics = ['accuracy', 'loss']
  # Inject distance loss if required
  if 'dloss' in th.developer_code:
    context.customized_loss_f_net = inject_distance_loss

  # Build model
  model.build(metric=metrics, batch_metric='accuracy',
              eval_metric='accuracy')
  return model


def inject_distance_loss(model):
  assert isinstance(model, Classifier)

  # Get flattened tensor
  nets = [n for n in model.children if 'flatten' in str(n)]
  assert len(nets) == 1
  wv = nets[0].output_tensor
  w, v = tf.split(wv, 2, axis=0)
  d = tf.reduce_mean(tf.square(w - v))

  # Put d in update_group
  from tframe.core.slots import TensorSlot
  slot = TensorSlot(model, 'WVLoss')
  slot.plug(d)
  model._update_group.add(slot)

  return [d]


# region: Customized Layers

class Reshape2Average(Layer):

  DEFAULT_PLACEHOLDER_KEY = 'indices'

  def __init__(self, batch_size=None, key=None):
    self.key = self.DEFAULT_PLACEHOLDER_KEY if key is None else key
    self.full_name = 'Reshape2Average'
    self.abbreviation = self.full_name
    self.batch_size = batch_size

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    indices = tf.placeholder(dtype=tf.int32, shape=(None,), name=self.key)
    tf.add_to_collection(pedia.default_feed_dict, indices)

    def branch_train(data, indices, img_num):
      res_list = []
      for i in range(img_num):
        mask = tf.equal(indices, i)
        data_slice = data[mask]
        res_list.append(tf.reduce_mean(data_slice, axis=0))
      res = tf.stack(res_list)
      return res

    return tf.cond(tf.get_collection(pedia.is_training)[0],
                   lambda: branch_train(x, indices, self.batch_size),
                   lambda: branch_train(x, indices, 1))


class PatchPooling_v1(HyperBase):
  full_name = 'PatchPooling_v1'
  abbreviation = 'ppooling'

  def _link(self, x: tf.Tensor):
    # print(x.shape)
    indices = tf.placeholder(dtype=tf.int32, shape=(None,), name='indices')
    tf.add_to_collection(pedia.default_feed_dict, indices)
    batch_size = self._nb_kwargs.get('batch_size')
    # flattened_data = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2] * x.shape[3]))
    flattened_data = x
    g = self.dense(1, flattened_data, scope='gate', activation='sigmoid')
    data = g * flattened_data
    # print(data.shape)

    def branch_train(data, indices, img_num):
      res_list = []
      for i in range(img_num):
        mask = tf.equal(indices, i)
        data_slice = data[mask]
        res_list.append(tf.reduce_sum(data_slice, axis=0))
      res = tf.stack(res_list)
      return res

    res = tf.cond(tf.get_collection(pedia.is_training)[0],
                  lambda: branch_train(data, indices, batch_size),
                  lambda: branch_train(data, indices, 1))
    # print(res.shape)

    # output = tf.reshape(res, shape=(-1,) + tuple(x.shape[1:]))
    # print(output.shape)
    return res


class AveragePoolingLayer(Layer):
  full_name = 'globalavgpool2d'
  abbreviation = 'gap2d'
  def __init__(self, **kwargs):
    super(AveragePoolingLayer, self).__init__(**kwargs)
  def _link(self, input_, **kwargs):
    average_pooled = tf.reduce_mean(input_, axis=[1, 2], keepdims=False)
    return average_pooled

# endregion: Customized Layers
