from tframe.core import Group
from tframe.core import TensorSlot, OperationSlot
from tframe.layers.common import Input
from tframe.models.model import Model
from tframe.nets.net import Net
from tframe import context
from tframe import DataSet
from tframe import hub as th
from tframe import pedia
from tframe import tf

import numpy as np



class PatchClassifier(Model):
  class Keys:
    WSI_input = 'WSI_input'

  def __init__(self, mark, input_shape):
    # Call parent's constructor
    super().__init__(mark)

    # Define generator and discriminator
    self.Encoder = Net('Encoder')
    self.Discriminator = Net(pedia.Discriminator)
    self.Classifier = Net('Classifier')
    # Alias
    self.E = self.Encoder
    self.D = self.Discriminator
    self.C = self.Classifier

    # Add input layers
    self.E.add(Input(sample_shape=input_shape))
    self.WSI_input = Input(sample_shape=input_shape,
                           name=PatchClassifier.Keys.WSI_input)

  # region : Properties

  @property
  def description(self):
    return [f'Encoder: {self.E.structure_string()}',
            f'Discriminator: {self.D.structure_string()}',
            f'Classifier: {self.C.structure_string()}']

  # endregion : Properties

  # region: Public Methods

  # endregion: Public Methods

  # region: Private Methods

  def _build(self, **kwargs):
    # Link encoder
    self._E_output = self.Encoder()

    # Plug self._G_output to GAN.output slot
    self.outputs.plug(self._G_output)

    # Link discriminator
    logits_dict = context.logits_tensor_dict
    self._Dr = self.Discriminator()
    self._logits_Dr = logits_dict.pop(list(logits_dict.keys())[0])
    self._Df = self.Discriminator(self._G_output)
    self._logits_Df = logits_dict.pop(list(logits_dict.keys())[0])

    # Define loss (extra losses are not supported yet)
    with tf.name_scope('Losses'):
      self._define_losses(loss, kwargs.get('smooth_factor', 0.9))

    # Define train steps
    if G_optimizer is None:
      G_optimizer = tf.train.AdamOptimizer(th.learning_rate)
    if D_optimizer is None:
      D_optimizer = tf.train.AdamOptimizer(th.learning_rate)

    # TODO: self.[DG].parameters should be checked
    with tf.name_scope('Train_Steps'):
      with tf.name_scope('G_train_step'):
        self._train_step_G.plug(G_optimizer.minimize(
          loss=self._loss_G.tensor, var_list=self.G.parameters))
      with tf.name_scope('D_train_step'):
        self._train_step_D.plug(D_optimizer.minimize(
          loss=self._loss_D.tensor, var_list=self.D.parameters))

  def _define_losses(self, loss, alpha):
    """To add extra losses, e.g., regularization losses, this method should be
    overwritten"""
    if callable(loss):
      self._loss_G, self._loss_D = loss(self)
      assert False
      return
    elif not isinstance(loss, str):
      raise TypeError('loss must be callable or a string')

    loss = loss.lower()
    if loss == pedia.default:
      loss_Dr_raw = -tf.log(self._Dr, name='loss_D_real_raw')
      loss_Df_raw = -tf.log(1. - self._Df, name='loss_D_fake_raw')
      loss_G_raw = -tf.log(self._Df, name='loss_G_raw')
    elif loss == pedia.cross_entropy:
      loss_Dr_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Dr, labels=tf.ones_like(self._logits_Dr) * alpha)
      loss_Df_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.zeros_like(self._logits_Df))
      loss_G_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.ones_like(self._logits_Df))
    else:
      raise ValueError('Can not resolve "{}"'.format(loss))

    with tf.name_scope('D_losses'):
      loss_Dr = tf.reduce_mean(loss_Dr_raw, name='loss_D_real')
      loss_Df = tf.reduce_mean(loss_Df_raw, name='loss_D_fake')
      loss_D = tf.add(loss_Dr, loss_Df, name='loss_D')
      self._loss_D.plug(loss_D)
    with tf.name_scope('G_loss'):
      self._loss_G.plug(tf.reduce_mean(loss_G_raw, name='loss_G'))

  # endregion: Private Methods

  # region: Overwriting

  def update_model(self, data_batch, **kwargs):
    assert isinstance(data_batch, DataSet)

    # (1) Update D
    feed_dict_D = {self.D.input_tensor: data_batch.features,
                   self.G.input_tensor: self._random_z(data_batch.size)}
    feed_dict_D.update(self.agent.get_status_feed_dict(is_training=True))
    results = self._update_group_D.run(feed_dict_D)

    # (2) Update G
    feed_dict_G = {self.G.input_tensor: self._random_z(data_batch.size)}
    feed_dict_G.update(self.agent.get_status_feed_dict(is_training=True))
    results.update(self._update_group_G.run(feed_dict_G))

    return results

  def validate_model(self, data_set, batch_size=None, allow_sum=False,
                     verbose=False, seq_detail=False, num_steps=None):
    pass

  def handle_structure_detail(self):
    E_rows, E_total_params, E_dense_total = self.E.structure_detail
    D_rows, D_total_params, D_dense_total = self.D.structure_detail
    C_rows, C_total_params, C_dense_total = self.C.structure_detail

    # Take some notes
    params_str = 'Encoder total params: {}'.format(E_total_params)
    self.agent.take_notes(params_str)
    params_str = 'Discriminator total params: {}'.format(D_total_params)
    self.agent.take_notes(params_str)
    params_str = 'Classifier total params: {}'.format(C_total_params)
    self.agent.take_notes(params_str)

    if th.show_structure_detail:
      print('.. Encoder structure detail:\n{}'.format(E_rows))
      print('.. Discriminator structure detail:\n{}'.format(D_rows))
      print('.. Classifier structure detail:\n{}'.format(C_rows))

    if th.export_structure_detail:
      self.agent.take_notes('Structure detail of Encoder:', False)
      self.agent.take_notes(E_rows, False)
      self.agent.take_notes('Structure detail of Discriminator:', False)
      self.agent.take_notes(D_rows, False)
      self.agent.take_notes('Structure detail of Classifier:', False)
      self.agent.take_notes(C_rows, False)

  # endregion: Overwriting

