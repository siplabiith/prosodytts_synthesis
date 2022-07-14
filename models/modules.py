import tensorflow as tf
from tensorflow.keras.layers import GRUCell
import numpy
#from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import Conv1D, Dense, Activation, MaxPooling1D, Add, Bidirectional, GRU, Dropout,BatchNormalization, Lambda, Dot, Multiply,Add,add,UpSampling1D

def prenet(inputs, is_training, layer_sizes, scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.compat.v1.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = Dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = Dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x

def decoder_prenet(inputs, is_training, layer_sizes, scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.compat.v1.variable_scope(scope or 'decoder_prenet'):
    for i, size in enumerate(layer_sizes):
      dense = Dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = Dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x

def encoder_cbhg(inputs, input_lengths, is_training, depth):
  input_channels = inputs.get_shape()[2]
  return cbhg(
    inputs,
    input_lengths,
    is_training,
    scope='encoder_cbhg',
    K=16,
    projections=[128, input_channels],
    depth=depth)


def post_cbhg(inputs, input_dim, is_training, depth):
  return cbhg(
    inputs,
    None,
    is_training,
    scope='post_cbhg',
    K=8,
    projections=[256, input_dim],
    depth=depth)


def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
  with tf.compat.v1.variable_scope(scope):
    with tf.compat.v1.variable_scope('conv_bank'):
      # Convolution bank: concatenate on the last axis to stack channels from all convolutions
      conv_outputs = tf.concat(
        [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
        axis=-1
      )

    # Maxpooling:
    maxpool_output = MaxPooling1D(
      conv_outputs,
      pool_size=2,
      strides=1,
      padding='same')

    # Two projection layers:
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

    # Residual connection:
    highway_input = proj2_output + inputs

    half_depth = depth // 2
    assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

    # Handle dimensionality mismatch:
    if highway_input.shape[2] != half_depth:
      highway_input = Dense(highway_input, half_depth)

    # 4-layer HighwayNet:
    for i in range(4):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
    rnn_input = highway_input

    # Bidirectional RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(half_depth),
      GRUCell(half_depth),
      rnn_input,
      sequence_length=input_lengths,
      dtype=tf.float32)
    return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope, depth):
  with tf.compat.v1.variable_scope(scope):
    H = Dense(
      inputs,
      units=depth,
      activation=tf.nn.relu,
      name='H')
    T = Dense(
      inputs,
      units=depth,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.compat.v1.variable_scope(scope):
    conv1d_output = Conv1D(
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')(inputs)
    return BatchNormalization()(conv1d_output)

def conv1d_new(inputs, kernel_size, channels, is_training, scope, activation=None, padding='valid', dilation_rate=1):
  with tf.compat.v1.variable_scope(scope):
    conv1d_output = Conv1D(
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      dilation_rate=dilation_rate,
      padding=padding)(inputs)
    return conv1d_output

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results 
        are concatenated.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
            
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs
        
def residual_block(data, block_number, in_shape=(None,1), is_training=True, convlayers=3, convwidth=3, padding='same', dilation_rate=1, activation='relu'):
    m,dim = in_shape  ## use dim for conv layers so that transformed and original data can be combined
    transformed_data = conv1d_new(data, convwidth, dim, is_training, 'conv1d_%d' % block_number, activation, padding=padding, dilation_rate=dilation_rate)
    for subsequent_layer in range(convlayers-1):
        transformed_data = conv1d_new(transformed_data, convwidth, dim, is_training, 'conv1d_tr_%d_%d' % (block_number,subsequent_layer), activation, padding=padding, dilation_rate=dilation_rate)
    transformed_data = tf.keras.layers.add([data, transformed_data])
    return transformed_data

def vocoder(mel, exc, noise, is_training):
    layers_per_block=[3,3,3,3,3,3,3,3]
    widths=[9,9,9,9,9,9,9,9]
    dilations=[20,1,1,1,1,1,1,1]
    convchannels=64
    
    up_samp_voco_inp = UpSampling1D(size=80)(mel)
    data = tf.concat([up_samp_voco_inp, exc, noise], axis=-1)
    data = conv1d(data, 1, convchannels, None, is_training, 'voco1')
    for (block_number, (l,w,d)) in enumerate(zip(layers_per_block, widths, dilations)):
        data = residual_block(data, block_number, in_shape=(None, convchannels), is_training=True, convlayers=l, convwidth=w, dilation_rate=d, activation=tf.nn.relu)
        data = BatchNormalization()(data)
    wave_pred = conv1d_new(data, 1, 1, is_training, 'voco_final', None)
    return wave_pred

def vocoder_only(voco_inp, wave, is_training):
    layers_per_block=[3,3,3,3]
    widths=[9,9,9,9]
    dilations=[20,1,1,1]
    convchannels=64
    
    data = UpSampling1D(size=80)(voco_inp)
    data = conv1d(data, 1, convchannels, None, is_training, 'voco1')
    for (block_number, (l,w,d)) in enumerate(zip(layers_per_block, widths, dilations)):
        data = residual_block(data, block_number, in_shape=(None, convchannels), is_training=True, convlayers=l, convwidth=w, dilation_rate=d, activation=tf.nn.relu)
        data = BatchNormalizationn(data, training=is_training)
    wave_pred = conv1d_new(data, 1, 1, is_training, 'voco_final', None)
    return wave_pred
    
