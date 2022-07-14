import tensorflow as tf, numpy as np
from text.symbols import symbols
from util.infolog import log
from .modules import encoder_cbhg, post_cbhg, prenet, decoder_prenet, conv1d, gru, vocoder, vocoder_only
from tensorflow.keras.layers import UpSampling1D, Bidirectional, GRU
from tensorflow.keras.layers import Conv1D, Dense, Activation, MaxPooling1D, Add, Bidirectional, GRU, Dropout,BatchNormalization, Lambda, Dot, Multiply,Add,add,UpSampling1D

class Prosody_linear():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, dur_targets=None, fo_targets=None, fo_targets_ip=None, frame_index=None, frame_ind_rev=None, mel_targets=None, linear_targets=None, mult_mat=None, bap_targets=None):
    '''Initializes the model for inference.

    Sets "dur_outputs", "mel_outputs", "linear_outputs", "mult_mat", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      dur_targets: float32 Tensor with shape [N, T_in, 1] where N is batch size, T_in is number of
        steps in the input time series, 1 is duration for each character, and values are entries in milliseconds.
        Only needed for training.
      fo_targets: float32 Tensor with shape [N, T_out, 1] where N is batch size, T_out is number of
        steps in the output time series, 1 is fo for each frame, and values are entries in Hz.
        Only needed for training.
      fo_targets_ip: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in 2*Hz.  (Double f0)
        Only needed for training.
      frame_index: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in frame indices for each upsampled feature.
        Only needed for training.
      frame_ind_rev: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in frame indices in opposite direction for 
        each upsampled feature. Only needed for training.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
      mult_mat: float32 Tensor with shape [N, T_out, T_in] where N is batch_size, T_out is number
        of steps in the output time series, T_in is number of
        steps in the input time series.
    '''
    with tf.compat.v1.variable_scope('inference') as scope:
      is_training = linear_targets is not None  #True #linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.compat.v1.get_variable(
        'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)          # [N, T_in, embed_depth=256]

      conv_bank_outputs = tf.concat([conv1d(embedded_inputs, k, 128, tf.nn.relu, is_training, 'conv_bank_%d' % k) for k in range(1, hp.encoder_conv_banks+1)], axis=-1)
      
      # Maxpooling:
      maxpool_output = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_bank_outputs)
      
      # Two projection layers:
      input_channels = embedded_inputs.get_shape()[2]
      projections = [128, input_channels]
      proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
      proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')
      
      # Residual connection:
      res_output = proj2_output + embedded_inputs
      
      # Bidirectional RNN
      half_depth = hp.encoder_depth//2
      encoder_gru = Bidirectional(GRU(half_depth, activation='relu', return_sequences=True))
      bi_dir_gru_op = encoder_gru(res_output)
      
      #Predict durations(in ms)  #Change by Giridhar
      dur_outputs = Dense(1)(bi_dir_gru_op)   # [N, T_in, 1]

      upsamp_enc = tf.matmul(mult_mat, bi_dir_gru_op)    # [N, T_out, 256]

      # Frame Embeddings
      frame_embedding_table_f = tf.compat.v1.get_variable(
        'frame_embed_f', [500, 16], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.25))
      frame_embedded_inputs_f = tf.nn.embedding_lookup(frame_embedding_table_f, frame_index)

      frame_embedding_table_b = tf.compat.v1.get_variable(
        'frame_embed_b', [500, 16], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.25))
      frame_embedded_inputs_b = tf.nn.embedding_lookup(frame_embedding_table_b, frame_ind_rev)

      upsamp_enc_frame = tf.concat([upsamp_enc, frame_embedded_inputs_f, frame_embedded_inputs_b], axis=-1)

      proj1 = conv1d(upsamp_enc_frame, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_1')
      proj2 = conv1d(proj1, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_2')
      proj3 = conv1d(proj2, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_3')


      #Predict fo(in Hz)  #Change by Giridhar
      fo_outputs = Dense(1, activation=tf.nn.relu)(proj3)   # [N, T_out, 1]
      bap_outputs = Dense(1, activation=tf.nn.sigmoid)(proj3)   # [N, T_out, 1]

      # Fo Embeddings
      fo_embed_table = tf.compat.v1.get_variable(
        'fo_embed', [500, 32], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0))
      fo_embedded_inputs = tf.nn.embedding_lookup(fo_embed_table, fo_targets_ip)          # [N, T_out, 128]
      
      fo_embedded_ip = tf.concat([proj3, fo_embedded_inputs], axis=-1)   # (N, T_out, 2*128)
      
      dec_conv_bank_outputs = tf.concat([conv1d(fo_embedded_ip, k, 128, tf.nn.relu, is_training, 'dec_conv_bank_%d' % k) for k in range(1, hp.decoder_conv_banks+1)],axis=-1)
      
      # Maxpooling:
      dec_maxpool_output = MaxPooling1D(pool_size=2, strides=1, padding='same')(dec_conv_bank_outputs)
      

      dec1 = conv1d(dec_maxpool_output, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_1')
      dec2 = conv1d(dec1, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_2')
      dec3 = conv1d(dec2, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_3')
      
      # Decoder Bidirectional RNN
      dec_input_lengths = None
      dec_half_depth = hp.decoder_depth//2
      decoder_gru = Bidirectional(GRU(dec_half_depth, activation='relu', return_sequences=True))
      dec_bi_dir_gru_op = decoder_gru(dec3)

      linear_outputs = Dense(hp.num_freq)(dec_bi_dir_gru_op)                # [N, T_out, F]

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.dur_outputs = dur_outputs
      self.fo_outputs = fo_outputs
      self.bap_outputs = bap_outputs
      self.bap_targets = bap_targets
      self.fo_targets_ip = fo_targets_ip
      self.frame_index = frame_index
      self.frame_ind_rev = frame_ind_rev
      self.linear_outputs = linear_outputs
      self.dur_targets = dur_targets
      self.fo_targets = fo_targets
      self.linear_targets = linear_targets
      self.mult_mat = mult_mat


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.compat.v1.variable_scope('loss') as scope:
      #hp = self._hparams
      self.dur_loss = 0.005 * (tf.reduce_mean(tf.abs(self.dur_targets - self.dur_outputs)))
      self.fo_loss = 0.001 * (tf.reduce_mean(tf.abs(self.fo_targets - self.fo_outputs)))
      self.bap_loss = (tf.reduce_mean(tf.abs(self.bap_targets - self.bap_outputs)))
#      self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      l1 = tf.abs(self.linear_targets - self.linear_outputs)
      # Prioritize loss for frequencies under 3000 Hz.
      #n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
      self.linear_loss = tf.reduce_mean(l1)     #0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
      self.loss = self.linear_loss + self.dur_loss + self.fo_loss + self.bap_loss #self.mel_loss + 

  def add_loss_new(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.compat.v1.variable_scope('loss') as scope:
      #hp = self._hparams
      self.dur_loss = 0.005 * (tf.reduce_mean(tf.abs(self.dur_targets - self.dur_outputs)))
      self.fo_loss = 0.001 * (tf.reduce_mean(tf.abs(self.fo_targets - self.fo_outputs)))
      self.mag_loss = tf.reduce_mean(tf.abs(self.linear_targets - self.linear_outputs))
      self.sc_loss = tf.norm(tf.math.exp(self.linear_targets) - tf.math.exp(self.linear_outputs)) / tf.norm(tf.math.exp(self.linear_targets))
      self.loss = self.sc_loss + self.mag_loss + self.dur_loss + self.fo_loss 


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.compat.v1.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)

class Pro_mel():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, dur_targets=None, fo_targets=None, fo_targets_ip=None, frame_index=None, frame_ind_rev=None, mel_targets=None, linear_targets=None, mult_mat=None, bap_targets=None):
    '''Initializes the model for inference.

    Sets "dur_outputs", "mel_outputs", "linear_outputs", "mult_mat", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      dur_targets: float32 Tensor with shape [N, T_in, 1] where N is batch size, T_in is number of
        steps in the input time series, 1 is duration for each character, and values are entries in milliseconds.
        Only needed for training.
      fo_targets: float32 Tensor with shape [N, T_out, 1] where N is batch size, T_out is number of
        steps in the output time series, 1 is fo for each frame, and values are entries in Hz.
        Only needed for training.
      fo_targets_ip: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in 2*Hz.  (Double f0)
        Only needed for training.
      frame_index: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in frame indices for each upsampled feature.
        Only needed for training.
      frame_ind_rev: int32 Tensor with shape [N, T_out] where N is batch size, T_out is number of
        steps in the output time series, and values are entries in frame indices in opposite direction for 
        each upsampled feature. Only needed for training.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
      mult_mat: float32 Tensor with shape [N, T_out, T_in] where N is batch_size, T_out is number
        of steps in the output time series, T_in is number of
        steps in the input time series.
    '''
    with tf.compat.v1.variable_scope('inference') as scope:
      is_training = linear_targets is not None  #True #linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.compat.v1.get_variable(
        'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)          # [N, T_in, embed_depth=256]

      conv_bank_outputs = tf.concat([conv1d(embedded_inputs, k, 128, tf.nn.relu, is_training, 'conv_bank_%d' % k) for k in range(1, hp.encoder_conv_banks+1)], axis=-1)
      
      # Maxpooling:
      maxpool_output = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_bank_outputs)
      
      # Two projection layers:
      input_channels = embedded_inputs.get_shape()[2]
      projections = [128, input_channels]
      proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
      proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')
      
      # Residual connection:
      res_output = proj2_output + embedded_inputs
      
      # Bidirectional RNN
      half_depth = hp.encoder_depth//2
      encoder_gru = Bidirectional(GRU(half_depth, activation='relu', return_sequences=True))
      bi_dir_gru_op = encoder_gru(res_output)
      
      #Predict durations(in ms)  #Change by Giridhar
      dur_outputs = Dense(1)(bi_dir_gru_op)   # [N, T_in, 1]

      upsamp_enc = tf.matmul(mult_mat, bi_dir_gru_op)    # [N, T_out, 256]

      # Frame Embeddings
      frame_embedding_table_f = tf.compat.v1.get_variable(
        'frame_embed_f', [500, 16], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.25))
      frame_embedded_inputs_f = tf.nn.embedding_lookup(frame_embedding_table_f, frame_index)

      frame_embedding_table_b = tf.compat.v1.get_variable(
        'frame_embed_b', [500, 16], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.25))
      frame_embedded_inputs_b = tf.nn.embedding_lookup(frame_embedding_table_b, frame_ind_rev)

      upsamp_enc_frame = tf.concat([upsamp_enc, frame_embedded_inputs_f, frame_embedded_inputs_b], axis=-1)

      proj1 = conv1d(upsamp_enc_frame, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_1')
      proj2 = conv1d(proj1, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_2')
      proj3 = conv1d(proj2, 3, hp.decoder_depth//2, tf.nn.relu, is_training, 'conv1_3')


      #Predict fo(in Hz)  #Change by Giridhar
      fo_outputs = Dense(1, activation=tf.nn.relu)(proj3)   # [N, T_out, 1]
      bap_outputs = Dense(1, activation=tf.nn.sigmoid)(proj3)   # [N, T_out, 1]

      # Fo Embeddings
      fo_embed_table = tf.compat.v1.get_variable(
        'fo_embed', [500, 32], dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0))
      fo_embedded_inputs = tf.nn.embedding_lookup(fo_embed_table, fo_targets_ip)          # [N, T_out, 128]
      
      fo_embedded_ip = tf.concat([proj3, fo_embedded_inputs], axis=-1)   # (N, T_out, 2*128)
      
      dec_conv_bank_outputs = tf.concat([conv1d(fo_embedded_ip, k, 128, tf.nn.relu, is_training, 'dec_conv_bank_%d' % k) for k in range(1, hp.decoder_conv_banks+1)],axis=-1)
      
      # Maxpooling:
      dec_maxpool_output = MaxPooling1D(pool_size=2, strides=1, padding='same')(dec_conv_bank_outputs)
      

      dec1 = conv1d(dec_maxpool_output, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_1')
      dec2 = conv1d(dec1, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_2')
      dec3 = conv1d(dec2, 3, hp.decoder_depth, tf.nn.relu, is_training, 'conv2_3')
      
      # Decoder Bidirectional RNN
      dec_input_lengths = None
      dec_half_depth = hp.decoder_depth//2
      decoder_gru = Bidirectional(GRU(dec_half_depth, activation='relu', return_sequences=True))
      dec_bi_dir_gru_op = decoder_gru(dec3)

      mel_outputs = Dense(hp.num_mels)(dec_bi_dir_gru_op)      # [N, T_out, M]


      self.inputs = inputs
      self.input_lengths = input_lengths
      self.dur_outputs = dur_outputs
      self.fo_outputs = fo_outputs
      self.bap_outputs = bap_outputs
      self.bap_targets = bap_targets
      self.fo_targets_ip = fo_targets_ip
      self.frame_index = frame_index
      self.frame_ind_rev = frame_ind_rev
      self.mel_outputs = mel_outputs
      self.dur_targets = dur_targets
      self.fo_targets = fo_targets
      self.mel_targets = mel_targets
      self.mult_mat = mult_mat


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.compat.v1.variable_scope('loss') as scope:
      #hp = self._hparams
      self.dur_loss = 0.05 * (tf.reduce_mean(tf.abs(self.dur_targets - self.dur_outputs)))
      self.fo_loss = 0.01 * (tf.reduce_mean(tf.abs(self.fo_targets - self.fo_outputs)))
      self.bap_loss = (tf.reduce_mean(tf.abs(self.bap_targets - self.bap_outputs)))
      self.mel_loss = 10*tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      #l1 = tf.abs(self.linear_targets - self.linear_outputs)
      # Prioritize loss for frequencies under 3000 Hz.
      #n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
      #self.linear_loss = tf.reduce_mean(l1)     #0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
      self.loss = self.dur_loss + self.fo_loss + self.bap_loss + self.mel_loss  #+self.linear_loss 

  def add_loss_new(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.compat.v1.variable_scope('loss') as scope:
      #hp = self._hparams
      self.dur_loss = 0.005 * (tf.reduce_mean(tf.abs(self.dur_targets - self.dur_outputs)))
      self.fo_loss = 0.001 * (tf.reduce_mean(tf.abs(self.fo_targets - self.fo_outputs)))
      self.mag_loss = tf.reduce_mean(tf.abs(self.linear_targets - self.linear_outputs))
      self.sc_loss = tf.norm(tf.math.exp(self.linear_targets) - tf.math.exp(self.linear_outputs)) / tf.norm(tf.math.exp(self.linear_targets))
      self.loss = self.sc_loss + self.mag_loss + self.dur_loss + self.fo_loss 


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.compat.v1.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)

class Pro_Vocoder():
  def __init__(self, hparams):
    self._hparams = hparams
    


  def initialize(self, mel_inputs=None, linear_inputs=None, noise_inputs=None, exc_inputs=None, wav_outputs=None):

    with tf.compat.v1.variable_scope('inference') as scope:
      is_training = mel_inputs is not None  #True #spec_inputs is not None
      batch_size = tf.shape(mel_inputs)[0]
      hp = self._hparams
     
      pred_wave = vocoder(mel_inputs, exc_inputs, noise_inputs, is_training)
            
      self.exc_inputs = exc_inputs 
      self.mel_inputs = mel_inputs
      self.noise_inputs = noise_inputs
      self.linear_inputs = linear_inputs
      self.pred_wave = pred_wave
      self.wav_outputs = wav_outputs
      
  def add_loss_vocoder(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.compat.v1.variable_scope('loss') as scope:
      #self.pred_wave = tf.dtypes.cast(self.pred_wave, tf.float16)
      #self.wave = tf.dtypes.cast(self.wave, tf.float16)
      self.loss_td = tf.reduce_mean((self.pred_wave - self.wav_outputs)**2)
      self.mr_loss1 = self.stftloss(self.wav_outputs, self.pred_wave, frame_length=1024, frame_step=256, fft_length=1024)
      self.mr_loss2 = self.stftloss(self.wav_outputs, self.pred_wave, frame_length=512, frame_step=128, fft_length=1024)
      self.mr_loss3 = self.stftloss(self.wav_outputs, self.pred_wave, frame_length=1024, frame_step=512, fft_length=2048)
      #self.loss_td = tf.reduce_mean(tf.abs(self.pred_wave - (self.wave/(2 ** 15 - 1))))
      self.loss = self.loss_td + self.mr_loss1 + self.mr_loss2 + self.mr_loss3


  def stftloss(self, y_true, y_pred, frame_length, frame_step, fft_length):
    epsilon=1e-5
    S = tf.signal.stft(tf.squeeze(y_true, axis=-1), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, pad_end=True)
    N = tf.signal.stft(tf.squeeze(y_pred, axis=-1), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, pad_end=True)
    S = tf.math.abs(S)
    N = tf.math.abs(N)
    L_spec = tf.reduce_mean(tf.math.abs(S-N)) 
    return L_spec


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.compat.v1.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
