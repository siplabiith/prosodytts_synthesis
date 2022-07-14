import io
import numpy as np
from numpy import newaxis
import tensorflow as tf
import hparams as hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
from scipy.signal import medfilt
from util.exc_from_f0 import synthesise_excitation


class Synthesizer:
  def load_mel(self, mel_checkpoint_path, vocoder_checkpoint_path, mel_model='prosodytts_mel', vocoder_model='prosodytts_vocoder'):
    #print('Constructing model: %s' % mel_model)
    tf.compat.v1.disable_eager_execution()
    inputs = tf.compat.v1.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.compat.v1.placeholder(tf.int32, [1], 'input_lengths')
    mult_mat = tf.compat.v1.placeholder(tf.float32, [1, None, None], 'mult_mat')
    fo_targets_ip = tf.compat.v1.placeholder(tf.int32, [1, None], 'fo_targets_ip')
    frame_index = tf.compat.v1.placeholder(tf.int32, [1, None], 'frame_index')
    frame_ind_rev = tf.compat.v1.placeholder(tf.int32, [1, None], 'frame_ind_rev')
    mel_ip = tf.compat.v1.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_inputs')
    lin_ip = tf.compat.v1.placeholder(tf.float32, [1, None, hparams.num_freq], 'linear_inputs')
    noise_ip = tf.compat.v1.placeholder(tf.float32, [1, None, 1], 'noise_inputs')
    exc_ip = tf.compat.v1.placeholder(tf.float32, [1, None, 1], 'exc_inputs')
    with tf.compat.v1.variable_scope('model') as scope:
      self.mel_model = create_model(mel_model, hparams)
      self.mel_model.initialize(inputs, input_lengths, mult_mat = mult_mat, fo_targets_ip = fo_targets_ip, frame_index = frame_index, frame_ind_rev = frame_ind_rev)
      self.dur_output = self.mel_model.dur_outputs[0]
      self.fo_output = self.mel_model.fo_outputs[0]
      self.bap_outputs = self.mel_model.bap_outputs[0]
      self.mel_outputs = self.mel_model.mel_outputs[0]
      #scope.reuse_variables()

    print('Loading checkpoint: %s' % mel_checkpoint_path)
    self.session = tf.compat.v1.Session()
    self.session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(self.session, mel_checkpoint_path)


  def load_vocoder(self, mel_checkpoint_path, vocoder_checkpoint_path, mel_model='prosodytts_mel', vocoder_model='prosodytts_vocoder'):
    #print('Constructing model: %s' % mel_model)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    mel_ip = tf.compat.v1.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_inputs')
    lin_ip = tf.compat.v1.placeholder(tf.float32, [1, None, hparams.num_freq], 'linear_inputs')
    noise_ip = tf.compat.v1.placeholder(tf.float32, [1, None, 1], 'noise_inputs')
    exc_ip = tf.compat.v1.placeholder(tf.float32, [1, None, 1], 'exc_inputs')
    with tf.compat.v1.variable_scope('model') as scope:
      self.vocoder_model = create_model(vocoder_model, hparams)
      self.vocoder_model.initialize(mel_inputs=mel_ip, linear_inputs=lin_ip, noise_inputs=noise_ip, exc_inputs=exc_ip)
      self.pred_wave = self.vocoder_model.pred_wave[0]

    print('Loading checkpoint: %s' % vocoder_checkpoint_path)
    self.session1 = tf.compat.v1.Session()
    self.session1.run(tf.compat.v1.global_variables_initializer())
    saver1 = tf.compat.v1.train.Saver()
    saver1.restore(self.session1, vocoder_checkpoint_path)


  def predict(self, text, dur_factor, fo_factor):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict_dur = {
      self.mel_model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.mel_model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    durs = self.session.run(self.dur_output, feed_dict=feed_dict_dur)
    durs[durs<0] = 0
    durs = durs*float(dur_factor)
    assert len(seq) == durs.shape[0]

    frame_index = np.array([[]])
    frame_index = frame_index.reshape(0,1)
    frame_ind_rev = np.array([[]])
    frame_ind_rev = frame_ind_rev.reshape(0,1)
    for j in range(len(durs)):
        frms = np.arange(np.round(durs[j]/5.0)).reshape(-1,1) + 1
        frms_rev = np.flip(frms, axis=0)
        frame_index = np.append(frame_index, frms, axis = 0)
        frame_ind_rev = np.append(frame_ind_rev, frms_rev, axis = 0)
    frame_index = frame_index.reshape(1,-1)
    frame_ind_rev = frame_ind_rev.reshape(1,-1)

    frames = np.round(durs/5.0).astype(np.int32)
    i = 0
    for j in range(len(frames)):
        one = np.eye(1, durs.shape[0], j, dtype=np.int32)
        one = np.repeat(one, frames[j], axis = 0)
        if i == 0:
            mult_mat = one
            i = i + 1
        else:
            mult_mat = np.concatenate((mult_mat, one), axis = 0)
    assert mult_mat.shape[1] == durs.shape[0]

    feed_dict_fo = {
      self.mel_model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.mel_model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
      self.mel_model.mult_mat: [np.asarray(mult_mat, dtype=np.float32)],
      self.mel_model.frame_index: frame_index,
      self.mel_model.frame_ind_rev: frame_ind_rev
    }
    fos = self.session.run(self.fo_output, feed_dict=feed_dict_fo) 
    bap = self.session.run(self.bap_outputs, feed_dict=feed_dict_fo) 
    fos = fos * float(fo_factor)

    fos = fos.reshape(-1)
    vad = (fos>100)*1.0
    fos = fos*vad
    sigma = 10
    y=np.log10(fos+10e-8) - 2
    y = 1/(1+np.exp(-y/0.2));
    y=np.power(10, y+1.5);
    new_fos = y*vad
    new_fos = medfilt(new_fos,3)
    #new_fos = fos
    new_fos = new_fos.reshape(-1,1)
    fo_targets_ip = np.round(new_fos).astype(np.int32)   #Should be NxTout 
    fo_targets_ip = fo_targets_ip.reshape(1,-1)
    fo_targets_ip[fo_targets_ip<0] = 0

    feed_dict_mel = {
      self.mel_model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.mel_model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
      self.mel_model.mult_mat: [np.asarray(mult_mat, dtype=np.float32)],
      self.mel_model.fo_targets_ip: fo_targets_ip,
      self.mel_model.frame_index: frame_index,
      self.mel_model.frame_ind_rev: frame_ind_rev
    }

    mel = self.session.run(self.mel_outputs, feed_dict=feed_dict_mel)

    self.session.close()

    return durs, fo_targets_ip, bap, mel

  def synthesize(self, mel, noise, exc):

    mel = np.expand_dims(mel, axis=0)
    noise = np.expand_dims(noise, axis=0)
    exc = np.expand_dims(exc, axis=0)
    feed_dict_vocoder = {
      self.vocoder_model.mel_inputs: mel,
      self.vocoder_model.noise_inputs: noise,
      self.vocoder_model.exc_inputs: exc
    }
   
    wav = self.session1.run(self.pred_wave, feed_dict=feed_dict_vocoder)
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()
