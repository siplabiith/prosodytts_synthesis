import argparse
import os
import re
import numpy as np
import hparams as hparams
from synthesizer import Synthesizer
from util.exc_from_f0 import synthesise_excitation
from util.convert_to_ipa import convert_to_ipa
from util.plot import plot_dur_fo

def get_noise(bap):
  noise = np.array([[]])
  noise = noise.reshape(0,1)
  for j in range(len(bap)):
    nse = np.random.normal(0, float(bap[j]), (80,1))
    noise = np.append(noise, nse, axis = 0)
  return noise


def run_eval(args):
  mel_weights = 'weights/mel_model.ckpt-170000'
  vocoder_weights = 'weights/voc_model.ckpt-114000'
  synth = Synthesizer()
  synth.load_mel(mel_weights, vocoder_weights)  
  base_path = os.path.join(os.getcwd(),  'synthesized.wav')
  g = open(args.text, 'r')
  text = g.readlines()
  text_ipa = convert_to_ipa(text)
  print('Synthesizing: %s' % base_path)
  duration, fo, bap, mel = synth.predict(text_ipa, args.dur_factor, args.fo_factor)

  noise = get_noise(bap)
  wav_len = int(fo.shape[1]*80)
  exc = synthesise_excitation(fo, wav_len).astype(np.int16)
  exc = exc*1.0/(2**15)
  exc = np.expand_dims(exc, axis=-1)
  synth.load_vocoder(mel_weights, vocoder_weights)
  with open(base_path, 'wb') as f:
    f.write(synth.synthesize(mel, noise, exc))

  if args.plot == True:
    plot_dur_fo(text_ipa, duration, fo)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--text', default='text.txt')
  parser.add_argument('--dur_factor', default=1.0)
  parser.add_argument('--fo_factor', default=1.0)
  parser.add_argument('--plot', default=True)

  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  run_eval(args)


if __name__ == '__main__':
  main()
