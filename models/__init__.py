from .prosodytts import Pro_mel, Prosody_linear, Pro_Vocoder

def create_model(name, hparams):
  if name == 'prosodytts_mel':
    return Pro_mel(hparams)
  elif name == 'prosodytts_linear':
    return Prosody_linear(hparams)
  elif name == 'prosodytts_vocoder':
    return Pro_Vocoder(hparams)
  else:
    raise Exception('Unknown model: ' + name)
