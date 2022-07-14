from text.symbols import symbols
cleaners='basic_cleaners' #english_cleaners


num_mels=80
num_freq=1025
sample_rate=16000
frame_length_ms=25
frame_shift_ms=5
preemphasis=0.97
min_level_db=-100
ref_level_db=20
sr=16000

pad=0
outputs_per_step=1
char_max = len(symbols)
char_max_length = None
embedding_size = 128

encoder_conv_banks = 16
decoder_conv_banks = 8
GRU_depth = 128

# Model:
outputs_per_step=1
encoder_conv_banks=8
decoder_conv_banks=16
embed_depth=128
prenet_depths=[256,128]
encoder_depth=256
postnet_depth=256
attention_depth=256
decoder_depth=256
xvec_depth=512

# Training:
batch_size=8
adam_beta1=0.9
adam_beta2=0.999
initial_learning_rate=0.002
decay_learning_rate=True
use_cmudict=False# Use CMUDict during training to learn pronunciation of ARPAbet phonemes

# Eval:
max_iters=600
griffin_lim_iters=50
power=1.5

