'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

_pad        = '_'
_eos        = '~'
_characters = ''
with open('text/map.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
f.close()
    
for line in lines:
    char, sym = line.strip().split('\t')
    _characters = _characters+sym

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet Change by Giridhar
