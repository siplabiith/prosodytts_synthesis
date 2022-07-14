import epitran
import os
epi=epitran.Epitran('tel-Telu-nc') 

def convert_to_ipa(text):
    ipastring=''
    for string in text:
        string = string.strip()
        ipastring=ipastring + 'Q'  #Beginning Silence
        prev = ''
        for word in string.split():
            epitran = epi.transliterate(word)
            for char in epitran:
                if char in [',', '.', '?', '!']:
                    ipastring=ipastring + 'Q'
                elif (char == 'k' or char == 'c' or char == 'ʈ' or char == 't' or char == 'p' or \
                    char == 'K' or char == 'C' or char == 'Ʈ' or char == 'T' or char == 'P') and prev != char:
                    ipastring=ipastring + 'Ƈ' + char
                else:
                    ipastring=ipastring + char
                prev = char
            ipastring=ipastring + ' '
                
        ipastring=ipastring + 'Q'  #Ending Silence
    return ipastring