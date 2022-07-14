# Prosody-TTS
The inference scripts of the Prosody-TTS system trained with Telugu language speech.



### Installing Dependencies
The required packages are placed in the **requirement.yml** file.

Create a conda environment using the yml file.

```
conda env create -f requirement.yml
```

We use the epitran library for translating the text into IPA characters. We have modified a few of the mappings and a few post-processing rules, which are placed in the **util** directory. Please copy the **tel-Telu-nc.csv** to the **epitran/data/map/** directory in the installed packages. Please copy **tel-Telu-nc.txt** to the **epitran/data/post/** directory. 




### Synthesize from the trained model

Run **synthesize.py** to synthesize the speech for the desired text input.

```
python synthesize.py --text text.txt --dur_factor 1.0 --fo_factor 1.0 --plot True
```

#### Arguments
--text: Text file contaning the input text.

--dur_factor: Duration modification factor.

--fo_factor: fo modification factor.

--plot: Plotting predicted durations and fo. (Boolean)



### Synthesized samples

The synthesized samples can be found in the [link.](https://siplabiith.github.io/prosody-tts.html) It contains short and long utterances compared with different models. It also includes the prosody-modified speech samples synthesized using the different duration and fo modification factors.




### Live demo

We have created [webpage]( https://speech.iith.ac.in/demos/tts/prosody_tts/ ) for the demo. It takes text, duration factor, and fo factor as input and generates the corresponding speech.
