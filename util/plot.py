from matplotlib import pyplot as plt
import numpy as np

def plot_dur_fo(text_ipa, duration, fo):
  text_ipa = text_ipa.replace(" ", "")
  text_list = [char for char in text_ipa]
  x = np.arange(len(text_list))
  plt.plot(x, duration.reshape(-1))
  plt.title("Predicted durations")
  plt.xlabel('Phone index')
  plt.ylabel('Duration in ms')
  plt.xticks(range(0,len(text_list)), text_list)
  plt.show()

  fo = fo.astype(np.float32)
  fo[fo<=100]=np.nan
  plt.plot(fo.reshape(-1))
  plt.title("Predicted fo")
  plt.xlabel('Frames')
  plt.ylabel('fo in Hz')
  plt.show()

  return
