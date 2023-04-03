import h5py as h
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fname = "/Volumes/ME424_HD/nanopores/2023_03_23/ch1/pore7/data0/EVENTS.hdf5"
file = h.File(fname, 'r')

data = file["current_data"]
sets = list(data.keys())

df = pd.DataFrame()
for set in data:
    props = dict(data[set].attrs)
    row = pd.DataFrame([pd.Series(data = [set,*props.values()], index = ["name", *props.keys()])])
    df = pd.concat([df, row], ignore_index = True)

df_filt = df.query('samples < 20000 & samples > 1000')

import random

choices = random.choices(list(df_filt.name), k=9)

fig, axs = plt.subplots(nrows=3,ncols=3)
fig.suptitle("Gallery of Lambda Translocations, 600mV 4M LiA")
fig.tight_layout()
fig.subplots_adjust(hspace=1,wspace=1)
# fig.subplots_adjust(hspace = 2, wspace = 2)
for i, c in enumerate(choices):
    d = data[c][:]
    axs[i//3,i%3].plot(np.arange(len(d))/1e6,d)
    axs[i//3,i%3].set_title(c)
    axs[i//3, i%3].set_xlabel("Time /s")
    axs[i//3, i%3].set_ylabel("Current /nA")

plt.show(block=True)