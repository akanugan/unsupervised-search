import numpy as np
import h5py
from glob import glob
import os
import json

# get list of input files
inFilePath = "./unsupervised-search/evaluate/*.h5"
inFileList = glob(inFilePath)

# histogram settings
bins_mass = np.linspace(0, 4000, 41)
density_mass = True

bins_loss = np.linspace(-7, 13, 51)
density_loss = True

# store histograms
store = {}

# create mass plot
store["mass_plot"] = {}
store["mass_plot"]["bins"] = list(bins_mass)

# create loss plot
store["loss_plot"] = {}
store["loss_plot"]["bins"] = list(bins_loss)

# loop over them
for inFile in inFileList:

    # get key
    key = os.path.basename(inFile).split("_transformer_classifier.h5")[0]
    
    with h5py.File(inFile,"r") as f:

        # mass plot
        mavg = np.array(f["pred_ptetaphim_max"][:,:,-1].mean(-1))
        hist, bin_edges = np.histogram(mavg, bins=bins_mass, density=density_mass)
        store["mass_plot"][key] = list(hist)
        # if key in store["mass_plot"].keys():
        #     print(f"{key} already dictionary")
        # else:
        #     store["mass_plot"][key] = list(hist)

        # loss plot
        loss = np.log(np.array(f["loss"]))
        hist, bin_edges = np.histogram(loss, bins=bins_loss, density=density_loss)
        store["loss_plot"][key] = list(hist)
        # if key in store["mass_plot"].keys():
        #     print(f"{key} already dictionary")
        # else:
        #     store["mass_plot"][key] = list(hist)

print(len(store.keys()), len(store["loss_plot"].keys()), len(store["mass_plot"].keys()))

# Open the file in write mode and dump the dictionary into the file
outFileName = "output.json"
with open(outFileName, 'w') as f:
    json.dump(store, f)

# load file and verify
with open(outFileName,"r") as f:
    o = json.load(f)

# load paper json
paper =	"./unsupervised-search/plotting/histdump_20230910.json"
with open(paper,"r") as f:
    h = json.load(f)
    
# compare w.r.t paper
print("Name, Mass Plot, Loss Plot")
for i in h['mass_plot'].keys():
    if i == "bins":
        print(i,
              np.all(np.array(o["mass_plot"][i]) == np.array(h['mass_plot'][i])),
              np.all(np.array(o["loss_plot"][i]) == np.array(h['loss_plot'][i]))
        )
    elif i == "Bkg":
        print(i,
              np.all(np.array(o["mass_plot"][f'{i}.sampled_200k']) == np.array(h['mass_plot'][i])),
              np.all(np.array(o["loss_plot"][f'{i}.sampled_200k']) == np.array(h['loss_plot'][i])),
        ) # bkg is 200k
    else:
        print(i,
              np.all(np.array(o["mass_plot"][f'{i}.sampled_10k']) == np.array(h['mass_plot'][i])),
              np.all(np.array(o["loss_plot"][f'{i}.sampled_10k']) == np.array(h['loss_plot'][i]))
        ) # signals are 10k
    #except:
    #    print(f"{i} not in o")

# verify bkg sample is the 200k one
#print("Bkg.sampled_200k", np.all(np.array(o['Bkg.sampled_200k']) == np.array(h['mass_plot']['Bkg'])))
#print("Bins",

# print(d)
