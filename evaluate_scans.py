#! /usr/bin/env python

'''
Author: Anthony Badea
'''

# python packages
import torch
import argparse
import numpy as np
import os, sys
import h5py
import json
import yaml
import glob
from tqdm import tqdm 
from model_blocks import x_to_p4
from sklearn.metrics import auc
from math import isnan

# multiprocessing
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# custom code
from batcher import loadDataFromH5
from model import StepLightning
from utils.get_mass import get_mass_max, get_mass_set

def get_binning(var, center=False):
    if var.startswith("m"):
        bins= np.linspace(0,4000,51)
    else:
        bins= np.linspace(-1,3,1001)
    if center:
        bins = (bins[:-1]+bins[1:])/2
    return bins


def evaluate(config):
    ''' perform the full model evaluation '''
    
    ops = options()
    config["model"]["weights"] = ops.weights

    print(f"evaluating on {config['inFileName']}")

    # load model
    model = StepLightning(**config["model"])
    model.to(config["device"])
    model.eval()
    model.Encoder.eval()

    # load data
    x = loadDataFromH5(config["inFileName"], ops.normWeights)
    if ops.normWeights:
        x, w = x
    mask = (x[:,:,0] == 0)
    
    # evaluate
    outData = {}
    with torch.no_grad():

        # make predictions
        p, ae = [], []
        niters = int(np.ceil(x.shape[0]/ops.batch_size))
        for i in tqdm(range(niters)):
            start, end = i*ops.batch_size, (i+1)*ops.batch_size
            # be careful about the memory transfers to not use all gpu memory
            temp = x[start:end].to(config["device"])
            ae_out, jet_choice = model(temp)
            c1, c2, c1_out, c2_out, c1random, c2random, c1random_out, c2random_out, cp4 = ae_out
            c1, c2, c1_out, c2_out = c1.cpu(), c2.cpu(), c1_out.cpu(), c2_out.cpu()
            jet_choice = jet_choice.cpu()
            ae.append(torch.stack([c1, c2, c1_out, c2_out],-1))
            p.append(jet_choice)

        # concat
        p = torch.concat(p)
        ae = torch.concat(ae)
        c1, c2, c1_out, c2_out = [ae[:,i] for i in range(4)]
        mse_loss = torch.mean((c1_out-c1)**2 + (c2_out-c2)**2,-1)
        mse_crossed_loss = torch.mean((c1_out-c2)**2 + (c2_out-c1)**2,-1)
        
        # convert x
        x = x_to_p4(x)
        # apply mask to x
        x = x.masked_fill(mask.unsqueeze(-1).repeat(1,1,x.shape[-1]).bool(), 0)
        pmom_max, pidx_max = get_mass_max(x, p)
        m0 = pmom_max[:,0,3]
        m1 = pmom_max[:,-2,3]
        m2 = pmom_max[:,-1,3]
        mavg = (m1+m2)/2
        mdiff = np.abs(m1-m2)/2
                
        histograms = {
          "m0"    : np.histogram(m0, bins=get_binning("m0")),
          "m1"    : np.histogram(m1, bins=get_binning("m1")),
          "m2"    : np.histogram(m2, bins=get_binning("m2")),
          "mavg"  : np.histogram(mavg, bins=get_binning("mavg")),
          "mdiff" : np.histogram(mdiff, bins=get_binning("mdiff")),

          "loss"    : np.histogram(np.log(mse_loss), bins=get_binning("loss")),
          "xloss"   : np.histogram(np.log(mse_crossed_loss), bins=get_binning("xloss")),
        }
        histograms = {k:h for k,(h,bins) in histograms.items()} #drop bin ranges

    return histograms



def get_auc(hsig, hbkg):
    if np.sum(hsig)==0 or np.sum(hbkg)==0: return 0.5
    hsigcum = np.cumsum(hsig/np.sum(hsig))
    hbkgcum = np.cumsum(hbkg/np.sum(hbkg))
    return auc(hsigcum,hbkgcum)

def get_aggregate(metrics):
    metrics = {k:v['avg'] for k, v in metrics.items()}
    if any(isnan(v) for v in metrics.values()):
        return 0
    separation =['loss','xloss']
    agg = max([metrics[k] for k in separation])
    agg += metrics['mavg']/2
    agg -= metrics['mdiff']/1000
    return agg

def get_outfile(weightfile, big=False, h5=False):
    tag = "_big" if big else "_"
    if h5:
        return weightfile+tag+'summary.h5'
    return weightfile+tag+'summary.json'

def get_truemean(sample, mean):
    if "1100" in sample: return 1100
    if "1500" in sample: return 1500
    if "1900" in sample: return 1900
    if "2300" in sample: return 2300
    return mean

def summarize(outData, weightfile):

    metrics = {}
    for var in outData['bkg'].keys():
        histos = {}
        metrics[var] = {}
        for sample in outData.keys():
            h = outData[sample][var]
            histos[sample] = h
            if np.sum(h)==0:
                metrics[var][sample] = 0
                continue
            bins = get_binning(var, center=True)
            mean = np.average(bins, weights=h)
            truemean = get_truemean(sample, mean)
            std =  np.sqrt(np.average((bins - truemean)**2, weights=h))
            if var.startswith('mdiff'):
                metrics[var][sample] = mean
            elif var.startswith('m'):
                metrics[var][sample] = truemean/std if std!=0 else 0
                if sample=='bkg': metrics[var][sample] = -mean/1000
        if not var.startswith('m'):
            for sample in outData.keys():
                if sample == 'bkg': continue
                metrics[var][sample] = get_auc(histos[sample],histos['bkg'])
        metrics[var]['avg'] = sum(metrics[var].values())/len(metrics[var])
    metrics['aggregate'] = get_aggregate(metrics)

    big = len(outData.keys()) > 10
    with open(get_outfile(weightfile, big),'w') as outfile:
         json.dump(metrics, outfile)
    with h5py.File(get_outfile(weightfile, big, h5=True), 'w') as hf:
        for key, histos in outData.items():
            grp = hf.create_group(key)
            for key, val in histos.items():
                grp.create_dataset(key,data=val)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bkg", help="Background file to evaluate on.", default=None, required=True)
    parser.add_argument("--sig", nargs="+", help="Signal file(s) to evaluate on.", default=None, required=True)
    parser.add_argument("-j",  "--ncpu", help="Number of cores to use for multiprocessing. If not provided multiprocessing not done.", default=1, type=int)
    parser.add_argument("-w",  "--weights", help="Pretrained weights to evaluate with.", default=None, required=True)
    parser.add_argument("--normWeights",action="store_true", help="Store also normalization weights")
    parser.add_argument("-b", "--batch_size", help="Batch size", default=10**4, type=int)
    parser.add_argument('--event_selection', default="", help="Enable event selection in batcher.")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files.")
    parser.add_argument('--noTruthLabels', action="store_true", help="Option to tell data loader that the file does not contain truth labels")
    parser.add_argument('--gpu', action="store_true", help="Run evaluation on gpu.")
    return parser.parse_args()
 
if __name__ == "__main__":

    # user options
    ops = options()
    outfolder = os.path.dirname(ops.weights)
    ops.config_file = os.path.join(outfolder,'hparams.yaml')
    ops.config_file2 = os.path.join(outfolder,'lightning_logs/version_0/hparams.yaml')
    if not os.path.isfile(ops.weights):
        print("Not a regular file:",ops.weights)
        sys.exit(0)
    if os.path.exists(ops.config_file2):
        ops.config_file = ops.config_file2
    if not os.path.exists(ops.config_file):
        print("File does not exist:",ops.config_file)
        sys.exit(0)
    if os.path.exists(get_outfile(ops.weights, len(ops.sig)>10)) and not ops.doOverwrite:
        print("File exists:",get_outfile(ops.weights, len(ops.sig)>10))
        sys.exit(0)

    # pick up model configurations
    print(f"Using configuration file: {ops.config_file}")
    model_config = {}
    with open(ops.config_file, 'r') as fp:
        if ops.config_file.endswith("yaml"):
            model_config["model"] = yaml.load(fp, Loader=yaml.Loader)
            if 'lightning_logs' in ops.config_file:
                model_config['model']['encoder_config']['do_gumbel'] = True
                model_config['model']['encoder_config']['mass_scale'] = 100
        else:
            model_config = json.load(fp)

    # understand device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if ops.gpu else "cpu"

    # create evaluation job dictionaries
    outputs = {}
    for inFileName in [ops.bkg]+ops.sig:

        tag = "bkg" if inFileName==ops.bkg else os.path.basename(inFileName).split(".")[0]

        # append configuration
        config = {
            "inFileName" : inFileName,
            "device" : device,
            **model_config
        }
        out = evaluate(config)
        outputs[tag] = out

    summarize(outputs, ops.weights)
