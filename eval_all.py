import subprocess
from multiprocessing import Pool
import os

input_files = [
    "../slimmed_ntuples/combined_QCD_new_HT500to700_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT700to1000_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT1000to1500_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT1500to2000_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT2000toInf_100k.h5",
]

weight_files = [os.path.join('experiments/minimal', folder, 'finalWeights.ckpt') for folder in os.listdir('experiments/minimal') if os.path.isdir(os.path.join('experiments/minimal', folder)) and 'training_2024.06.26' in folder]
print(weight_files)

output_dir = "evaluate/on_old_config_training"
#config_file = "config_files/minimal_config.json"
config_file = "config_files/old_config.json"

def evaluate(command):
    subprocess.call(command, shell=True)

if __name__ == '__main__':
    commands = []
    for input_file in input_files:
        for weight_file in weight_files:
            command = "python3 evaluate.py -c {} -i {} -o {} -w {} --noTruthLabels --gpu --doOverwrite".format(config_file, input_file, output_dir, weight_file)
            #commands.append(command)
            subprocess.call(command, shell=True)
    # num_processes = 4  

    # with Pool(processes=num_processes) as pool:
    #     pool.map(evaluate, commands)

    print("All evaluations completed.")
