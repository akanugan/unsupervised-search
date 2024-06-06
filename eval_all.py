import subprocess
from multiprocessing import Pool

input_files = [
    "../slimmed_ntuples/combined_QCD_new_HT500to700_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT700to1000_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT1000to1500_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT1500to2000_100k.h5",
    "../slimmed_ntuples/combined_QCD_new_HT2000toInf_100k.h5",
]

weight_files = [
    #"experiments/minimal/training_2024.04.19.12.51.52_combined_QCD_new_HT500to700_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.19.11.14.46_combined_QCD_new_HT700to1000_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.19.11.37.55_combined_QCD_new_HT1000to1500_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.19.11.50.17_combined_QCD_new_HT1500to2000_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.19.12.18.10_combined_QCD_new_HT2000toInf_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.26.12.44.52_combined_QCD_new_HT700to1000_100k/finalWeights.ckpt",
    #"experiments/minimal/training_2024.04.26.12.55.46_combined_QCD_new_HT2000toInf_100k/finalWeights.ckpt"
    #"experiments/minimal/training_2024.05.03.10.13.57_signal1500/finalWeights.ckpt"
    "experiments/minimal/training_2024.05.03.10.46.56_signal1250/finalWeights.ckpt"
]

output_dir = "evaluate/new_train/"
config_file = "config_files/minimal_config.json"

def evaluate(command):
    subprocess.call(command, shell=True)

if __name__ == '__main__':
    commands = []
    for input_file in input_files:
        for weight_file in weight_files:
            command = "python3 evaluate.py -c {} -i {} -o {} -w {}".format(config_file, input_file, output_dir, weight_file)
            #commands.append(command)
            subprocess.call(command, shell=True)
    # num_processes = 4  

    # with Pool(processes=num_processes) as pool:
    #     pool.map(evaluate, commands)

    print("All evaluations completed.")
