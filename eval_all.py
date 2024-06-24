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
    #"experiments/minimal/training_2024.05.03.10.46.56_signal1250/finalWeights.ckpt"
    ##new trainings2
    "experiments/minimal/training_2024.06.03.14.14.04_combined_QCD_new_HT500to700_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.06.06.12.33.25_combined_QCD_new_HT500to700_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.05.29.14.33.26_combined_QCD_new_HT700to1000_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.06.03.15.06.53_combined_QCD_new_HT700to1000_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.06.06.12.42.49_combined_QCD_new_HT1000to1500_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.05.29.14.42.51_combined_QCD_new_HT1000to1500_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.05.29.14.58.35_combined_QCD_new_HT1500to2000_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.05.29.13.59.11_combined_QCD_new_HT1500to2000_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.06.03.14.50.11_combined_QCD_new_HT2000toInf_100k/finalWeights.ckpt",
    "experiments/minimal/training_2024.05.29.14.16.55_combined_QCD_new_HT2000toInf_100k/finalWeights.ckpt"
]

output_dir = "evaluate/new_train2/"
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
