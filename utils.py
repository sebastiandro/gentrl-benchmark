import re
import matplotlib.pyplot as plt
import numpy as np
from guacamol.standard_benchmarks import (
    similarity
)


def get_log_data(logfile_path):
    float_re = r"[-+]?\d*\.\d+|\d+"

    loss_arr = []
    rec_arr = []
    with open(logfile_path, "r") as f:
        for line in f:
            if line[:4] == "loss":
                loss = float(re.findall(float_re,line)[0])
                if loss < -5: #issues with scientific notation, e.g 10e-7
                    loss = 0
                loss_arr.append(loss)
            if line[:3] == "rec":
                rec = float(re.findall(float_re, line)[0])
                rec_arr.append(rec)
    return loss_arr, rec_arr

def plot_log(loss_arr, rec_arr):
    steps = range(1,len(loss_arr)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, loss_arr)
    plt.title("Training Loss: 10 epochs, lr=0.010, beta=0.001")
    plt.ylabel("loss")
    plt.xlabel("step")
    plt.subplot(1, 2, 2)
    plt.plot(steps, -1*np.array(rec_arr))
    plt.title("Recreation loss: 10 epochs. lr=0.010, beta=0.001")
    plt.ylabel("loss")
    plt.xlabel("step")
    plt.show()

def celecoxib_similarity(smiles):
    guac_benchmark = similarity(
        smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
        name="Celecoxib",
        fp_type="ECFP4",
        threshold=1.0,
        rediscovery=True,
    )

    return guac_benchmark.wrapped_objective.score(smiles)