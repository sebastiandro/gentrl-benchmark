import gentrl
import torch
from moses.metrics.utils import get_mol
import pandas as pd
import pickle
import moses
from moses.utils import CharVocab
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

# Load vocab
dataset_path = "../data/moses_qed_props.csv.gz"
df = pd.read_csv(dataset_path, compression="gzip")
vocab = CharVocab.from_data(df['SMILES'])

enc = gentrl.RNNEncoder(vocab, latent_size=50)
dec = gentrl.DilConvDecoder(vocab, latent_input_size=50, split_len=100)
model = gentrl.GENTRL(enc, dec, 50 * [('c', 20)], [('c', 20)], beta=0.001)
model.cuda()

torch.cuda.set_device(0)

moses_qed_props_model_path = "../models/moses/"
model.load(moses_qed_props_model_path)
model.cuda()

import random
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

generated = []
verbose_lim = 10000

print("Sampling smiles", flush=True)

while len(generated) < 30000:
    sampled = model.sample(1000)
    sampled_valid = [s for s in sampled if get_mol(s)]
    
    generated += sampled_valid
    n_generated = len(generated)
    if n_generated >= verbose_lim:
        print("Generated %d of %d SMILES" % (len(generated), 30000), flush=True)
        verbose_lim += 10000

with open("../moses_sampling/sampled_smiles.csv", "w") as f:
    f.writelines("%s\n" % sm for sm in generated)

print("Calculating Metrics", flush=True)
metrics = moses.get_all_metrics(generated)
pickle.dump( metrics, open( "metrics.pkl", "wb" ) )