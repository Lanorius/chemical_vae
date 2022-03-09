# tensorflow backend
from os import environ

# vae stuff
from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
# import scientific py
import numpy as np
import pandas as pd

# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

import h5py
from silx.io.dictdump import dicttoh5
import matplotlib.pyplot as plt

import sys

environ['KERAS_BACKEND'] = 'tensorflow'

vae = VAEUtils(directory='./models/zinc_properties')
# vae = VAEUtils(directory='./')

smiles = pd.read_csv(sys.argv[1], header=0, index_col=None, sep='\t')
smiles.iloc[:, 0] = smiles.iloc[:, 0].astype(str)


encoded_smiles = {}

for i in range(smiles.shape[0]):
	print(i / smiles.shape[0])
	smiles_1 = mu.canon_smiles(smiles.iat[i, 1])
	X_1 = vae.smiles_to_hot(smiles_1, canonize_smiles=True)
	z_1 = vae.encode(X_1)
	encoded_smiles[smiles.iat[i, 0]] = z_1[0].astype('float32')
	
dicttoh5(encoded_smiles, h5file='smiles_embeddings_as_hd5.h5', h5path='/', mode='w', overwrite_data=None, create_dataset_args=None, update_mode=None)

