#!/usr/bin/env python3
import numpy as np

for state in ['Reference', 'Solution']:
    print('=== {} ==='.format(state))

    for conc, Natoms in zip(['0.00','2.00','4.00'], [0,368,792]):
        if state == 'Solution':
            V = np.loadtxt('PEG36mer/NaCl/'+conc+'/Solute/output.dat', usecols=(4), skiprows=1, unpack=True)
        elif state == 'Reference':
            V = np.loadtxt('References/NaCl/'+conc+'/output.dat', usecols=(4), skiprows=1, unpack=True)
        Na = 6.02214076*1e23 # [1/mol]
        nm3_to_l = 1e-24
        molarity = Natoms / (V.mean() * Na * nm3_to_l)
    
        print('The salt concentration is: {:.2f} given {} NaCl atoms'.format(molarity, Natoms))
    print('')
