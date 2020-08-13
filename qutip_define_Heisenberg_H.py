#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:44:10 2020

@author: thsiao
"""


#%%
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import os
os.chdir("/home/thsiao/qutip_simulation_result/data")

#%% Define Pauli spin operators
num_site = 4 # number of sites
sz_list = [[]]*num_site # list for the S_z operators.
sp_list = [[]]*num_site # list for the S_+ operators.
sm_list = [[]]*num_site # list for the S_+ operators.
spin_op_list = [sz_list,sp_list,sm_list]
spin_op_type_list = [sigmaz(),sigmap(),sigmam()]
for idx_op, op_list in enumerate(spin_op_list):
    spin_op_type = spin_op_type_list[idx_op]
    for idx_site in range(num_site):
        tmp_list = []
        tmp_list.extend([identity(2)]*idx_site)
        tmp_list.extend([spin_op_type])
        tmp_list.extend([identity(2)]*(num_site-1-idx_site))
        op_list[idx_site] = tensor(tmp_list)/2
        
#%% Define Hamiltonian
def init_empty_H(num_site):
    # initialise an empty H matrix with dimensions corresponding to num_mode
    dim_list = [[2]*num_site,[1]*num_site]
    empty_H = ket2dm(zero_ket(2**(num_site),dim_list))
    return empty_H

def H_exchange(num_site, spin_op_list, j_list):
    # Create matrix for nearest-neighbor exchange H
    H_matrix = init_empty_H(num_site)
    for idx_site in range(num_site-1):
        j = j_list[idx_site] # J_(i,i+1)
        sz = spin_op_list[0][idx_site]
        sp = spin_op_list[1][idx_site]
        sm = spin_op_list[2][idx_site]
        sz_nn = spin_op_list[0][idx_site+1]
        sp_nn = spin_op_list[1][idx_site+1]
        sm_nn = spin_op_list[2][idx_site+1]
        H_matrix+=j*(2*(sp*sm_nn+sm*sp_nn)+sz*sz_nn)
    return H_matrix

def H_bfield(num_site, spin_op_list, b_list):
    # Create matrix for nearest-neighbor exchange H
    H_matrix = init_empty_H(num_site)
    for idx_site in range(num_site):
        b = b_list[idx_site] # J_(i,i+1)
        sz = spin_op_list[0][idx_site]
        H_matrix+=b*sz
    return H_matrix

#%% Define S, T0, T+, T-
def S_state():
    # create singlet state
    up_down = tensor([basis(2,0),basis(2,1)])
    down_up = tensor([basis(2,1),basis(2,0)])
    singlet_output = (up_down - down_up)/np.sqrt(2)
    return singlet_output

def T0_state():
    # create triplet0 state
    up_down = tensor([basis(2,0),basis(2,1)])
    down_up = tensor([basis(2,1),basis(2,0)])
    triplet_0_output = (up_down + down_up)/np.sqrt(2)
    return triplet_0_output

def TP_state():
    # create triplet+ state
    triplet_p_output = tensor([basis(2,0),basis(2,0)])
    return triplet_p_output

def TM_state():
    # create triplet- state
    triplet_m_output = tensor([basis(2,1),basis(2,1)])
    return triplet_m_output

#%% FFT function
def fft_func(signal_array,sampling_rate):
    freq_array = np.fft.fftfreq(len(signal_array),1/sampling_rate)
    signal_fft_array = np.abs(np.fft.fft(signal_array))
    return freq_array, signal_fft_array
#%% Pickle save data
def pickle_save(data):
    datetime_obj = datetime.datetime.now()
    file_name = datetime_obj.strftime("%Y%m%d_%H%M%S")+'_data'
    fp = open(file_name,'wb')
    pickle.dump(data, fp)
    print('data location : %s/%s' %(os.getcwd(),file_name))
    
def pickle_load(file_name):
    fp = open(file_name,'rb')
    loaded_data = pickle.load(fp)
    return loaded_data
#%% four-spin states
# Note: spin up = basis(2,0). spin down = basis(2,1)
SS_0 = tensor([S_state(),S_state()]) # S12*S34
SS_1 = (1/np.sqrt(3))*(tensor([TP_state(),TM_state()])+tensor([TM_state(),TP_state()])-tensor([T0_state(),T0_state()]))
SS_middle_0 = SS_0.permute([2,0,1,3]) # S_23*S_14
SS_middle_1 = SS_1.permute([2,0,1,3])
S_TP_0 = tensor([TP_state(),S_state()]) # TP_12*S_34
S_TP_1 = tensor([S_state(),TP_state()]) # S_12*TP_34
S_TP_2 = (1/np.sqrt(2))*(tensor([TP_state(),T0_state()])-tensor([T0_state(),TP_state()]))

#%% Test
j_list = [1,1,1]
b_list = [0,0,0,0]
H_test = H_exchange(num_site, spin_op_list, j_list) + H_bfield(num_site, spin_op_list, b_list)

# state_basis = [SS_0, SS_1]
state_basis = [S_TP_0, S_TP_2, S_TP_1]
H_test_matrix = np.zeros([len(state_basis),len(state_basis)], dtype=complex)
for idx_column in range(len(state_basis)):
    for idx_row in range(len(state_basis)):
        H_test_matrix[idx_row][idx_column] = H_test.matrix_element(state_basis[idx_row].dag(),state_basis[idx_column])