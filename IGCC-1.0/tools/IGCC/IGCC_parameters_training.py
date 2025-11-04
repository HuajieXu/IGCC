# -*- coding: utf-8 -*-
from os import remove
from re import findall
from rdkit import Chem
from shutil import copy
from os.path import exists
from functools import lru_cache
from collections import Counter
from scipy.optimize import least_squares
from configparser import RawConfigParser


from _CBH import CBH
from _basic import Basic

import numpy as np

@lru_cache(maxsize=128)
def get_input_parameters(main_path = '../..'):
    para = {'work_path': main_path}

    # Process the Capitalization issue.
    config = RawConfigParser()
    config.optionxform = lambda option: option
        
    config.read(f"{para['work_path']}/parameters.ini")
    para.update({k: eval(v) for k, v in config.items('default_parameters')})

    if fitting_model == 1:
        if exists(f"{para['work_path']}/{'IGCC_parameters.txt'}"):
            with open(f"{para['work_path']}/{'IGCC_parameters.txt'}") as f:
                para.update({'IGCC_parameters': eval(f.read())})
        else:
            para.update({'IGCC_parameters': {}})
    else:
        para.update({'IGCC_parameters': {}})

    if exists(f"{para['work_path']}/same_inchikey.txt"):
        with open(f"{para['work_path']}/same_inchikey.txt") as f:
            para.update({'same_inchikey': eval(f.read().replace(',\n', ','))})
    else:
        para.update({'same_inchikey':{}})

    if exists(f"{para['work_path']}/geometry_state.txt"):
        with open(f"{para['work_path']}/geometry_state.txt") as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    _, inc, tar_smi, cur_smi, state = line.split()
                    if exists(f"{para['work_path']}/rawfiles/B3LYP/{inc}.out"):
                        para.setdefault('geometry_state', {}).update({inc: [tar_smi, cur_smi, state]})
    else:
        para.update({'geometry_state':{}})

    return para

@lru_cache(maxsize=128)
def get_final_thermo_data(smi, typ = 'CBS', rung = 3):
    cal_smi, std_smi, inc = Basic(para).smi_to_std_format(smi)

    atoms = Counter(x.GetSymbol() for x in Chem.AddHs(Chem.MolFromSmiles(cal_smi)).GetAtoms())
    Exp_SO = round((-0.14 / 1000) * len(findall('C|c', cal_smi)) + (-0.36 / 1000) * len(findall('O|o', cal_smi)), 7)
    data = {}
    reac, prod, reaction = CBH(para).get_CBH_reactions(std_smi, rung)

    species = {x: Basic(para).smi_to_std_format(x)[-1] for x in reac + prod}
    for method in ['B3LYP', 'GFNFF', 'MP2', 'CCSDT']:
        for species_inc in species.values():
            data.setdefault(species_inc, {}).update(eval(f"get_{method}_energy_from_out('{species_inc}')"))

    if typ == 'CBS':
        H_mol = round(data[inc]['CC_CBS/(Eh)'] + data[inc]['H_corr/(Eh)'] + (0.981 - 0.961) * data[inc]['ZPE/(Eh)'] + Exp_SO, 7)
    else:
        E_reac = sum(v * (data[species[k]]['CC_CBS/(Eh)'] - data[species[k]]['E_MP2/(Eh)']) for k, v in (reac - Counter([std_smi])).items())
        E_prod = sum(v * (data[species[k]]['CC_CBS/(Eh)'] - data[species[k]]['E_MP2/(Eh)']) for k, v in prod.items())
        H_mol = round(data[inc]['E_MP2/(Eh)'] + E_prod - E_reac + data[inc]['H_corr/(Eh)'] + (0.981 - 0.961) * data[inc]['ZPE/(Eh)'] + Exp_SO, 7)

    Hf_MP2 = 627.5095 * (data[inc]['E_MP2/(Eh)'] - sum(v * para['h_atoms']['MP2'][k] for k, v in atoms.items()) + sum(v * para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)']
    Hf_CCSDT = 627.5095 * (H_mol - sum(v * para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) + sum(v * para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)']
    Hf_CCSDT = round(Hf_CCSDT, 2)
    
    return {'Hf_MP2/(kcal/mol)': Hf_MP2, 'Hf_CCSDT/(kcal/mol)': Hf_CCSDT}

@lru_cache(maxsize=128)
def get_GFNFF_energy_from_out(inc):
    Tlist = [100, 150, 200, 250, 298.15] + list(range(300, 1000, 50)) + list(range(1000, 2000, 100)) + list(range(2000, 5001, 250))
    with open(f"{para['work_path']}/rawfiles/GFNFF/{inc}.out", encoding='utf-8') as f:
        content, E, g, p = f.read(), [], [], []
    try:
        aim_content = findall('origin\n([\s\S]+?)\nT', content)[-1]
    except:
        return {'Hf_conf_298/(kcal/mol)': 0.000}
    
    Hf_conf = float(findall('H\(T\)-H\(0\)\s+=\s+(\S+)', content)[-1])

    return {'Hf_conf_298/(kcal/mol)': round(Hf_conf, 3)}

@lru_cache(maxsize=128)
def get_B3LYP_energy_from_out(inc):
    with open(f"{para['work_path']}/rawfiles/B3LYP/{inc}.out") as f:
        content = f.read()
    E_BYP, H_BYP = np.round(np.array(findall('\WHF=(-\d*\.\d+).+?HTot=(-\d*\.\d+)', content.replace('\n ', ''))[-1], float), 7)
    
    try:
        ZPE = round(float(findall('ZeroPoint=(\d*\.\d+)', content.replace('\n ', ''))[-1]), 7)
    except:
        ZPE = 0
    
    H_corr = round(H_BYP - E_BYP, 7)
    
    return {'ZPE/(Eh)': ZPE, 'H_corr/(Eh)': H_corr}

@lru_cache(maxsize=128)
def get_MP2_energy_from_out(inc):
    with open(f"{para['work_path']}/rawfiles/MP2/{inc}.out") as f:
        content = f.read()
    SPE = float(findall('MP2=(-*\w*\.\w*)', (content.replace('\n ', '')))[-1])
    
    return {'E_MP2/(Eh)': SPE}

@lru_cache(maxsize=128)
def get_CCSDT_energy_from_out(inc):
    with open(f"{para['work_path']}/rawfiles/CCSDT/{inc}.out") as f:
        content = f.read().replace('\n ', '')
    HF_DZ, CC_DZ, HF_TZ, CC_TZ = np.array(findall('\WHF=(-\d*\.\d+).+?CCSD\(T\)=(-\d*\.\d+)', content), dtype = float).flatten()
        
    HF_CBS = (HF_TZ * 3 ** 3.4 - HF_DZ * 2 ** 3.4) / (3 ** 3.4 - 2 ** 3.4)
    Corr_CBS = ((CC_TZ - HF_TZ) * 3 ** 2.4 - (CC_DZ - HF_DZ) * 2 ** 2.4) / (3 ** 2.4 - 2 ** 2.4)
    CC_CBS = round(HF_CBS + Corr_CBS, 7)
    
    return {'CC_CBS/(Eh)': CC_CBS}

@lru_cache(maxsize=128)
def get_training_data(file = 'IGCC_training.txt'):

    with open(file) as f:
        all_smiles, std_smiles, all_MP2, all_CBH1, all_CBH2, all_CBH3, all_CCSDT = [], [], [], [], [], [], []
        lines = f.readlines()

    for line in lines[1:]:
        if line.strip():
            line = line.strip()
            smi = line.split()[1]
            all_smiles.append(smi)

            bonds = CBH(para).get_CBH_delta_bonds(smi, 3)[-1]

            cal_smi, std_smi, inc = Basic(para).smi_to_std_format(smi)
            std_smiles.append(std_smi)

            if rawdata_model == 0:
                Hf_MP2 = get_final_thermo_data(std_smi, 'CBS')['Hf_MP2/(kcal/mol)']
                Hf_CCSDT_CBH1 = get_final_thermo_data(std_smi, 'CBH', 1)['Hf_CCSDT/(kcal/mol)']
                Hf_CCSDT_CBH2 = get_final_thermo_data(std_smi, 'CBH', 2)['Hf_CCSDT/(kcal/mol)']
                Hf_CCSDT_CBH3 = get_final_thermo_data(std_smi, 'CBH', 3)['Hf_CCSDT/(kcal/mol)']
                Hf_CCSDT = get_final_thermo_data(std_smi, 'CBS')['Hf_CCSDT/(kcal/mol)']
            else:
                all_MP2, all_CBH1, all_CBH2, all_CBH3, all_CCSDT = map(float, line.split()[1:]) 
            
            all_MP2.append(round(Hf_MP2, 2))
            all_CBH1.append(round(Hf_CCSDT_CBH1, 2))
            all_CBH2.append(round(Hf_CCSDT_CBH2, 2))
            all_CBH3.append(round(Hf_CCSDT_CBH3, 2))
            all_CCSDT.append(round(Hf_CCSDT, 2))
            
    with open('IGCC_EOF_data.txt', 'w') as f:
        f.write('%6s%60s%60s%22s%22s%22s%22s%22s\n'%('S/N', 'SMILES', 'standard_SMILES', 'MP2/(kcal/mol)', 'CBH1/(kcal/mol)', 'CBH2/(kcal/mol)', 'CBH3/(kcal/mol)', 'CCSDT/(kcal/mol)'))
        for i, (smi, std_smi, MP2, CBH1, CBH2, CBH3, CCSDT) in enumerate(zip(all_smiles, std_smiles, all_MP2, all_CBH1,all_CBH2, all_CBH3, all_CCSDT)):
            f.write('%6s%60s%60s%22.2f%22.2f%22.2f%22.2f%22.2f\n'%(i+1, smi, std_smi, MP2, CBH1, CBH2, CBH3, CCSDT))

    return [std_smiles, all_MP2, all_CBH1, all_CBH2, all_CBH3, all_CCSDT]


def get_all_mol_bonds(std_smiles):
    all_bond_types, all_bonds, CBH_class = [], [], CBH(para)
    
    for smi in std_smiles:

        for i in [1, 2, 3]:
            bonds = CBH_class.get_CBH_delta_bonds(smi, i)[-1]
            all_bond_types.extend(set(bonds))

    all_bond_types = list(para['IGCC_parameters']) + list(set(all_bond_types) - set(para['IGCC_parameters']))

    for smi in std_smiles:
        for i in [1, 2, 3]:
            bonds = CBH_class.get_CBH_delta_bonds(smi, i)[-1]

            all_bonds.append([bonds.get(x, 0) for x in all_bond_types])
    
    return all_bond_types, all_bonds


def get_fitted_parameters(input_data):
    std_smiles, all_MP2, all_CBH1, all_CBH2, all_CBH3, all_CCSDT = input_data

    bond_types, all_bonds = get_all_mol_bonds(std_smiles)

    xdata = np.array(all_bonds)
    ydata = (np.array([all_CBH1, all_CBH2, all_CBH3]) - np.array([all_CCSDT, all_CCSDT, all_CCSDT])).T.flatten()

    if fitting_model == 0:
        parameters = - np.round(np.linalg.pinv(xdata).dot(ydata), 3)
    else:
        known_para_num = len(para['BAC_parameters'])
        known_para = np.array([para['BAC_parameters'][x] for x in bond_types[:known_para_num]])
        initial_guess = np.zeros(len(bond_types) - known_para_num)
        residual = lambda params, X, Y, known_para: X.dot(np.concatenate([known_para, params])) - Y
        
        try:
            result = least_squares(residual, initial_guess, args=(xdata, ydata, known_para), bounds=[-10, 10])
            if not result.success:
                raise ValueError("Optimization failed: " + result.message)
            fitted_para = result.x
        except ValueError as e:
            raise ValueError("!!! Error: Optimization failed. !!! ")
        parameters = - np.round(np.concatenate([known_para, fitted_para]), 3)
    
    bond_parameters = dict(zip(bond_types, parameters.flatten()))
    
    benchmark = all_CCSDT
    rawdata = all_CBH3
    fitted_data = rawdata + np.dot(xdata, parameters)[2::3]
    residual_error = fitted_data - benchmark

    with open('IGCC_fitting_data.txt','w') as f:
        print('%25s%6.2f kcal/mol'%('Largest deviation: ', np.absolute(residual_error).max()), file = f)
        print('%25s%6.2f kcal/mol'%('Mean absolute deviation: ', np.absolute(residual_error).mean()), file = f)
        print('%6s%60s%24s%24s%24s%24s%24s'%('S/N', 'SMILES', 'benchmark (kcal/mol)', 'raw data (kcal/mol)', 'fitted data (kcal/mol)', 'original error', 'residual error'), file = f)
        
        file_content = ''
        for i, smi in enumerate(std_smiles):
            file_content += '%6s%60s%24.2f%24.2f%24.2f%24.2f%24.2f\n'%(i+1, smi, benchmark[i], rawdata[i], fitted_data[i], rawdata[i] - benchmark[i], residual_error[i])
        print(file_content, file = f)

    return bond_parameters



def write_IGCC_parameters(bond_parameters):
    with open('IGCC_parameters.txt','w') as f:
        f.write(str(bond_parameters))

    if replace_model == 1:
        copy('IGCC_parameters.txt', para['work_path'])

######################################## Model Selection ########################################
# rawdata_model 0: Automatically obtain the raw EOFs for the given SMILES in IGCC_training.txt.  #
# rawdata_model 1: Set the raw EOFs to be equal to the provided values in IGCC_training.txt.     #
# fitting_model 0: Train the model from scratch and update all IGCC parameters in main path.     #
# fitting_model 1: Train based on existing IGCC parameters in main path and supply vacant ones.  #
# replace_model 0: Do not replace IGCC parameters in main path and refresh in the train path.    #
# replace_model 1: Replace IGCC parameters in main path according to the trained parameters.     #
#################################################################################################

rawdata_model = 0 # Please select a proper rawdata model parameter before training!
fitting_model = 0 # Please select a proper fitting model parameter before training!
replace_model = 0 # Please select a proper replace model parameter before training!

main_path = '../..' # Please give the main path of TDOC program before training!


if __name__ == '__main__':

    #################  Fitting  #################

    print('# 1: Initializing program.\n')

    para = get_input_parameters(main_path)
    
    print('# 2: Getting training data.\n')

    input_data = get_training_data()

    print('# 3: Training IGCC parameters.\n')
    bond_parameters = get_fitted_parameters(input_data)

    print('# 4: Writing IGCC parameters.\n')
    
    write_IGCC_parameters(bond_parameters)

    print('# 5: Exiting current program.\n')
