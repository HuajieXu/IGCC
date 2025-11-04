from os import remove
from re import findall
from rdkit import Chem
from shutil import copy
from os.path import exists
from itertools import chain
from functools import lru_cache
from collections import Counter
from scipy.optimize import least_squares
from configparser import RawConfigParser

from _basic import Basic
from _CBH import CBH

import numpy as np
import time

@lru_cache(maxsize=128)
def get_input_parameters(main_path = '../..'):
    para = {'work_path': main_path}

    # Process the Capitalization issue.
    config = RawConfigParser()
    config.optionxform = lambda option: option
        
    config.read(f"{para['work_path']}/parameters.ini")
    para.update({k: eval(v) for k, v in config.items('input_parameters')})
    para.update({k: eval(v) for k, v in config.items('default_parameters')})

    if fitting_model == 1:
        if exists(f"{para['work_path']}/{'mBAC_parameters.txt'}"):
            with open(f"{para['work_path']}/{'mBAC_parameters.txt'}") as f:
                para.update({'mBAC_parameters': eval(f.read())})
        else:
            para.update({'mBAC_parameters': {}})
    else:
        para.update({'mBAC_parameters': {}})

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
def get_GFNFF_energy_from_out(inc):
    Tlist = [100, 150, 200, 250, 298.15] + list(range(300, 1000, 50)) + list(range(1000, 2000, 100)) + list(range(2000, 5001, 250))
    with open(f"{para['work_path']}/rawfiles/GFNFF/{inc}.out", encoding='utf-8') as f:
        content, E, g, p = f.read(), [], [], []
    try:
        aim_content = findall('origin\n([\s\S]+?)\nT', content)[-1]
    except:
        return {'Hf_conf_298/(kcal/mol)': 0.000, 'S_conf_298/(cal/mol/K)': 0.000, 'Cp_conf/(cal/mol/K)': dict(zip(Tlist, [0.000] * len(Tlist)))}

    for x in [x.split() for x in aim_content.split('\n')]:
        if len(x) > 5:
            E.append(x[1]), g.append(x[-2])
        p.append(x[3])
    E, g, T = np.array(E, float) * 1000 , np.array(g, int), np.array(Tlist, float)
    T = T.reshape(len(T), 1)
    E_beta = E / (1.987 * T)
    
    Hf_conf = float(findall('H\(T\)-H\(0\)\s+=\s+(\S+)', content)[-1])
    S_conf = float(findall('Sconf\s+=\s+(\S+)', content)[-1])
    Cp_298 = float(findall('Cp\(total\)\s+=\s+(\S+)', content)[-1])
    Cp_conf = 1.987 * np.dot(E_beta ** 2 * np.exp(- E_beta), g) / np.dot(np.exp(- E_beta), g) - 1.987 * (np.dot(E_beta * np.exp(- E_beta), g) / np.dot(np.exp(-E_beta), g)) ** 2
    
    if Cp_298:
        Cp_conf = Cp_conf * Cp_298 / Cp_conf[4]
    
    Cp_conf = dict(zip(Tlist, np.round(Cp_conf, 3)))

    return {'Hf_conf_298/(kcal/mol)': round(Hf_conf, 3), 'S_conf_298/(cal/mol/K)': round(S_conf, 3), 'Cp_conf/(cal/mol/K)': Cp_conf}

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
    Cv_RRHO_298, S_RRHO_298 = np.round(np.array(findall('\n Total\s+\d.+?(\S+)\s+(\S+)\n', content)[-1], float), 2)
    Freqs = np.array(''.join(findall(' Frequencies -- (.+?\n)', content)).split(), float)

    # Get Cp from Harmonic-Oscillator (HO) approximation.
    h = 6.62606957 * 10 ** -34
    kB = 1.3806488 * 10 ** -23
    c = 2.99792458 * 10 ** 10
    v = c * Freqs * 0.961
    func_Cv_vib = lambda T: sum(1.987 * np.exp(h * v / kB / T) * (h * v / kB / T / (np.exp(h * v / kB / T) - 1)) ** 2)
    Tlist = [100, 150, 200, 250, 298.15] + list(range(300, 1000, 50)) + list(range(1000, 2000, 100)) + list(range(2000, 5001, 250))
    Cp_RRHO = {T: round(func_Cv_vib(T) - func_Cv_vib(298.15) + Cv_RRHO_298 + 1.987, 2) for T in Tlist}
    Freqs = Freqs.tolist()
    return {'E_BYP/(Eh)': E_BYP, 'ZPE/(Eh)': ZPE, 'H_corr/(Eh)': H_corr,
            'S_RRHO_298/(cal/mol/K)': S_RRHO_298, 'Cp_RRHO/(cal/mol/K)': Cp_RRHO, 'Freqs/(cm-1)': Freqs}

@lru_cache(maxsize=128)
def get_CCSDT_energy_from_out(inc):
    with open(f"{para['work_path']}/rawfiles/CCSDT/{inc}.out") as f:
        content = f.read().replace('\n ', '')
    HF_DZ, CC_DZ, HF_TZ, CC_TZ = np.array(findall('\WHF=(-\d*\.\d+).+?CCSD\(T\)=(-\d*\.\d+)', content), dtype = float).flatten()
        
    HF_CBS = (HF_TZ * 3 ** 3.4 - HF_DZ * 2 ** 3.4) / (3 ** 3.4 - 2 ** 3.4)
    Corr_CBS = ((CC_TZ - HF_TZ) * 3 ** 2.4 - (CC_DZ - HF_DZ) * 2 ** 2.4) / (3 ** 2.4 - 2 ** 2.4)
    CC_CBS = HF_CBS + Corr_CBS
    HF_DZ, HF_TZ, CC_DZ, CC_TZ, HF_CBS, Corr_CBS, CC_CBS = map(lambda x: round(x, 7), [HF_DZ, HF_TZ, CC_DZ, CC_TZ, HF_CBS, Corr_CBS, CC_CBS])
    return {'HF_DZ/(Eh)': HF_DZ, 'HF_TZ/(Eh)': HF_TZ, 'CC_DZ/(Eh)': CC_DZ, 'CC_TZ/(Eh)': CC_TZ, 'HF_CBS/(Eh)': HF_CBS, 'Corr_CBS/(Eh)': Corr_CBS, 'CC_CBS/(Eh)': CC_CBS}

@lru_cache(maxsize=128)
def get_final_thermo_data(inc, cal_smi):
    atoms = Counter(x.GetSymbol() for x in Chem.AddHs(Chem.MolFromSmiles(cal_smi)).GetAtoms())
    Exp_SO = round((-0.14 / 1000) * len(findall('C|c', cal_smi)) + (-0.36 / 1000) * len(findall('O|o', cal_smi)), 7)

    data = {}

    for method in ['B3LYP', 'GFNFF', 'CCSDT']:
        data.setdefault(inc, {}).update(eval(f"get_{method}_energy_from_out('{inc}')"))

    H_mol = round(data[inc]['CC_CBS/(Eh)'] + data[inc]['H_corr/(Eh)'] + (0.981 - 0.961) * data[inc]['ZPE/(Eh)'] + Exp_SO, 7)

    Hf_CCSDT = 627.5095 * (H_mol - sum(v * para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) + sum(v * para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)']
    Hf_CCSDT = round(Hf_CCSDT, 2)
    
    return {'Hf_CCSDT/(kcal/mol)': Hf_CCSDT}

@lru_cache(maxsize=128)
def get_training_data(file = 'mBAC_training.txt'):

    with open(file) as f:
        all_smiles, std_smiles, all_benchmark, all_rawdata = [], [], [], []
        lines = f.readlines()

    for line in lines[1:]:
        if line.strip():
            line = line.strip()
            smi, benchmark = line.split()[1], line.split()[2]
            all_smiles.append(smi), all_benchmark.append(round(float(benchmark), 2))

            cal_smi, std_smi, inc = Basic(para).smi_to_std_format(smi)
            std_smiles.append(std_smi)
            
            if rawdata_model == 0:
                Hf_CCSDT = get_final_thermo_data(inc, cal_smi)['Hf_CCSDT/(kcal/mol)']
                all_rawdata.append(round(Hf_CCSDT, 2))
            else:
                all_rawdata.append(round(float(line.split()[3]), 2))
    
    with open('mBAC_EOF_data.txt', 'w') as f:
        f.write('%6s%60s%30s%22s%22s\n'%('S/N','SMILES', 'standard_SMILES', 'benchmark/(kcal/mol)', 'rawdata/(kcal/mol)'))
        for i, (smi, std_smi, benchmark, rawdata) in enumerate(zip(all_smiles, std_smiles, all_benchmark, all_rawdata)):
            f.write('%6s%60s%30s%22.2f%22.2f\n'%(i+1, smi, std_smi, benchmark, rawdata))

    if inspect_model == 1 and exists(f"{para['work_path']}/{file}"):
        vacant_smiles = []
        with open(f"{para['work_path']}/{file}") as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    smi = line.split()[1]
                    if smi not in std_smiles:
                        vacant_smiles.append(smi)

        if vacant_smiles:
            print(f"{' Missing training data! ':#^30}\n")
            with open('vacant_smiles.txt', 'w', newline='\n') as f:
                f.write(f"{'S/N':>6}{'Train SMILES':>60}\n")
                for i, v in enumerate(vacant_smiles):
                    f.write(f'{i+1:>6}{v:>60}\n')
        else:
            if exists('vacant_smiles.txt'):
                remove('vacant_smiles.txt')
    else:
        if exists('vacant_smiles.txt'):
            remove('vacant_smiles.txt')

    return tuple(all_smiles), tuple(all_benchmark), tuple(all_rawdata)

@lru_cache(maxsize=128)
def get_all_mol_bonds(smiles):
    all_bond_types, all_bonds, CBH_class = [], [], CBH(para)
    all_bond_types = chain(*map(CBH_class.get_mBAC_bonds, smiles))
    all_bond_types = list(para['mBAC_parameters']) + list(set(all_bond_types) - set(para['mBAC_parameters']))

    for smi in smiles:
        bonds = CBH_class.get_mBAC_bonds(smi)
        all_bonds.append([bonds.get(x, 0) for x in all_bond_types])

    return all_bond_types, all_bonds

@lru_cache(maxsize=128)
def get_fitted_parameters(input_data):
    smiles, benchmark, rawdata = list(input_data)
    benchmark, rawdata = np.array(benchmark, dtype = float), np.array(rawdata, dtype = float)
    bond_types, all_bonds = get_all_mol_bonds(tuple(smiles))

    xdata = np.array(all_bonds)
    ydata = rawdata - benchmark

    intime = time.time()


    if fitting_model == 0:
        parameters = - np.round(np.linalg.pinv(xdata).dot(ydata), 3)
    else:
        known_para_num = len(para['mBAC_parameters'])
        known_para = np.array([para['mBAC_parameters'][x] for x in bond_types[:known_para_num]])
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

    bond_parameters = dict(zip(bond_types, parameters))

    fitted_data = rawdata + np.dot(xdata, parameters)
    residual_error = fitted_data - benchmark

    with open('mBAC_fitting_data.txt','w') as f:
        print('%25s%6.2f kcal/mol'%('Largest deviation: ', np.absolute(residual_error).max()), file = f)
        print('%25s%6.2f kcal/mol'%('Mean absolute deviation: ', np.absolute(residual_error).mean()), file = f)
        print('%6s%60s%24s%24s%24s%24s%24s'%('S/N', 'SMILES', 'benchmark (kcal/mol)', 'raw data (kcal/mol)', 'fitted data (kcal/mol)', 'original error', 'residual error'), file = f)
        
        file_content = ''

        for i, smi in enumerate(smiles):
            file_content += '%6s%60s%24.2f%24.2f%24.2f%24.2f%24.2f\n'%(i+1, smi, benchmark[i], rawdata[i], fitted_data[i], rawdata[i] - benchmark[i], residual_error[i])
        print(file_content, file = f)

    return bond_parameters

@lru_cache(maxsize=128)
def write_mBAC_parameters(bond_parameters):
    bond_parameters = dict(bond_parameters)
    with open('mBAC_parameters.txt','w') as f:
        f.write(str(bond_parameters))

    if replace_model == 1:
        copy('mBAC_parameters.txt', para['work_path'])


######################################## Model Selection ########################################
# inspect_model 0: Do not inspect and only use the SMILES of mBAC_training.txt in current path.  #
# inspect_model 1: Inspect the vacant SMILES of mBAC_training.txt in main path and current path. #
# rawdata_model 0: Automatically obtain the raw EOFs for the given SMILES in mBAC_training.txt.  #
# rawdata_model 1: Set the raw EOFs to be equal to the provided values in mBAC_training.txt.     #
# fitting_model 0: Train the model from scratch and update all mBAC parameters in main path.     #
# fitting_model 1: Train based on existing mBAC parameters in main path and supply vacant ones.  #
# replace_model 0: Do not replace mBAC parameters in main path and refresh in the train path.    #
# replace_model 1: Replace mBAC parameters in main path according to the trained parameters.     #
#################################################################################################

inspect_model = 0 # Please select a proper inspect model parameter before training!
rawdata_model = 0 # Please select a proper rawdata model parameter before training!
fitting_model = 0 # Please select a proper fitting model parameter before training!
replace_model = 0 # Please select a proper replace model parameter before training!


main_path = '../..' # Please give the main path of TDOC program before training!


if __name__ == '__main__':
    
    print('# 1: Initializing program.\n')

    para = get_input_parameters(main_path)

    
    print('# 2: Getting training data.\n')

    input_data = get_training_data()

    print('# 3: Training mBAC parameters.\n')
    
    bond_parameters = get_fitted_parameters(input_data)

    print('# 4: Writing mBAC parameters.\n')
    
    write_mBAC_parameters(Basic(para).dict_to_tuple(bond_parameters))
  
    print('# 5: Exiting current program.\n')

