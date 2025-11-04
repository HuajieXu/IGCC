# -*- coding: utf-8 -*-
from rdkit import Chem
from shutil import copy
from re import findall, sub
from collections import Counter
from openbabel import openbabel
from os import chdir, listdir, makedirs

import numpy as np

from igcc._check import Check
from igcc._conformer import Conformer
from igcc._basic import Basic


class Submit(object):
    
    """ To initiate parameters for Submit. """
    def __init__(self, para):
        self.para = para
        self.basic = Basic(self.para)



    """ To get submitted file of B3LYP. """
    def get_B3LYP_out(self):
        
        # Go to working directory and determine absent molecules.
        chdir(f"{self.para['work_path']}/rawfiles/B3LYP")
        inchikeys = []
        for inc in self.para['species']:
            if f'{inc}.dat' not in listdir(f"{self.para['work_path']}/datfiles/B3LYP"):
                state = self.para['geometry_state'].get(inc, ['N'])[-1]
                if state != 'Y':
                    inchikeys.append(inc)
        
        # Check whether calculations are completed.
        Check(self.para, inchikeys, 'B3LYP').check_method_out()

        for inc in inchikeys:
            if f'{inc}.gjf' not in listdir('.'):
                self.build_structure_from_inc(inc, self.para['conformer_method'])
                print(f'This molecule has been constructed: {inc}')

        # Generate submitted scripts.
        submitted_inchikeys = [inc for inc in inchikeys if f'{inc}.gjf' in listdir('.') and f'{inc}.out' not in listdir('.')]
        if submitted_inchikeys:
            
            # Update submitted process.
            for inchikeys in submitted_inchikeys:
                with open(f'{inchikeys}.gjf') as f:
                    content = sub('%nprocshared=(\d+)', f"%nprocshared={self.para['number_process']}", f.read())
                with open(f'{inchikeys}.gjf', 'w') as f:
                    f.write(content)
            self.write_submitted_script(submitted_inchikeys, 'B3LYP')

        # Output the file of geometry state for all species. 
        with open(f"{self.para['work_path']}/geometry_state.txt", 'w') as f:
            f.write(f"{'S/N':<6}{'inchikeys':<32}{'target smiles':>64}{'current smiles':>64}{'checked state':>16}\n")
            for i, inc in enumerate(sorted(self.para['geometry_state'])):
                f.write(f"{i+1:<6}{inc:<32}{self.para['geometry_state'][inc][0]:>64}{self.para['geometry_state'][inc][1]:>64}{self.para['geometry_state'][inc][2]:>16}\n")

        # Give a prompt message on whether to manually check the geometry between the target smiles and the current smiles.
        suspected_smiles = [k for k, v in self.para['geometry_state'].items() if v[-1] == 'N' and k in self.para['species']]
        if suspected_smiles:
            print(f"\n**** Suspected smiles! ****\n\n{chr(10).join(' '.join(f'{y:30}' for y in x) for x in [suspected_smiles[i*2:i*2+2] for i in range(len(suspected_smiles)//2+1)])}\n")


    
    """ To get submitted file of GFNFF. """
    def get_GFNFF_out(self):
        
        # Go to working directory and determine absent molecules.
        chdir(f"{self.para['work_path']}/rawfiles/GFNFF")
        inchikeys = []
        for inc in self.para['species']:
            if f'{inc}.dat' not in listdir(f"{self.para['work_path']}/datfiles/MP2"):
                inchikeys.append(inc)
        
        # Check whether calculations are completed.
        special_molecules = Check(self.para, inchikeys, 'GFNFF').check_method_out()
        submitted_inchikeys = [inc for inc in inchikeys if f'{inc}.out' not in listdir('.')]

        # Construct input files of GFNFF for CREST.
        for inc in inchikeys:
            state = self.para['geometry_state'].get(inc, ['N'])[-1]
            if f'{inc}.out' in listdir('../B3LYP') and f'{inc}.xyz' not in listdir('.') and state == 'Y':
                with open(f'../B3LYP/{inc}.out') as f, open(f'{inc}.xyz', 'w', newline = '\n') as p:
                    content = f.read()
                    res = findall('Charge =\s+(-*\d) Multiplicity =\s*(\d+)[\s\S]+?Standard[\s\S]+?Z\n -+\n((.{70}\n)+?) -+', content)[-1]
                    coord = ''.join('%-3s%15s%15s%15s\n'%(openbabel.GetSymbol(int(x.split()[1])), *x.split()[3:]) for x in res[2].split('\n')[:-1])
                    p.write(f"{len(coord.strip().split(chr(10)))}\n\n{coord}")
                    print(f'This molecule has been converted: {inc}')    
        
        # Generate submitted scripts.
        submitted_inchikeys = [inc for inc in inchikeys if f'{inc}.xyz' in listdir('.') and f'{inc}.out' not in listdir('.')]
        if submitted_inchikeys:
            self.write_submitted_script(submitted_inchikeys, 'GFNFF', special_molecules)



    """ To get submitted file of MP2. """
    def get_MP2_out(self):

        # Go to working directory and determine absent molecules.
        chdir(f"{self.para['work_path']}/rawfiles/MP2")
        inchikeys = []
        for inc in self.para['species']:
            if f'{inc}.dat' not in listdir(f"{self.para['work_path']}/datfiles/MP2"):
                inchikeys.append(inc)
 
        # Check whether calculations are completed.
        Check(self.para, inchikeys, 'MP2').check_method_out()

        # Construct input files of CCSDT for Gaussian.
        for inc in inchikeys :
            state = self.para['geometry_state'].get(inc, ['N'])[-1]
            if f'{inc}.out' in listdir('../B3LYP') and f'{inc}.gjf' not in listdir('.') and state == 'Y':
                with open(f'../B3LYP/{inc}.out') as f, open(f'../B3LYP/{inc}.gjf') as p, open(f'{inc}.gjf', 'w', newline = '\n') as q:
                    content = f.read()
                    cal_smi = self.basic.smi_to_std_format(self.para['species'][inc])[0]
                    
                    res = findall('Charge =\s+(-*\d) Multiplicity =\s*(\d+)[\s\S]+?Standard[\s\S]+?Z\n -+\n((.{70}\n)+?) -+', content)[-1]
                    charge, spin = res[:2]
                    cal_spin = sum(x.GetNumRadicalElectrons() for x in Chem.MolFromSmiles(cal_smi).GetAtoms()) + 1
                    MP2_level = 'ROMP2' if 'RO' in p.read() else 'MP2'
                    coord = ''.join('%-3s%15s%15s%15s\n'%(openbabel.GetSymbol(int(x.split()[1])), *x.split()[3:]) for x in res[2].split('\n')[:-1])
                    N_heavyatoms = max(Chem.MolFromSmiles(cal_smi).GetNumHeavyAtoms(), 1)
                    memory = N_heavyatoms * self.para['number_process'] * self.para['MP2_mem_per_core']
                    
                    # Add additional keywords to Multiple free radicals.
                    if len(findall('\[', self.para['species'][inc])) > 1 and int(spin) > 2 or cal_spin < int(spin):
                        additional_keywords = 'guess=mix'
                    else:
                        additional_keywords = ''
                    
                    # Write input content.
                    q.write(f"%nprocshared={self.para['number_process']}\n%mem={memory}MB\n#T {MP2_level}/cc-pVDZ symm=veryloose"
                        f" {additional_keywords}\n\n{self.para['species'][inc]}\n\n{charge} {spin}\n{''.join(coord)}\n\n")
                    print(f'This molecule has been converted: {inc}')

        # Generate submitted scripts.
        submitted_inchikeys = [inc for inc in inchikeys if f'{inc}.gjf' in listdir('.') and f'{inc}.out' not in listdir('.')]
        if submitted_inchikeys:
            # Update submitted process.
            for inchikeys in submitted_inchikeys:
                with open(f'{inchikeys}.gjf') as f:
                    content = sub('%nprocshared=(\d+)', f"%nprocshared={self.para['number_process']}", f.read())
                with open(f'{inchikeys}.gjf', 'w') as f:
                    f.write(content)
            self.write_submitted_script(submitted_inchikeys, 'MP2')

    
    """ To get submitted file of CCSD(T). """
    def get_CCSDT_out(self):

        # Go to working directory and determine absent molecules.
        chdir(f"{self.para['work_path']}/rawfiles/CCSDT")
        inchikeys = []
        for inc in self.para['micro_mol']:
            if f'{inc}.dat' not in listdir(f"{self.para['work_path']}/datfiles/CCSDT"):
                inchikeys.append(inc)
 
        # Check whether calculations are completed.
        Check(self.para, inchikeys, 'CCSDT').check_method_out()

        # Construct input files of CCSDT for Gaussian.
        for inc in inchikeys:
            state = self.para['geometry_state'].get(inc, ['N'])[-1]
            if f'{inc}.out' in listdir('../B3LYP') and f'{inc}.gjf' not in listdir('.') and state == 'Y':
                with open(f'../B3LYP/{inc}.out') as f, open(f'../B3LYP/{inc}.gjf') as p, open(f'{inc}.gjf', 'w', newline = '\n') as q:
                    content = f.read()
                    cal_smi = self.basic.smi_to_std_format(self.para['species'][inc])[0]
                    
                    res = findall('Charge =\s+(-*\d) Multiplicity =\s*(\d+)[\s\S]+?Standard[\s\S]+?Z\n -+\n((.{70}\n)+?) -+', content)[-1]
                    charge, spin = res[:2]
                    cal_spin = sum(x.GetNumRadicalElectrons() for x in Chem.MolFromSmiles(cal_smi).GetAtoms()) + 1
                    CCSDT_level = 'ROCCSD(T)' if 'RO' in p.read() else 'CCSD(T)'
                    coord = ''.join('%-3s%15s%15s%15s\n'%(openbabel.GetSymbol(int(x.split()[1])), *x.split()[3:]) for x in res[2].split('\n')[:-1])
                    N_heavyatoms = max(Chem.MolFromSmiles(cal_smi).GetNumHeavyAtoms(), 1)
                    memory = N_heavyatoms * self.para['number_process'] * self.para['CCSDT_mem_per_core']

                    # Add additional keywords to Multiple free radicals.
                    if len(findall('\[', self.para['species'][inc])) > 1 and int(spin) > 2 or cal_spin < int(spin):
                        additional_keywords = 'guess=mix'
                    else:
                        additional_keywords = ''
                    
                    # Write input content.
                    q.write(f"%nprocshared={self.para['number_process']}\n%mem={memory}MB\n%chk={inc}.chk\n#T {CCSDT_level}/cc-pVDZ symm=veryloose"
                        f" {additional_keywords}\n\n{self.para['species'][inc]}\n\n{charge} {spin}\n{''.join(coord)}\n\n")
                    q.write(f"--link1--\n%nprocshared={self.para['number_process']}\n%mem={memory}MB\n%chk={inc}.chk\n#T {CCSDT_level}/cc-pVTZ"
                        f" symm=veryloose geom=allcheck {additional_keywords}\n\n")
                    print(f'This molecule has been converted: {inc}')

        # Generate submitted scripts.
        submitted_inchikeys = [inc for inc in inchikeys if f'{inc}.gjf' in listdir('.') and f'{inc}.out' not in listdir('.')]
        if submitted_inchikeys:
            # Update submitted process.
            for inchikeys in submitted_inchikeys:
                with open(f'{inchikeys}.gjf') as f:
                    content = sub('%nprocshared=(\d+)', f"%nprocshared={self.para['number_process']}", f.read())
                with open(f'{inchikeys}.gjf', 'w') as f:
                    f.write(content)
            self.write_submitted_script(submitted_inchikeys, 'CCSDT')


 
    """ To construct 3D coordinates from inchikeys. """
    def build_structure_from_inc(self, inc, structure_method = 1):

        # Build conformational structure.
        gjf = Conformer(inc, self.para, structure_method).output_standard_gjf()

        with open(f'{inc}.gjf', 'w', newline='\n') as f:
            f.write(gjf)
    
    
    """ To write submitted scripts for linux systems. """
    def write_submitted_script(self, inchikeys, method, special_molecules = []):

        # Generate the submitted directory.
        makedirs(f"{self.para['work_path']}/subfiles/{method}", True)
        
        # Generate sumbmitted scripts for CREST.
        if method == 'GFNFF':
            for i, v in enumerate([x for x in np.array_split(inchikeys, self.para['number_task']) if x.size]):
                with open(f"{self.para['work_path']}/subfiles/GFNFF/job{i + 1}.sh", 'w', newline = '\n') as f:
                    f.write('#!/bin/bash\n\n')
                    if self.para['submitted_type'] == 3:
                        task_node = f"#SBATCH -N {self.para['task_node']}\n" if self.para['task_node'] == '1' else f"#SBATCH -w {self.para['task_node']}\n"
                        f.write(f"#SBATCH -J {method}-job{i + 1}\n#SBATCH -p {self.para['task_queue']}\n{task_node}#SBATCH -n {self.para['number_process']}\n"
                                f'#SBATCH -t 3600:00:00\n\ncd $SLURM_SUBMIT_DIR\nmkdir -p $SLURM_JOB_ID\ncd $SLURM_JOB_ID\n\n')
                        for inc in v:
                            with open(f'../B3LYP/{inc}.gjf') as p:
                                charge, spin = map(int, findall('\n\n(-*\d)\s+(\d+)\s*\n', p.read())[0])
                            
                            if inc not in special_molecules:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")
                            else:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -cbonds -noreftopo -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")

                        f.write(f'\nrm -rf $SLURM_SUBMIT_DIR/$SLURM_JOB_ID\n')
                    elif self.para['submitted_type'] == 2:
                        f.write(f"#PBS -N {method}-job{i + 1}\n#PBS -q {self.para['task_queue']}\n#PBS -l nodes={self.para['task_node']}:ppn={self.para['number_process']}\n"
                                f'#PBS -l walltime=3600:00:00\n#PBS -j oe\n\ncd $PBS_O_WORKDIR\nmkdir -p $PBS_JOBID\ncd $PBS_JOBID\n\n')
                        for inc in v:
                            with open(f'../B3LYP/{inc}.gjf') as p:
                                charge, spin = map(int, findall('\n\n(-*\d)\s+(\d+)\s*\n', p.read())[0])
                            
                            if inc not in special_molecules:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")
                            else:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -cbonds -noreftopo -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")

                        f.write(f'\nrm -rf $PBS_O_WORKDIR/$PBS_JOBID\n')
                    else:
                        f.write(f"mkdir -p {i + 1}\ncd {i + 1}\n\n")
                        for inc in v:
                            with open(f'../B3LYP/{inc}.gjf') as p:
                                charge, spin = map(int, findall('\n\n(-*\d)\s+(\d+)\s*\n', p.read())[0])
                            
                            if inc not in special_molecules:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")
                            else:
                                f.write(f"cp ../{inc}.xyz .\ncrest {inc}.xyz >{inc}.out -gfnff -cbonds -noreftopo -entropy -T {self.para['number_process']}"
                                        f" -chrg {charge} -uhf {spin - 1}\nwait\ncp {inc}.out ..\n\n")

                        f.write(f'cd..\nrm -rf {i + 1}\n')
        
        # Generate sumbmitted scripts for Gaussian.
        else:
            for i, v in enumerate([x for x in np.array_split(inchikeys, self.para['number_task']) if x.size]):
                with open(f"{self.para['work_path']}/subfiles/{method}/job{i + 1}.sh", 'w', newline = '\n') as f:
                    f.write('#!/bin/bash\n\n')
                    if self.para['submitted_type'] == 3:
                        task_node = f"#SBATCH -N {self.para['task_node']}\n" if self.para['task_node'] == '1' else f"#SBATCH -w {self.para['task_node']}\n"
                        f.write(f"#SBATCH -J {method}-job{i + 1}\n#SBATCH -p {self.para['task_queue']}\n{task_node}#SBATCH -n {self.para['number_process']}\n"
                                f'#SBATCH -t 3600:00:00\n\ncd $SLURM_SUBMIT_DIR\nmkdir -p $SLURM_JOB_ID\ncd $SLURM_JOB_ID\n\n')
                        for inc in v:
                            f.write(f'cp ../{inc}.gjf .\ng16 <{inc}.gjf >{inc}.out\nwait\ncp {inc}.out ..\n\n')
                        f.write(f'\nrm -rf $SLURM_SUBMIT_DIR/$SLURM_JOB_ID\n')
                    elif self.para['submitted_type'] == 2:
                        f.write(f"#PBS -N {method}-job{i + 1}\n#PBS -q {self.para['task_queue']}\n#PBS -l nodes={self.para['task_node']}:ppn={self.para['number_process']}\n"
                                f'#PBS -l walltime=3600:00:00\n#PBS -j oe\n\ncd $PBS_O_WORKDIR\nmkdir -p $PBS_JOBID\ncd $PBS_JOBID\n\n')
                        for inc in v:
                            f.write(f'cp ../{inc}.gjf .\ng16 <{inc}.gjf >{inc}.out\nwait\ncp {inc}.out ..\n\n')
                        f.write(f'\nrm -rf $PBS_O_WORKDIR/$PBS_JOBID\n')
                    else:
                        for inc in v:
                            f.write(f"cp ../{inc}.gjf .\nnohup g16 <{inc}.gjf >{inc}.out&\nwait\ncp {inc}.out ..\n\n")
                        f.write(f'cd..\nrm -rf {i + 1}\n')

        # Prepare related files to the submitted directory.
        with open(f"{self.para['work_path']}/subfiles/{method}/job-sub", 'w', newline = '\n') as f:
            if self.para['submitted_type'] == 3:
                f.write('#!/bin/bash\n\nfor s in *.sh;\ndo\n sbatch $s;\ndone\n')
            elif self.para['submitted_type'] == 2:
                f.write('#!/bin/bash\n\nfor s in *.sh;\ndo\n qsub $s;\ndone\n')
            else:
                f.write('#!/bin/bash\n\nfor s in *.sh;\ndo\n nohup bash $s &;\ndone\n')
        if method == 'GFNFF':
            for inc in inchikeys:
                copy(inc + '.xyz', '{}/subfiles/{}'.format(self.para['work_path'], method))
        else:
            for inc in inchikeys:
                copy(f'{inc}.gjf', f"{self.para['work_path']}/subfiles/{method}")



    """ To construct submitted files for Linux system. """
    def get_submitted_out(self):
        print('\nProcess submited files ...\n')
        self.get_B3LYP_out()
        self.get_GFNFF_out()
        self.get_MP2_out()
        self.get_CCSDT_out()