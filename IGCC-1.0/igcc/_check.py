# -*- coding: utf-8 -*-
from glob import glob
from rdkit import Chem
from numpy import array
from os.path import getsize
from re import findall, sub
from os import listdir, remove
from openbabel import openbabel
from pymatgen.core import Molecule
from rdkit import DataStructs
from rdkit.Chem import AllChem

from igcc._conformer import Conformer

import igcc._submit




class Check(object):

    """ To initiate parameters for Check. """
    def __init__(self, para, inchikeys, method):
        self.para = para
        self.inchikeys = inchikeys
        self.method = method



    """ To check whether the calculations of related methods are completed. """
    def check_method_completed(self):
        unfinish_inchikeys = [inc for inc in self.inchikeys if f'{inc}.out' not in listdir('.')]
        
        if unfinish_inchikeys:
            print(f"\n{f' {self.method} incompleted! ':*^30}\n")
        
        else:
            print(f"\n{f' {self.method} completed! ':*^30}\n")


    
    """ To check B3LYP calculations. """
    def check_B3LYP_out(self, finished_inchikeys):
        manually_process = []
        for inc in finished_inchikeys:

            #check whether the output file is empty.
            if getsize(f'{inc}.out') == 0:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')
                continue
            
            # Get basic information.
            with open(f'{inc}.out', 'rb') as f:
                content = f.read().decode()
                f.seek(-500, 2)
                endinfo = f.read().decode()
            
            with open(f'{inc}.gjf') as f:
                lines = f.readlines()

            # Deal with unfinished calculations.
            if len(findall('Normal termination', content)) != 2 and 'Error' not in endinfo:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')

            # Deal with runtime errors.
            elif 'Error' in endinfo:
                print(f'\nError end file: {inc}\nError messages:\n{chr(10).join(endinfo.split(chr(10))[-6:-4])}')
                remove(f'{inc}.out')

                # Process the problem of the empty input file.
                if 'Route card not found' in endinfo:
                    print('Processed: Remove the empty input file.\n')
                    remove(f'{inc}.gjf')

                # Process the problem of the missing line break.
                elif 'l101' in endinfo:
                    print('Processed: Add a line break to the last line.\n')
                    lines[-1] = f'{lines[-1]}\n'
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                # Process the problem of linear coordinates.
                elif 'l103' in endinfo:
                    if 'cartesian' not in lines[2]:
                        print('Processed: Change to cartesian optimization on the former structure.\n')
                        aim_content = findall('Standard orientation[\s\S]+?Z\n.+\n([\s\S]+?\n) --+', content)[-1]
                        coord = array(findall('(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\n', aim_content), float)
                        
                        for i, v in enumerate(coord.tolist()):
                            lines[7 + i] = f"{openbabel.GetSymbol(int(v[0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"
                        
                        if 'opt ' in lines[2]:
                            lines[2] = lines[2].replace('opt ', 'opt(cartesian) ')

                        elif 'maxcyc=40' in lines[2]:
                            lines[2] = lines[2].replace('maxcyc=40', 'maxcyc=40,cartesian')
                        
                        elif 'tight' in lines[2]:
                            lines[2] = lines[2].replace('tight', 'tight,cartesian')
                        
                        with open(f'{inc}.gjf', 'w', newline='\n') as f:
                            f.write(''.join(lines))
                    else:
                        print('Processed: Change conformer construction.\n')
                        IGCC._submit.Submit(self.para).build_structure_from_inc(inc, 2)

                # Process the problem of bad 3D structure.
                elif 'l202' in endinfo:
                    structure_method = 1 if '_' not in lines[4] else int(lines[4][-2])
                    print('Processed: Change conformer construction.\n')
                    if structure_method == 1:
                        IGCC._submit.Submit(self.para).build_structure_from_inc(inc, 2)
                    
                # Process the problem of SCF convergence.
                elif 'l502' in endinfo or 'l913' in endinfo:
                    if 'maxcyc' not in lines[2]:
                        print('Processed: Increase iteration steps and apply level shift.\n')
                        lines[2] = lines[2].replace('symm=veryloose', 'symm=veryloose scf(maxcyc=1600,vshift=300)')

                    elif 'nodiis' not in lines[2]:
                        print('Processed: No DIIS acceleration.\n')
                        lines[2] = lines[2].replace('maxcyc=1600', 'nodiis,maxcyc=1600')
                    
                    elif 'nodiis' in lines[2] and 'maxcyc=3000' not in lines[2]:
                        print('Processed: Increase more iteration steps.\n')
                        lines[2] = lines[2].replace('maxcyc=1600', 'maxcyc=3000')
                    
                    elif 'xqc' not in lines[2]:
                        print('Processed: Apply quadratic convergence.\n')
                        lines[2] = lines[2].replace('vshift=300', 'vshift=300,xqc')
                    
                    else:
                        print('Processed: Change conformer construction.\n')
                        IGCC._submit.Submit(self.para).build_structure_from_inc(inc, 2)
                        
                        # Update new conformer coordinates.
                        with open(f'{inc}.gjf') as f:
                            lines = f.readlines()

                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))


                # Process the problem of too high symmetry.
                elif 'l716' in endinfo:
                    aim_content = findall('Standard orientation[\s\S]+?Z\n.+\n([\s\S]+?\n) --+', content)[-1]
                    coord = array(findall('(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\n', aim_content), float)
                        
                    for i, v in enumerate(coord.tolist()):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(v[0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"
                    
                    if 'cartesian' not in lines[2]:
                        print('Processed: Change to cartesian optimization on the former structure.\n')
                        
                        if 'opt ' in lines[2]:
                            lines[2] = lines[2].replace('opt ', 'opt(cartesian) ')

                        elif 'maxcyc=40' in lines[2]:
                            lines[2] = lines[2].replace('maxcyc=40', 'maxcyc=40,cartesian')
                        
                        elif 'tight' in lines[2]:
                            lines[2] = lines[2].replace('tight', 'tight,cartesian')
                        
                    elif 'symm=veryloose' in lines[2]:
                        print('Processed: Change to no symmetry optimization on the former structure.\n')
                        lines[2] = lines[2].replace('symm=veryloose', 'nosymm')

                    else:
                        print('Processed: Change conformer construction.\n')
                        IGCC._submit.Submit(self.para).build_structure_from_inc(inc, 2)
                        
                        # Update new conformer coordinates.
                        with open(f'{inc}.gjf') as f:
                            lines = f.readlines()

                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                # Process the problem of optimization convergence.
                elif 'l9999' in endinfo:
                    structure_method = 1 if '_' not in lines[4] else int(lines[4][-2])
                    if 'opt(maxcyc' not in lines[2]:
                        print(f'Processed: Remove tight optimization together with increase grid accuracy and iteration steps.\n')

                         # Increase grid accuracy.
                        if 'int=superfine' not in lines[2]:
                            lines[2] = lines[2].replace('\n', ' int=superfine\n')

                        # Remove tight optimization.
                        if 'tight' in lines[2]:
                            lines[2] = lines[2].replace('(tight)', '').replace('tight,', '')

                        # Increase iteration steps.
                        if 'cartesian' not in lines[2]:
                            lines[2] = lines[2].replace('opt ', 'opt(maxcyc=40) ')
                        
                        else:
                            lines[2] = lines[2].replace('opt(cartesian)', 'opt(maxcyc=40,cartesian)')

                    elif 'cartesian' not in lines[2]:
                        print(f'Processed: Change to cartesian optimization.\n')
                        lines[2] = lines[2].replace('maxcyc=40', 'maxcyc=40,cartesian')

                    elif 'calcfc' not in lines[2] and 'recalc' not in lines[2]:
                        print(f'Processed: Calculate force constants.\n')
                        lines[2] = lines[2].replace('cartesian', 'cartesian,calcfc')

                    elif 'maxstep' not in lines[2] and 'recalc' not in lines[2]:
                        print(f'Processed: Change the up limit of step size.\n')
                        lines[2] = lines[2].replace('calcfc', 'calcfc,maxstep=5,notrust')

                    elif structure_method == 1:
                        print('Processed: Change conformer construction.\n')
                        IGCC._submit.Submit(self.para).build_structure_from_inc(inc, 2)
                        
                        # Update new conformer coordinates.
                        with open(f'{inc}.gjf') as f:
                            lines = f.readlines()

                    else:
                        if 'recalc' not in lines[2]:
                            print('Processed: Recalculate force constants after 5 steps and remove maximum step uplimit.\n')
                            lines[2] = lines[2].replace('calcfc,maxstep=5,notrust', 'recalc=5')
                        
                        else:
                            print('Processed: Recalculate force constants every step on the former structure.\n')
                            aim_content = findall('Standard orientation[\s\S]+?Z\n.+\n([\s\S]+?\n) --+', content)[-1]
                            coord = array(findall('(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\n', aim_content), float)
                        
                            for i, v in enumerate(coord.tolist()):
                                lines[7 + i] = f"{openbabel.GetSymbol(int(v[0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"
                            lines[2] = lines[2].replace('recalc=5', 'recalc=1')

                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))
                    
                # Process the unknown problem by users manually.
                else:
                    manually_process.append(inc)
                    print(f'Please manually choose applicable key words to elimate error for {inc}\n')
                
            # Deal with imaginary frequencies.
            elif 'imaginary frequencies' in content:
                remove(f'{inc}.out')
                aim_content = findall('Standard orientation[\s\S]+?Z\n.+\n([\s\S]+?\n) --+', content)[-1]
                coord = array(findall('(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\n', aim_content), float)    
                
                # Write coordinates of the former structure if vibration displacement are not applied.  
                for i, v in enumerate(coord.tolist()):
                    lines[7 + i] = f"{openbabel.GetSymbol(int(v[0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"

                # Get new coordinates of utilizing vibration displacement.
                res = findall(' Frequencies --(\s.+\n)[\s\S]+?Z(\n[\s\S]+?)\n\s{15}', content)[0]
                imag = [x for x in res[0].split() if float(x) < 0]
                offset = array(findall('\n\s+\d+\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)', res[1]), float)
                newcoord = (coord + 0.3 * offset).tolist()
                print(f"\nImaginary frequencies file: {inc}\nImaginary frequencies: {', '.join(imag)}\n")
                
                if 'int=superfine' not in lines[2]:
                    print('Processed: Apply vibration displacement and increase grid accuracy.\n')
                    lines[2] = lines[2].replace('\n', ' int=superfine\n')
                    
                    for i, v in enumerate(newcoord):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(coord[i, 0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"

                elif 'tight' not in lines[2]:
                    print('Processed: Change to tight optimization.\n')
                    
                    if 'opt ' in lines[2]:
                        lines[2] = lines[2].replace('opt ', 'opt(tight) ')
                    
                    elif 'maxcyc=40' in lines[2]:
                        lines[2] = lines[2].replace('opt(', 'opt(tight,')
                    
                    else:
                        lines[2] = lines[2].replace('cartesian', 'tight,cartesian')

                elif 'B3LYP' in lines[2] and '6-31G(2df,p)' not in lines[2]:
                    print('Processed: Change to the other basis set on the former structure.\n')
                    lines[2] = lines[2].replace('def2SVPP', '6-31G(2df,p)')
                    for i, v in enumerate(newcoord):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(coord[i, 0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"

                elif '6-31G(2df,p)' in lines[2] and 'M062X' not in lines[2]:
                    print('Processed: Change to high-level optimization by M062X on the former structure.\n')
                    lines[2] = lines[2].replace('B3LYP', 'M062X').replace('em=gd3bj', 'em=gd3')

                    for i, v in enumerate(newcoord):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(coord[i, 0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"

                else:
                    print('Processed: Continue to apply vibration displacement.\n')
                    for i, v in enumerate(newcoord):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(coord[i, 0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"
                
                with open(f'{inc}.gjf', 'w', newline='\n') as f:
                    f.write(''.join(lines))
                
            # Deal with the spin contamination.
            elif int(lines[6].strip().split()[-1]) > 1:
                spin = int(lines[6].strip().split()[-1])
                ini_S2 = ((spin - 1) / 2) * (((spin - 1) / 2) + 1)
                res_S2 = float(findall('annihilation\s+(\S+),', content)[-1])
                
                if abs(res_S2 - ini_S2) > ini_S2 * 0.1:
                    remove(f'{inc}.out')
                    
                    aim_content = findall('Standard orientation[\s\S]+?Z\n.+\n([\s\S]+?\n) --+', content)[-1]
                    coord = array(findall('(\d+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\n', aim_content), float)
                        
                    for i, v in enumerate(coord.tolist()):
                        lines[7 + i] = f"{openbabel.GetSymbol(int(v[0])):<3}{v[1]:15.6f}{v[2]:15.6f}{v[3]:15.6f}\n"

                    print(f'\nSpin contamination exceeds 5% of S**2: {inc}\nProcessed: Change B3LYP to ROB3LYP.\n')
                    lines[2] = lines[2].replace(' B3LYP', ' ROB3LYP').replace(' M062X', ' ROM062X')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))
            
            # Determine the checked state of the current species.
            if f'{inc}.out' in listdir('.'):
                tar_smi = self.para['species'][inc]
                mol = Molecule.from_file(f'{inc}.out')
                molblock = mol.to(fmt='mol')
                cur_mol = Chem.MolFromMolBlock(molblock, removeHs=False)
                
                if '/' in tar_smi or '\\' in tar_smi:
                    cur_smi = Chem.MolToSmiles(cur_mol, isomericSmiles=True)
                    cur_smi = cur_smi.replace('@', '')
                else:
                    cur_smi = Chem.MolToSmiles(cur_mol, isomericSmiles=False)
                
                cur_smi = Chem.MolToSmiles(Chem.MolFromSmiles(cur_smi))
                
                try:
                    tar_smi = Chem.MolToSmiles(Chem.MolFromSmiles(tar_smi))
                except:
                    cur_smi = cur_smi + tar_smi[-2:]
                
                if self.para['conformer_method'] == 2:
                    molblock = Conformer(inc, self.para, self.para['conformer_method']).rdkit_to_molblock(tar_smi)
                else:
                    molblock = Conformer(inc, self.para, self.para['conformer_method']).pybel_to_molblock(tar_smi)
                tar_mol = Chem.MolFromMolBlock(molblock, removeHs=False)
                
                similarity = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(cur_mol), Chem.RDKFingerprint(tar_mol))

                if tar_smi ==  cur_smi or similarity > 0.95:
                    self.para['geometry_state'].update({inc: [tar_smi, cur_smi, 'Y']})
                else:
                    self.para['geometry_state'].update({inc: [tar_smi, cur_smi, 'N']})


        # Show manually processed molecules.
        if manually_process:
            print(f'\n\n!!! Manually Processed inchikeys. !!!\n{chr(9).join(manually_process)}\n')

        return None


    """ To check GFNFF calculations. """
    def check_GFNFF_out(self, finished_inchikeys):
        special_molecules = []
        for inc in finished_inchikeys:

            #check whether the output file is empty.
            if getsize(f'{inc}.out') == 0:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')
                continue
            
            # Get basic information.
            with open(f'{inc}.out', 'rb') as f:
                content = f.read().decode()

            # Deal with unfinished calculations.
            if 'normally' not in content:
                if 'Initial geometry optimization failed' in content and '-cbonds' not in content:
                    remove(f'{inc}.out')
                    special_molecules.append(inc)
                    print(f'\nInitial failed: {inc}\nProcessed: Constrain bonds to rerun again.\n')
                elif 'Initial geometry optimization failed' in content and '-cbonds' in content:
                    remove(f'{inc}.out')
                    content = '\n'.join(content.split('\n')[:-1] + ['\n CREST terminated normally.\n'])
                    print(f'\nInitial failed: {inc}\nProcessed: Omit this initial problem.\n')
                    with open(f'{inc}.out', 'w', encoding='utf-8', newline='\n') as f:
                        f.write(content)
                elif 'empty ensemble' in content.split('\n')[-2] or 'Initial Geometry' in content.split('\n')[-3]:
                    remove(f'{inc}.out')
                    content = '\n'.join(content.split('\n')[:-1] + ['\n CREST terminated normally.\n'])
                    print(f'\nInitial failed: {inc}\nProcessed: Omit this initial problem.\n')
                    with open(f'{inc}.out', 'w', encoding='utf-8', newline='\n') as f:
                        f.write(content)
                else:
                    remove(f'{inc}.out')
                    print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')

        return special_molecules


    """ To check MP2 calculations. """
    def check_MP2_out(self, finished_inchikeys):
        manually_process = []
        for inc in finished_inchikeys:       

            #check whether the output file is empty.
            if getsize(f'{inc}.out') == 0:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')
                continue
            
            # Get basic information.
            with open(f'{inc}.out') as f:
                content = f.read()
            
            with open(f'{inc}.gjf') as f:
                lines = f.readlines()

            # Deal with the error of allocate memory.
            if 'could not allocate memory' in content:
                remove(f'{inc}.out')
                print(f'\nError memory input: {inc}\nProcessed: Reduce the allocate memory.\n')
                memory = findall('=(\d+)MB', lines[1])[0]
                lines = ''.join(lines).replace(f'{memory}MB', f'{int(max(int(memory) / 2, 500))}MB')

                with open(f'{inc}.gjf', 'w', newline='\n') as f:
                    f.write(''.join(lines))

            # Deal with unfinished calculations.
            elif 'Normal' not in content and 'Error' not in content:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')

            # Deal with runtime errors.
            elif 'Error' in content:
                remove(f'{inc}.out')
                print(f'\nError end file: {inc}\nError messages:\n{chr(10).join(content.split(chr(10))[-6:-4])}')
                

                # Process the problem of the empty input file.
                if 'Route card not found' in content:
                    print('Processed: Remove the empty input file.\n')
                    remove(f'{inc}.gjf')

                # Process the problem of the missing line break.
                elif 'l101' in content:
                    print('Processed: Add a line break to the last line.\n')
                    lines[-1] = f'{lines[-1]}\n'
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                # Process the problem of too high symmetry.
                elif 'Consistency failure #1 in PutSyO' in content:
                    print('Processed: Reduce symmetry to default.\n')
                    lines[2] = lines[2].replace(' symm=veryloose', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                # Process the problem of SCF convergence and the problem of maximum cycles convergence.
                elif 'l502' in content:
                    if 'maxcyc' not in lines[2]:
                        print('Processed: Increase iteration steps and apply level shift.\n')
                        lines[2] = lines[2].replace('symm=veryloose', 'symm=veryloose scf(maxcyc=1600,vshift=300)')

                    elif 'nodiis' not in lines[2]:
                        print('Processed: No DIIS acceleration.\n')
                        lines[2] = lines[2].replace('maxcyc=1600', 'nodiis,maxcyc=1600')
                    
                    elif 'nodiis' in lines[2] and 'maxcyc=3000' not in lines[2]:
                        print('Processed: Increase more iteration steps.\n')
                        lines[2] = lines[2].replace('maxcyc=1600', 'maxcyc=3000')
                    
                    elif 'xqc' not in lines[2] and 'ROMP2' not in lines[2]:
                        print('Processed: Apply quadratic convergence.\n')
                        lines[2] = lines[2].replace('vshift=300', 'vshift=300,xqc')

                    else:
                        manually_process.append(inc)
                        print(f'Please manually choose applicable key words to elimate error for {inc}\n')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                elif 'Operation on file out of range' in content:
                    print(f'\nThe wave function of HF does not match with MP2: {inc}\nProcessed: Change MP2 to ROMP2.\n')
                    lines = ''.join(lines).replace(' MP2', ' ROMP2').replace(',xqc', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(lines)

            # Deal with the spin contamination.
            elif int(lines[6].strip().split()[-1]) > 1:
                spin = int(lines[6].strip().split()[-1])
                ini_S2 = ((spin - 1) / 2) * (((spin - 1) / 2) + 1)
                res_S2 = float(findall('annihilation\s+(\S+),', content)[-1])
                
                if abs(res_S2 - ini_S2) > ini_S2 * 0.1:
                    remove(f'{inc}.out')
                    print(f'\nSpin contamination exceeds 10% of S**2: {inc}\nProcessed: Change MP2 to ROMP2.\n')
                    lines = ''.join(lines).replace(' MP2', ' ROMP2').replace(',xqc', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))
        
        # Show manually processed molecules.
        if manually_process:
            print(f'\n\n!!! Manually Processed inchikeys. !!!\n{chr(9).join(manually_process)}\n')
        
        return None




    """ To check CCSDT calculations. """
    def check_CCSDT_out(self, finished_inchikeys):
        manually_process = []
        for inc in finished_inchikeys:

            #check whether the output file is empty.
            if getsize(f'{inc}.out') == 0:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')
                continue
            
            # Get basic information.
            with open(f'{inc}.out') as f:
                content = f.read()
            
            with open(f'{inc}.gjf') as f:
                lines = f.readlines()
            

            # Deal with the error of allocate memory.
            if 'could not allocate memory' in content:
                remove(f'{inc}.out')
                print(f'\nError memory input: {inc}\nProcessed: Reduce the allocate memory.\n')
                memory = findall('=(\d+)MB', lines[1])[0]
                lines = ''.join(lines).replace(f'{memory}MB', f'{int(max(int(memory) / 2, 5000))}MB')

                with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

            # Deal with unfinished calculations.
            elif len(findall('Normal termination', content)) != 2 and 'Error' not in content:
                remove(f'{inc}.out')
                print(f'\nUnfinished work: {inc}\nProcessed: Perform calculation again.\n')

            # Deal with runtime errors.
            elif 'Error' in content:
                remove(f'{inc}.out')
                print(f'\nError end file: {inc}\nError messages:\n{chr(10).join(content.split(chr(10))[-6:-4])}')
                
                # Process the problem of the empty input file.
                if 'Route card not found' in content:
                    print('Processed: Remove the empty input file.\n')
                    remove(f'{inc}.gjf')

                # Process the problem of the missing line break.
                elif 'l101' in content:
                    print('Processed: Add a line break to the last line.\n')
                    lines[-1] = f'{lines[-1]}\n'
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

                # Process the problem of too high symmetry.
                elif 'Consistency failure #1 in PutSyO' in content:
                    print('Processed: Reduce symmetry to default.\n')
                    lines = ''.join(lines).replace(' symm=veryloose', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(f"{''.join(lines)}\n\n")

                # Process the problem of SCF convergence and the problem of maximum cycles convergence.
                elif 'l502' in content or 'l913' in content:
                    if 'maxcyc' not in ''.join(lines):
                        print('Processed: Increase iteration steps and apply level shift.\n')
                        lines = ''.join(lines).replace('symm=veryloose', 'symm=veryloose scf(maxcyc=1600,vshift=300)')

                    elif 'nodiis' not in ''.join(lines):
                        print('Processed: No DIIS acceleration.\n')
                        lines = ''.join(lines).replace('maxcyc=1600', 'nodiis,maxcyc=1600')

                    elif 'nodiis' in ''.join(lines) and 'maxcyc=3000' not in ''.join(lines):
                        print('Processed: Increase more iteration steps.\n')
                        lines = ''.join(lines).replace('maxcyc=1600', 'maxcyc=3000')
                    
                    elif 'xqc' not in ''.join(lines) and 'ROCCSD' not in ''.join(lines):
                        print('Processed: Apply quadratic convergence.\n')
                        lines = ''.join(lines).replace('vshift=300', 'vshift=300,xqc')

                    else:
                        manually_process.append(inc)
                        print(f'Please manually choose applicable key words to elimate error for {inc}\n')

                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(f"{''.join(lines)}\n\n")

                elif 'Operation on file out of range' in content:
                    print(f'\nThe wave function of HF does not match with CCSD(T): {inc}\nProcessed: Change CCSD(T) to ROCCSD(T).\n')
                    lines = ''.join(lines).replace(' CCSD', ' ROCCSD').replace(',xqc', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))

            # Deal with the spin contamination.
            elif int(lines[7].strip().split()[-1]) > 1:
                spin = int(lines[7].strip().split()[-1])
                ini_S2 = ((spin - 1) / 2) * (((spin - 1) / 2) + 1)
                res_S2 = array(findall('annihilation\s+(\S+),', content), dtype = float)

                
                if any(x > 0.1 for x in abs(res_S2 / ini_S2 - 1)):
                    remove(f'{inc}.out')
                    print(f'\nSpin contamination exceeds 10% of S**2: {inc}\nProcessed: Change CCSD(T) to ROCCSD(T).\n')
                    lines = ''.join(lines).replace(' CCSD', ' ROCCSD').replace(',xqc', '')
                    
                    with open(f'{inc}.gjf', 'w', newline='\n') as f:
                        f.write(''.join(lines))
        
        # Show manually processed molecules.
        if manually_process:
            print(f'\n\n!!! Manually Processed inchikeys. !!!\n{chr(9).join(manually_process)}\n')

        return None



    """ To check the output results and provide solutions to solve incorrect output. """
    def check_method_out(self):
        local_vars = {'self': self, 'special_molecules': None}
        
        # Remove redundant files and determine finished molecules.
        print(f'\nCheck calculations of {self.method} ...\n')
        redundant_files = glob('*.chk') + glob('job*') + glob('nohup.out') + glob('fort.7')
        for file in redundant_files: remove(file)
        finished_inchikeys = [inc for inc in self.inchikeys if f'{inc}.out' in listdir('.')]
        
        # Check calculations of each method whether completed.
        special_molecules = exec(f'special_molecules = self.check_{self.method}_out({finished_inchikeys})', local_vars)
        special_molecules = local_vars['special_molecules']
        self.check_method_completed()

        return special_molecules
