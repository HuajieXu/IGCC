## IGCC-1.0:
IGCC: Isodesmic Group Contribution Compensation 

Date: August 30, 2025.

Author: Huajie Xu, Qinghao Sun, Jinhun Zhong, Yang Wang.

Email: yang.wang@swjtu.edu.cn.

Copyright: Southwest Jiao Tong University.

Cite: XXXXXX


## Characteristic:
IGCC-1.0 is developed to obtain accurate thermodynamic parameters of mechanisms within chemical accuracy for C/H/O hydrocarbons.
It can automatically build input files and deal with occurred errors, such as spin contaminations, imaginary frequencies and so on.
After all errors are eliminated, the accurate thermodynamic data will be generated in the format of 14 parameters for Chemkin use.

The enthalpies of formation of larger molecules are derived by IGCC method with mBAC corrections, while those of small molecules
are evaluated by CCSD(T)/CBS method. Combined the corrections of anharmonic and conformational sampling, the calculated accuracy
of all thermodynamic data generally meets the requirements of chemical accuracy.


## Preparation:
1. Platform: Windows or Linux. Note that the Linux system is required to get QM calculations of Gaussian.
2. Environment: Conda with Python (>=3.10). Conda can be obtained by Anaconda or Miniconda (https://docs.conda.io/en/latest/).
3. Modules: requirements.txt. Enter "conda install --yes -c conda-forge --file requirements.txt".
4. Extensions: Gaussian, xtb, CREST packages. Gaussian package is commercialized to get optimized structures and single point energies.
   For conformational sampling, open-source CREST with xtb are taken from https://github.com/grimme-lab/.


## Setup
1. Unpack "IGCC-1.0.tar.gz" or "IGCC-1.0.zip" file and enter the main directory of "IGCC-1.0".
2. Enter "conda install --yes -c conda-forge --file requirements.txt" to install required modules.
3. Add the main directory to system environment. For Linux system, the IGCC.py file needs key words of ':set ff=unix' in vim editor 
   to convert dos to unix.


## Usage:
1. Enter "python tdoc.py smiles.txt" in another directory to automatically generate input scripts and files. 
2. Copy submitted files in the newly generated "subfiles" directory to Linux platform for Gaussian, Molpro, and CREST calculations.
3. Input "bash job-sub" to run jobs.
4. Copy all output files to the directory of "rawfiles".
5. Repeat previous steps until the all tasks are completed.
6. If any IGCC or mBAC parameter is missing, one can utilize the corresponding scripts in "tools" directory der to train parameters.


## Examples:
1. smiles.txt: The basic file for input species with their SMILES. It is based on manually input by users.
2. parameters.ini: The control parameters should be set according to system configuration and user demand.
3. mBAC_parameters.txt: The mBAC paramters for bond additivity correction which are continuously updated. 
4. IGCC_parameters.txt: The IGCC paramters for bond difference correction which are continuously updated. 
5. datfiles: The data directory to store generated thermodynamic parameters files by IGCC for CHEMKIN use.
6. rawfiles: The output directory to deposit input and output files of QM calculations for Linux system.
7. csvfiles: The result directory to contain some summary analysis and overall data files for species.
8. subfiles: The submitted directory for QM calculations on Linux system. It will be removed if all data are completed.
