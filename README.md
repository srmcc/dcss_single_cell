# Deterministic Column Subset Selection for single-cell RNA-Seq

This repository has been developed to reproduce the results in McCurdy, Ntranos, and Pachter (2017, BIORXIV/2017/159079).  We implement the deterministic column subset selection (DCSS) algorithm (Papailiopoulos, Kyrillidis, and Boutsidis, 2014).  We integrate the DCSS method and three comparision thresholding methods into two workflows for clustering (clustering_on_transcript_compatibility_counts and wishbone, forked into this repository) and apply it to two single-cell RNA-Seq datasets (Paul et al., 2015; Zeisel et al. 2015).  This repository will download the datasets for you.

# Installing the dependencies
To run this code you will need to install all of the dependencies for clustering_on_transcript_compatibility_counts and wishbone.  Please see the respective README.md files.  We used the anaconda package manager which can be obtained at https://www.continuum.io/downloads, and we include the two .yml files that we used for the analyses.  You can simply create an "conda env" by 
```bash
conda env create -f env.yml 
#
# To activate this environment, use:
# $ source activate env
#
# To deactivate this environment, use:
# $ source deactivate
#
```
# Getting Started
In order to run the code, first clone this repository and the submodules.

```bash
git clone --recursive https://github.com/srmcc/dcss_single_cell.git
```
Then activate your conda environment:

```bash
source activate wishbone
```
Change directory to the cloned repository:
```bash
cd to /path/to/dcss_single_cell/src/dcss/
```
And run the analysis files.
```bash
python analysis_wishbone.py
```
The user experience and documentation for this package is still under construction.

