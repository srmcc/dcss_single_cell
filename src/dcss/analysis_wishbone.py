# Plotting and miscellaneous imports
import os
import scipy
import scipy.sparse.linalg
import pandas as pd
import numpy as np
import dls_funct
import sklearn.preprocessing
import pickle
import re
import imp

import matplotlib
#workaround for x - windows

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.pyplot as plt

num_processes=40
SEED=429496729
np.random.seed(SEED)
check=np.random.normal(0, 1)
np.random.seed(SEED)


dir_path = os.path.dirname(os.path.realpath(__file__))
setup_dir =  dir_path[:-8]+'wishbone'
filename, pathname, description = imp.find_module('wishbone', [setup_dir+'/src/'])
wishbone= imp.load_module('wishbone', filename, pathname, description)


analysis_dir =setup_dir +'/analysis/'
plot_loc = analysis_dir

#download, load data and perform some other unpacking
scdata_raw = dls_funct.wishbone_my_setup(setup_dir)

##plotting eigenvalues of AA^T
n_eig =30
dls_funct.plot_eigenvalues(scipy.sparse.csr_matrix(scdata_raw.data), n_eig, plot_loc)

##the final analysis choice of k, epsikon
k=14
epsilon=0.05

#making a bunch of plots and calculating the umi depth
dls_funct.wishbone_powerlaw_meanvar_index_comparision_gene_set(scdata_raw, k, epsilon, analysis_dir, plot_loc)

#performing the wishbone pre-analysis on the full data (normalization, pca, diffusion components)
components = dls_funct.wishbone_pipeline(scdata_raw, 'all', analysis_dir)	
#scdata is now normalized in the loaded file below
scdata= wishbone.wb.SCData.load(analysis_dir + 'mouse_marrow_scdata_all.p')

assert check==np.random.normal(0, 1)
np.random.seed(SEED)

#setting the number of random repetitions and performing the wishbone analysis on the full data
nrep=10
scdata_wb_list= dls_funct.make_wishbone_assignments_all_random_multi(scdata, components, analysis_dir, plot_loc, nrep, num_processes)


#making the error plot for changing k, fixed epsilon
SEED2=3434534
np.random.seed(SEED2)

error_all=[]
epsilon=0.05
kset=[4, 6, 8, 10, 12, 14, 16]
for k in kset:
	error=dls_funct.make_wishbone_error_multi(scdata_raw, k, epsilon, scdata_wb_list, analysis_dir, plot_loc, nrep=nrep, num_processes=num_processes)
	error_all.append(error)

with open(analysis_dir + 'error_all_' +str(epsilon).replace('.', '')+ '.dat', 'wb') as outfile:
	pickle.dump(error_all,outfile, pickle.HIGHEST_PROTOCOL)

err= pd.DataFrame(np.empty((len(kset), 4)), index= ['k=4','k=6', 'k=8', 'k=10', 'k=12', 'k=14', 'k=16'], columns= ['DCSS', 'Var.', 'Count', 'I.D.'])
for i, item in enumerate(error_all):
	err.iloc[i, :] = np.average(item, axis=0)

with open(analysis_dir + 'err_' +str(epsilon).replace('.', '')+ '.dat', 'wb') as outfile:
	pickle.dump(err,outfile, pickle.HIGHEST_PROTOCOL)

with open(analysis_dir + 'err_' +str(epsilon).replace('.', '')+ '.dat', 'rb') as infile:
	err=pickle.load(infile)

#kset
dls_funct.plot_error_rate(err, kset, 'epsilon', '', epsilon, num_clust='', data_type='wishbone', plot_loc=plot_loc)

#making the error plot for changing epsilon, fixed k
SEED3=2343243
np.random.seed(SEED3)

error_all_epsilon=[]
k=14
epset=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
for epsilon in epset:
	error=dls_funct.make_wishbone_error_multi(scdata_raw, k, epsilon, scdata_wb_list, analysis_dir, plot_loc, nrep=nrep, num_processes=num_processes)
	error_all_epsilon.append(error)

###epsilon error
with open(analysis_dir + 'error_all_epsilon_'+ str(k)+'.dat', 'wb') as outfile:
	pickle.dump(error_all_epsilon,outfile, pickle.HIGHEST_PROTOCOL)

err_epsilon= pd.DataFrame(np.empty((len(epset), 4)), index= ['$\epsilon$=0.01', '$\epsilon$=0.05','$\epsilon$=0.1', '$\epsilon$=0.15', '$\epsilon$=0.2', '$\epsilon$=0.25'], columns= ['DCSS', 'Var.', 'Count', 'I.D.'])
for i, item in enumerate(error_all_epsilon):
	err_epsilon.iloc[i, :] = np.average(item, axis=0)

with open(analysis_dir + 'err_epsilon_'+ str(k)+'.dat', 'wb') as outfile:
	pickle.dump(err_epsilon,outfile, pickle.HIGHEST_PROTOCOL)

with open(analysis_dir + 'err_epsilon_'+ str(k)+'.dat', 'rb') as infile:
	err_epsilon=pickle.load(infile)

##epsilon set
dls_funct.plot_error_rate(err_epsilon , epset, 'k', k, '', num_clust='', data_type='wishbone', plot_loc=plot_loc)


