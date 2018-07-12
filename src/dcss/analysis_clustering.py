import os 
import pickle
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import pandas as pd
import itertools
import sklearn.preprocessing
from time import time
import dls_funct

import matplotlib
#workaround for x - windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
setup_dir =  dir_path[:-8]+'clustering_on_transcript_compatibility_counts'
zeisel_dir= setup_dir +'/Zeisel_pipeline/'

# # Directory with SRA files
SRA_dir=zeisel_dir+'SRA_files/'

# # Path to our version of kallisto that outputs transcript compatibility counts.  
## checking to see if kallisto is built and if not, building it.
modified_kallisto_path = setup_dir+ '/modified-kallisto-source/kallisto_pseudo_single/build/src/kallisto'
if not os.path.exists(modified_kallisto_path):
	os.system('mkdir ' + modified_kallisto_path[:-12])	
	os.chdir(modified_kallisto_path[:-12])
	os.system('cmake ..'  )
	os.system('make')
	os.chdir(dir_path)


# # Path to transcriptome
transcriptome_path_gz = zeisel_dir + '/Mus_musculus.GRCm38.cdna.all.fa.gz'
transcriptome_path= transcriptome_path_gz [:-3]
analysis_dir = zeisel_dir +'analysis/'
plot_loc = analysis_dir


if not os.path.exists(analysis_dir): 
	os.system('mkdir '+ analysis_dir)

##changing to the Zeisel_pipeline directory because all those files have
## relative paths within them.
os.chdir(zeisel_dir)

if not os.path.exists(SRA_dir):
	os.system('python ' +zeisel_dir+ 'get_files.py')

num_processes=40
SEED=429496729
np.random.seed(SEED)
check=np.random.normal(0, 1)
np.random.seed(SEED)

#get transcriptome reference
if not os.path.exists(transcriptome_path): 
	os.system('wget -P '+zeisel_dir+ ' ftp://ftp.ensembl.org/pub/release-79/fasta/mus_musculus/cdna/Mus_musculus.GRCm38.cdna.all.fa.gz')
	#os.system('mv Mus_musculus.GRCm38.cdna.all.fa.gz ' + zeisel_dir)
	os.system('gunzip '+ transcriptome_path_gz)


#load data and perform some other unpacking
os.chdir(zeisel_dir)
os.system('python ' +zeisel_dir+ 'Zeisel_wrapper.py -i '+SRA_dir+' -k '+modified_kallisto_path+ ' -n ' + str(num_processes)+' -t '+ transcriptome_path)

sampling_suffix=['100']
index=0

TCC_base_flname=zeisel_dir+ 'Zeisel_TCC_subsample'
TCC_dist_base_flname=zeisel_dir+ 'Zeisel_TCC_distribution_subsample'
TCC_distance_base_flname=zeisel_dir+ 'Zeisel_TCC_pairwise_JS_distance_subsample'

TCC_base_flname_ss=TCC_base_flname+sampling_suffix[index]
TCC_dist_base_flname_ss= TCC_dist_base_flname+sampling_suffix[index]
TCC_distance_base_flname_ss= TCC_distance_base_flname+sampling_suffix[index]


TCC_flname=TCC_base_flname_ss+".dat"
TCC_dist_flname=TCC_dist_base_flname_ss+".dat"
TCC_distance_flname=TCC_distance_base_flname_ss+".dat"


#Loading unnormalized file and distance file.
with open(TCC_flname,'rb') as infile:
        TCC = pickle.load(infile)

with open(TCC_distance_flname ,'rb') as infile:
    D = pickle.load(infile)
    assert np.all(np.isclose(D,D.T))
    assert np.all(np.isclose(np.diag(D),np.zeros(np.diag(D).shape)))


##plotting eigenvalues of AA^T
n_eig=30
dls_funct.plot_eigenvalues(TCC, n_eig, plot_loc)


##the final analysis choice of k, epsikon
epsilon=0.1
k=5

##time report
dls_funct.zeisel_time_report(TCC, k, epsilon, analysis_dir)


#making a bunch of plots and calculating the umi depth
dls_funct.zeisel_powerlaw_meanvar_index_comparision_gene_set(TCC, k, epsilon, analysis_dir, plot_loc, num_processes, TCC_flname[:-4], TCC_dist_flname[:-4], TCC_distance_flname[:-4])

assert check==np.random.normal(0, 1)
np.random.seed(SEED)


#making the error plot for changing k, fixed epsilon
SEED2=3434534
np.random.seed(SEED2)

error_all=[]
epsilon=0.1
kset=[3,5,7,9,11,13,15]
for k in kset:
	error=dls_funct.make_zeisel_error(D, TCC, k, epsilon, analysis_dir, plot_loc, num_processes, TCC_flname[:-4], TCC_dist_flname[:-4], TCC_distance_flname[:-4])
	error_all.append(error)

with open(analysis_dir + 'error_all_' +str(epsilon).replace('.', '')+ '.dat', 'wb') as outfile:
	pickle.dump(error_all,outfile, pickle.HIGHEST_PROTOCOL)

err= pd.DataFrame(np.empty((len(kset), 8)), index= ['k=3','k=5', 'k=7', 'k=9', 'k=11', 'k=13', 'k=15'], columns= ['DCSS 2 cluster', 'DCSS 9 cluster', 'Var. 2 cluster', 'Var. 9 cluster', 'Count 2 cluster', 'Count 9 cluster', 'I.D. 2 cluster', 'I.D. 9 cluster'])
for i, item in enumerate(error_all):
	err.iloc[i, :] = np.average(item, axis=0)

with open(analysis_dir + 'err_' +str(epsilon).replace('.', '')+ '.dat', 'wb') as outfile:
	pickle.dump(err,outfile, pickle.HIGHEST_PROTOCOL)

with open(analysis_dir + 'err_' +str(epsilon).replace('.', '')+ '.dat', 'rb') as infile:
	err=pickle.load(infile)

#kset
dls_funct.plot_error_rate(err, kset, 'epsilon', '', epsilon, num_clust='2', data_type='TCC', plot_loc=plot_loc)
dls_funct.plot_error_rate(err, kset, 'epsilon', '', epsilon, num_clust='9', data_type='TCC', plot_loc=plot_loc)

#making the error plot for changing epsilon, fixed k
SEED3=2343243
np.random.seed(SEED3)

error_all_epsilon=[]
k=5
epset=[0.05,0.1,0.15,0.2,0.25]
for epsilon in epset:
	error=dls_funct.make_zeisel_error(D, TCC, k, epsilon, analysis_dir, plot_loc, num_processes, TCC_flname[:-4], TCC_dist_flname[:-4], TCC_distance_flname[:-4])
	error_all_epsilon.append(error)



###epsilon error
with open(analysis_dir + 'error_all_epsilon_'+ str(k)+'.dat', 'wb') as outfile:
	pickle.dump(error_all_epsilon,outfile, pickle.HIGHEST_PROTOCOL)


err_epsilon= pd.DataFrame(np.empty((len(epset), 8)), index= ['$\epsilon$=0.05','$\epsilon$=0.1', '$\epsilon$=0.15', '$\epsilon$=0.2', '$\epsilon$=0.25'], columns= ['DCSS 2 cluster', 'DCSS 9 cluster', 'Var. 2 cluster', 'Var. 9 cluster', 'Count 2 cluster', 'Count 9 cluster', 'I.D. 2 cluster', 'I.D. 9 cluster'])
for i, item in enumerate(error_all_epsilon):
	err_epsilon.iloc[i, :] = np.average(item, axis=0)

with open(analysis_dir + 'err_epsilon_'+ str(k)+'.dat', 'wb') as outfile:
	pickle.dump(err_epsilon,outfile, pickle.HIGHEST_PROTOCOL)

with open(analysis_dir + 'err_epsilon_'+ str(k)+'.dat', 'rb') as infile:
	err_epsilon=pickle.load(infile)

##epsilon set
dls_funct.plot_error_rate(err_epsilon , epset, 'k', k, '', num_clust='2', data_type='TCC', plot_loc=plot_loc)
dls_funct.plot_error_rate(err_epsilon , epset, 'k', k, '', num_clust='9', data_type='TCC', plot_loc=plot_loc)




