import os
import shutil
import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn import cluster
import sklearn.preprocessing
import networkx as nx
import itertools
import scipy.sparse
import scipy.sparse.linalg
import pickle
import re
from time import time
import sys
import imp
import multiprocessing
import traceback
from resource import getrusage as resource_usage, RUSAGE_SELF

import matplotlib
#workaround for x - windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

python_version= sys.version[0]

if python_version=='3':
    filename, pathname, description=imp.find_module('wishbone', [dir_path[:-9]+'/wishbone/src/'])
    wishbone= imp.load_module('wishbone', filename, pathname, description)

def plot_eigenvalues(data, n_eig, plot_loc):
    """
    takes in data (A) and plots eigenvalues of A A^T
    """
    umi_svd = scipy.sparse.linalg.svds(data , n_eig)
    fig, ax = plt.subplots()
    # im=ax.scatter(range(0, n_eig), sorted(umi_svd [1]**2, reverse=True), color='grey', s=36,  lw=0.5) #, label=r'Eigenvalues of A')
    im=ax.plot(range(0, n_eig), sorted(umi_svd [1]**2, reverse=True), color='grey', marker = 'o', ls ='solid' ,  lw=1) #, label=r'Eigenvalues of A')
    # ax.legend(loc='upper right')
    ax.set_xlabel('Eigenvalue rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_yscale('log')
    ax.set_xlim((-1, 30))
    fig.tight_layout()
    plt.savefig(plot_loc + 'plot_eigenvalues.pdf')



def det_leverage(V, k, epsilon):
    """
    for the data matrix A=U \Sigma V^T
    A.shape =(n=number of samples, d=number of features)
    V.shape=(d, n)
    k is the rank of the PCA leverage score
    epsilon is the error parameter.
    the function returns
    theta: the number of kept columns
    index_keep: the index of the selected columns
    tau_sorted:  the sorted leverage scores of all the columns
    index_drop:  the index of the dropped columns
    """
    V = pd.DataFrame(V)
    Vk= V.iloc[:, 0:k]
    print(Vk.shape)
    tau= np.sum(Vk**2, axis=1)
    tau_sorted= tau.sort_values(ascending=False, inplace=False)
    lev_sum=0
    for i in range(V.shape[0]):
        lev_sum =lev_sum+ tau_sorted.iloc[i]
        if lev_sum > k - epsilon:
            theta=i+1
            if theta >= k:
                break
    index_keep= tau_sorted.index[0:theta]
    index_keep = index_keep.values
    index_drop = tau_sorted.index[theta:]
    index_drop= index_drop.values
    return(theta, index_keep, tau_sorted, index_drop)


def plot_tau(tau_sorted, theta, k, plot_loc):
    """ 
    """
    fig, ax = plt.subplots()
    ax.scatter(range(len(tau_sorted)),tau_sorted,c='red',s=36,edgecolors='gray',
                    lw = 0.5, label='rank ' + str(k) + ' leverage scores')
    ax.axvline(x=theta, label='theta')
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    ax.set_xlim((-10, len(tau_sorted)))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel('rank of leverage score')
    ax.set_ylabel('leverage score')
    plt.title('Leverage scores')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_leverage_score_' +str(k) +'.pdf')

def make_leverage_norm_jsdistance(TCC, V, k, epsilon, TCC_dls_flname, TCC_dls_dist_flname, TCC_dls_distance_flname, num_processes, plot_loc):
    theta_TCC, index_keep_TCC, tau_sorted, index_drop = det_leverage(V, k, epsilon)
        with open(TCC_dls_flname, 'wb') as outfile:
    TCC_dls= TCC[:, index_keep_TCC]
    if not os.path.exists(TCC_dls_flname):
                pickle.dump(scipy.sparse.csr_matrix( TCC_dls.todense()), outfile, pickle.HIGHEST_PROTOCOL)   
    if not os.path.exists(TCC_dls_dist_flname):
        TCC_dls_dist= sklearn.preprocessing.normalize(TCC_dls, norm='l1', axis=1)
        with open(TCC_dls_dist_flname, 'wb') as outfile:
                    pickle.dump(scipy.sparse.csr_matrix(TCC_dls_dist.todense()), outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(TCC_dls_dist_flname,'rb') as infile:
            TCC_dls_dist= pickle.load(infile)
    if not os.path.exists(TCC_dls_distance_flname):
        t=time()
        os.system('python get_pairwise_distances.py '+TCC_dls_dist_flname+' '+TCC_dls_distance_flname+' '+str(num_processes))
        print(time() - t) / 60, 'min'
    #filepath='./'
    filepath=''
    with open(filepath+TCC_dls_distance_flname ,'rb') as infile:
        D_dls = pickle.load(infile)
    assert np.all(np.isclose(np.diag(D_dls),np.zeros(np.diag(D_dls).shape)))
    assert np.all(np.isclose(D_dls ,D_dls.T))
    return(TCC_dls, TCC_dls_dist, D_dls, theta_TCC, index_keep_TCC, tau_sorted)

# def make_var_norm_jsdistance(TCC, columns_dls, TCC_vf_dist_flname, TCC_vf_distance_flname, num_processes):
#     scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
#     min_std_keep= sorted(scaler.std_, reverse=True)[columns_dls]
#     vf_keep_bool=(scaler.std_ >min_std_keep)
#     TCC_vf= TCC[:, (scaler.std_ >min_std_keep)]
#     umi_depth= TCC_vf.sum()/TCC.sum()
#     if not os.path.exists(TCC_vf_dist_flname):
#         TCC_vf_dist= sklearn.preprocessing.normalize(TCC_vf, norm='l1', axis=1)
#         with open(TCC_vf_dist_flname, 'wb') as outfile:
#             pickle.dump(scipy.sparse.csr_matrix(TCC_vf_dist.todense()), outfile, pickle.HIGHEST_PROTOCOL)
#     else:
#         with open(TCC_vf_dist_flname,'rb') as infile:
#             TCC_vf_dist= pickle.load(infile)
#     if not os.path.exists(TCC_vf_distance_flname):
#         t=time()
#         os.system('python get_pairwise_distances.py '+TCC_vf_dist_flname+' '+TCC_vf_distance_flname+' '+str(num_processes))
#         print(time() - t) / 60, 'min'
#     filepath='./'
#     with open(filepath+TCC_vf_distance_flname ,'rb') as infile:
#         D_vf = pickle.load(infile)
#     assert np.all(np.isclose(D_vf ,D_vf.T))
#     assert np.all(np.isclose(np.diag(D_vf),np.zeros(np.diag(D_vf ).shape)))
#     return(TCC_vf, TCC_vf_dist, D_vf, vf_keep_bool, umi_depth)

# def make_counts_norm_jsdistance(TCC, columns_dls, TCC_lc_dist_flname, TCC_lc_distance_flname, num_processes):
#     ## filter top features based on counts.
#     min_counts_keep= sorted(np.array(TCC.sum(axis=0))[0,:], reverse=True)[columns_dls]
#     counts_keep_bool=(np.array(TCC.sum(axis=0))[0,:] >min_counts_keep)
#     TCC_lc= TCC[:, counts_keep_bool]
#     umi_depth=TCC_lc.sum()/TCC.sum()
#     if not os.path.exists(TCC_lc_dist_flname):
#         TCC_lc_dist= sklearn.preprocessing.normalize(TCC_lc, norm='l1', axis=1)
#         with open(TCC_lc_dist_flname, 'wb') as outfile:
#             pickle.dump(scipy.sparse.csr_matrix(TCC_lc_dist.todense()), outfile, pickle.HIGHEST_PROTOCOL)
#     else:
#         with open(TCC_lc_dist_flname,'rb') as infile:
#             TCC_lc_dist= pickle.load(infile)
#     if not os.path.exists(TCC_lc_distance_flname):
#         t=time()
#         os.system('python get_pairwise_distances.py '+TCC_lc_dist_flname+' '+TCC_lc_distance_flname+' '+str(num_processes))
#         print(time() - t) / 60, 'min'
#     filepath='./'
#     with open(filepath+TCC_lc_distance_flname ,'rb') as infile:
#         D_lc = pickle.load(infile)
#     assert np.all(np.isclose(D_lc,D_lc.T))
#     assert np.all(np.isclose(np.diag(D_lc),np.zeros(np.diag(D_lc).shape)))
#     return(TCC_lc, TCC_lc_dist, D_lc, counts_keep_bool, umi_depth)

def make_norm_jsdistance(TCC, columns_dls, TCC_lc_dist_flname, TCC_lc_distance_flname, num_processes, filter_type, distance=True):
    ## filter top features based on counts.
    if python_version == '2':
        nonzero_bool=(np.array(TCC.sum(axis=0))[0,:] >0)
        TCC=TCC[:, nonzero_bool]
    elif python_version == '3':
        nonzero_bool=(np.array(TCC.sum(axis=0))>0)
        TCC=TCC[:, nonzero_bool]
    print('shape non-zero TCC', TCC.shape)
    if python_version == '2':
        if filter_type=='counts':
            min_keep= sorted(np.array(TCC.sum(axis=0))[0,:], reverse=True)[columns_dls]
            keep_bool=(np.array(TCC.sum(axis=0))[0,:] >min_keep)
        if filter_type == 'std':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            min_keep= sorted(scaler.std_, reverse=True)[columns_dls]
            keep_bool=(scaler.std_ >min_keep)
        if filter_type=='coeff_var':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            mean= np.array(TCC.sum(axis=0))[0,:]/TCC.shape[0]
            coeff_var=scaler.std_/mean
            min_keep= sorted(coeff_var, reverse=True)[columns_dls]
            keep_bool=(coeff_var > min_keep)
        if filter_type == 'index_disp':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            mean= np.array(TCC.sum(axis=0))[0,:]/TCC.shape[0]
            index_disp=scaler.std_**2/mean
            min_keep= sorted(index_disp, reverse=True)[columns_dls]
            keep_bool=(index_disp > min_keep)
    elif python_version == '3':
        if filter_type=='counts':
            min_keep= sorted(np.array(TCC.sum(axis=0)), reverse=True)[columns_dls]
            keep_bool=(np.array(TCC.sum(axis=0)) >min_keep)
        if filter_type == 'std':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            min_keep= sorted(scaler.std_, reverse=True)[columns_dls]
            keep_bool=(scaler.std_ >min_keep)
        if filter_type=='coeff_var':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            mean= np.array(TCC.sum(axis=0))/TCC.shape[0]
            coeff_var=scaler.std_/mean
            min_keep= sorted(coeff_var, reverse=True)[columns_dls]
            keep_bool=(coeff_var > min_keep)
        if filter_type == 'index_disp':
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
            mean= np.array(TCC.sum(axis=0))/TCC.shape[0]
            index_disp=scaler.std_**2/mean
            min_keep= sorted(index_disp, reverse=True)[columns_dls]
            keep_bool=(index_disp > min_keep)
    print('type, min_keep', filter_type, min_keep)
    TCC_lc= TCC[:, keep_bool]
    umi_depth=TCC_lc.sum()/TCC.sum()
    if distance==True:
        if not os.path.exists(TCC_lc_dist_flname):
            TCC_lc_dist= sklearn.preprocessing.normalize(TCC_lc, norm='l1', axis=1)
            with open(TCC_lc_dist_flname, 'wb') as outfile:
                pickle.dump(scipy.sparse.csr_matrix(TCC_lc_dist.todense()), outfile, pickle.HIGHEST_PROTOCOL)
        else:
            with open(TCC_lc_dist_flname,'rb') as infile:
                TCC_lc_dist= pickle.load(infile)
        if not os.path.exists(TCC_lc_distance_flname):
            t=time()
            os.system('python get_pairwise_distances.py '+TCC_lc_dist_flname+' '+TCC_lc_distance_flname+' '+str(num_processes))
            print(time() - t) / 60, 'min'
        #filepath='./'
        filepath=''
        with open(filepath+TCC_lc_distance_flname ,'rb') as infile:
            D_lc = pickle.load(infile)
        assert np.all(np.isclose(D_lc,D_lc.T))
        assert np.all(np.isclose(np.diag(D_lc),np.zeros(np.diag(D_lc).shape)))
        return(TCC_lc, TCC_lc_dist, D_lc, keep_bool, umi_depth, min_keep)
    else:
        return(TCC_lc, keep_bool, umi_depth, min_keep)



def powerlaw(x, amp, index):
    return amp * (x**index)
    
    
def powerlaw_fit(xdata, ydata, k, plot_loc):
    """
    http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
    Fitting the data -- Least Squares Method
    #########
    Power-law fitting is best done by first converting
    to a linear equation and then fitting to a straight line.
    
     y = a * x^b
     log(y) = log(a) + b*log(x)
    modified by SRM
    """
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    pinit = [1.0, -1.0]
    out = scipy.optimize.leastsq(errfunc, pinit,
                           args=(logx, logy), full_output=1)
    pfinal = out[0]
    covar = out[1]
    print( "pfinal", pfinal)
    print( "covar", covar)
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    ##########
    # Plotting data
    ##########
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, np.log10(powerlaw(xdata, amp, index)))     # Fit
    plt.plot(xdata, logy)  # Data
    plt.text(400, 0, 'b = %5.2f +/- %5.2f' % (amp, ampErr))
    plt.text(400, -1, 'a = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title('Best Fit Power Law for rank '+str(k))
    plt.xlabel('X')
    plt.ylabel('log_10(Y) ')
    #plt.xlim(1, 11)
    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.plot(xdata, ydata)  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    #plt.xlim(1.0, 11)   
    plt.savefig(plot_loc + 'plot_powerlaw_2p_' +str(k) +'.pdf')
    plt.close()
    fig, ax = plt.subplots()
    ax.loglog(xdata, powerlaw(xdata, amp, index), color='gray', ls='dashed', label='Fit')
    ax.plot(xdata, ydata, color='black', ls='solid', label='Data')  # Data
    ax.annotate('b = %5.2f +/- %5.2f' % (amp, ampErr),  xy=(.30, .80), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom')
    ax.annotate('a = %5.2f +/- %5.2f' % (index, indexErr),  xy=(.30, .85), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='bottom')
    ax.set_xlabel('Sorted column index')
    ax.set_ylabel('Subspace leverage score')
    ax.set_xlim([0, len(xdata)])
    ax.legend(loc='upper right',bbox_to_anchor=(0.95, 1))
    fig.tight_layout()
    #plt.xlim(1.0, 11)   
    fig.savefig(plot_loc+ 'plot_powerlaw_' +str(k) +'.pdf')
    return(amp, index)

def predict_n_columns(index, k, epsilon):
    eta= -index-1
    return np.max( ((2*k/epsilon)**(1/(1+eta)),(2*k/epsilon/eta)**(1/eta), k))


def plot_svd(sigma_full, sigma_dls, k, plot_loc):
    """ Plot the variance explained by different principal components
    :param n_components: Number of components to show the variance
    :param ylim: y-axis limits
    :param fig: matplotlib Figure object
    :param ax: matplotlib Axis object
    :return: fig, ax
    """
    fig, ax = plt.subplots()
    ax.scatter(range(len(sigma_full)),sigma_full,c='red',s=36,edgecolors='gray',
                    lw = 0.5, label='TCC singular values')
    ax.scatter(range(len(sigma_dls)),sigma_dls,c='blue',s=36,edgecolors='gray',
                    lw = 0.5, label='TCC_dls singular values')
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('Components')
    ax.set_ylabel('Singular Values')
    plt.title('TCC Distribution Singular Values')
    fig.tight_layout()
    plt.savefig(plot_loc+ 'plot_pca_variance_explained_' +str(k) +'.pdf')


def tSNE_pairwise(D):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    tsne = manifold.TSNE(n_components=2, random_state=0, metric='precomputed', n_iter=2000, verbose=1);
    X_tsne = tsne.fit_transform(D);
    return X_tsne

# Plot function with Zeisel's colors corresponding to labels 
def tru_plot9(X,labels,t,plot_suffix,clust_names,clust_color, plot_loc):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(15,10))
    for i in unique_labels:
        ind = np.squeeze(labels == i)
        plt.scatter(X[ind,0],X[ind,1],c=clust_color[i],s=36,edgecolors='gray',
                    lw = 0.5, label=clust_names[i])        
    plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1))
    plt.legend(loc='upper right',bbox_to_anchor=(1.19, 1.01))
    plt.title(t)
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.axis('off')
    plt.savefig(plot_loc+ 't-SNE_plot_tru_plot9_'+ plot_suffix +'.pdf', bbox_inches='tight')

    # Plot function with Zeisel's colors corresponding to labels 
def tru_plot2_new(X,labels1,labels2,t,plot_suffix,clust_names,clust_color, plot_loc):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    labels=copy.deepcopy(labels1)
    for i in range(len(labels)):
        if labels[i]!= labels2[i]:
            if labels[i] == 0:
                labels[i] = 2
            else:
                labels[i] = 3
    unique_labels = np.unique(labels)
    plt.figure(figsize=(15,10))
    for i in unique_labels:
        ind = np.squeeze(labels == i)
        plt.scatter(X[ind,0],X[ind,1],c=clust_color[i],s=36,edgecolors='gray',
                    lw = 0.5, label=clust_names[i])        
    plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1))
    plt.legend(loc='upper right',bbox_to_anchor=(1.19, 1.01))
    plt.title(t)
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.axis('off')
    plt.savefig(plot_loc+ 't-SNE_plot_tru_plot2_'+ plot_suffix +'.pdf', bbox_inches='tight')



# obtain labels via spectral clustering
def spectral(k,D, rs):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    if D[1,1] < 1: D = 1-D # Convert distance to similarity matrix
    spectral = cluster.SpectralClustering(n_clusters=k,affinity='precomputed', random_state=rs)
    spectral.fit(D)
    labels = spectral.labels_
    return labels

# gets max weight matching of a biparetite graph with row_label x column_label
# (weights are given by weight_matrix)
def get_max_wt_matching(row_label,column_label, weight_matrix):
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    # Create a bipartite graph where each group has |unique labels| nodes 
    G = nx.complete_bipartite_graph(len(row_label), len(column_label))
    # Weight each edge by the weight in weight matrix.. 
    for u,v in G.edges(): G[u][v]["weight"]=weight_matrix[u,v-len(row_label)]
    # Perform weight matching using Kuhn Munkres
    H=nx.max_weight_matching(G)
    max_wt=0
    for u,v in H.items(): max_wt+=G[u][v]["weight"]/float(2)
    return max_wt

def compute_clustering_accuracy(label1, label2, type='ARI'):
    """
    type== 'ARI' is adjusted rand index.
    type!='ARI' is  clustering error.  From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    if type == 'ARI':
        return(sklearn.metrics.adjusted_rand_score(label1, label2))
    else:
        uniq1,uniq2 = np.unique(label1),np.unique(label2)
        # Create two dictionaries. Each will store the indices of each label
        entries1,entries2 = {},{}
        for label in uniq1: entries1[label] = set(np.flatnonzero((label1==label)))
        for label in uniq2: entries2[label] = set(np.flatnonzero((label2==label)))
        # Create an intersection matrix which counts the number of entries that overlap for each label combination        
        W = np.zeros((len(uniq1),len(uniq2)))
        for i,j in itertools.product(range(len(uniq1)),range(len(uniq2))):
            W[i,j]=len(entries1[uniq1[i]].intersection(entries2[uniq2[j]]))
        # find the max weight matching
        match_val = get_max_wt_matching(uniq1,uniq2,W)
        # return the error rate
        return (1-match_val/float(len(label1)))*100

def load_labels():
    """
    From clustering_on_transcript_compatibility_counts, see github for MIT license
    """
    # Zeisel's 9 main clusters
    labels9 = np.loadtxt('./Zeisels_data/Zeisels_labels9.txt',dtype=str).astype(int)-1
    # Zeisel's neurons (labeled as 0) and non-neurons (labeled as 1) obtained from labels9
    labels2 = np.copy(labels9)
    for i in [0,1,2]: labels2[labels2 == i] = 0       
    labels2[labels2 != 0] = 1
    # Zeisel's 47 total clusters
    labels47 = np.loadtxt('./Zeisels_data/Zeisels_labels47.txt',dtype=str)
    return(labels2, labels9, labels47)


def stain_plot(X,labels,stain,title, nc=2, plot_loc=''):
    unique_labels = np.unique(labels)
    N = len(unique_labels)
    max_value = 16581375 #255**3
    interval = int(max_value / N)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]    
    color= [(int(i[:2], 16)/float(255), int(i[2:4], 16)/float(255), 
             int(i[4:], 16)/float(255)) for i in colors]
    i=0;
    plt.figure(figsize=(15,10))
    for label in unique_labels:
        ind = np.squeeze(labels == label)
        if label in stain: 
            plt.scatter(X[ind,0],X[ind,1],c='red',s=146,edgecolors='black',
                        lw = 0.5, alpha=1,marker='*',label=label)
        else:
            plt.scatter(X[ind,0],X[ind,1],c=color[i],s=36,edgecolors='gray',
                        lw = 0.5,label=label)        
        i+=1   
    plt.title(title)
    plt.legend(loc='upper right',bbox_to_anchor=(1.18, 1.01),ncol=nc)
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.axis('off')
    plt.savefig(plot_loc+ 'plot_stain.pdf')


##take TCC and TCC_dls, for set number of people [10, 30, 100, 300, 1000] save file and then calculate distance, saving time.
def timing_experiments(TCC_dist, TCC_dls_dist, num_cells, distribution_flname, distance_flname):
    num_processes=1
    distance_time=[]
    distance_time_dls=[]
    for num in num_cells:  
        TCC_dist_short= TCC_dist[0:num, :] 
        TCC_dls_dist_short= TCC_dls_dist[0:num, :]
        dist_flname='timing_exp/'+ distribution_flname + str(num) +'.dat'
        dist_dls_flname= 'timing_exp/'+ distribution_flname + '_dls_' + str(num) +'.dat'
        distan_flname='timing_exp/'+ distance_flname + str(num) +'.dat'
        distan_dls_flname= 'timing_exp/'+ distance_flname + '_dls_' + str(num) +'.dat'
        with open(dist_flname , 'wb') as outfile:
            pickle.dump(scipy.sparse.csr_matrix(TCC_dist_short.todense()), outfile, pickle.HIGHEST_PROTOCOL)
        with open(dist_dls_flname , 'wb') as outfile:
            pickle.dump(scipy.sparse.csr_matrix(TCC_dls_dist_short.todense()), outfile, pickle.HIGHEST_PROTOCOL)
        t=time()
        os.system('python get_pairwise_distances.py '+dist_flname +' '+distan_flname+' '+str(num_processes))
        distance_time.append( time() - t )
        t=time()
        os.system('python get_pairwise_distances.py '+dist_dls_flname +' '+distan_dls_flname+' '+str(num_processes))
        distance_time_dls.append( time() - t)
    return(distance_time, distance_time_dls)

def quadratic_fit(x, y):
    fitfunc = lambda p, x: p * x**2
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    pinit=10.
    out = scipy.optimize.leastsq(errfunc, pinit, args=(x, y), full_output=1)
    return(out[0], np.sqrt(out[1]))


def mean_var_plot(TCC, V, k, epsilon, min_id_keep, min_std_keep, data_type='TCC', plot_loc = ''):
    theta_TCC, index_keep_TCC, tau_sorted, index_drop_TCC = det_leverage(V, k, epsilon)
    TCC_dls= TCC[:, index_keep_TCC]
    columns_dls= TCC_dls.shape[1]
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False).fit(TCC)
    #wrong if data includes all zero columns.
    #min_std_keep= sorted(scaler.std_, reverse=True)[columns_dls]
    if python_version =='2':
        counts= np.array(TCC.sum(axis=0))[0,:]
        min_counts_keep= sorted(np.array(TCC.sum(axis=0))[0,:], reverse=True)[columns_dls]
    if python_version =='3':
        counts= np.array(TCC.sum(axis=0))
        min_counts_keep= sorted(np.array(TCC.sum(axis=0)), reverse=True)[columns_dls]
    fig, ax = plt.subplots()
    im=ax.scatter(counts[index_drop_TCC], scaler.std_[index_drop_TCC]**2, color='silver', s=16, edgecolors='gray', lw=0.5, label='DCSS dropped')
    im=ax.scatter(counts[index_keep_TCC], scaler.std_[index_keep_TCC]**2, color='gray', s=16, edgecolors='black', lw=0.5, label='DCSS retained')
    xdata=np.array([1e0, 1e8])
    ax.plot(xdata,min_id_keep*xdata/TCC.shape[0]  , color='black', ls='solid', lw=1, label='I.D.')
    ax.axvline(x=min_counts_keep, color='black', ls='dashed', lw=1, label='Count')
    ax.axhline(y=min_std_keep**2, color='gray', ls='solid', lw=2, label='Var.')
    ax.legend(loc='upper left')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Variance')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if data_type == 'TCC':
        ax.set_xlim([1e0, 1e8])
        ax.set_ylim([1e-4, 1e9])
    if data_type =='wishbone':
        ax.set_xlim([1e0, 2e5])
        ax.set_ylim([2e-4, 1e3])
    fig.tight_layout()
    plt.savefig(plot_loc + 'plot_mean_var_' +str(k)+ '_' +str(epsilon).replace('.', '') +'.pdf')


def zeisel_powerlaw_meanvar_index_comparision_gene_set(TCC, k, epsilon, analysis_dir, plot_loc, num_processes, TCC_flname, TCC_dist_flname, TCC_distance_flname):
#making a bunch of plots and calculating the umi depth
    method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
    TCC_flnames=[]
    TCC_dist_flnames=[]
    TCC_distance_flnames=[]

    for i, item in enumerate(method):
        TCC_flnames.append( TCC_flname+"_" + method[i]+ ".dat" )
        TCC_dist_flnames.append( TCC_dist_flname+"_" + method[i]+  ".dat" )
        TCC_distance_flnames.append( TCC_distance_flname+"_" + method[i]+  ".dat" )

        # TCC_flnames.append( TCC_base_flname_ss+"_" + method[i]+ ".dat" )
        # TCC_dist_flnames.append( TCC_dist_base_flname_ss+"_" + method[i]+  ".dat" )
        # TCC_distance_flnames.append( TCC_distance_base_flname_ss+"_" + method[i]+  ".dat" )
    # TCC_dls_flname=TCC_base_flname_ss+"_DCSS_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_dls_dist_flname=TCC_dist_base_flname_ss+ "_DCSS_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_dls_distance_flname=TCC_distance_base_flname_ss+ "_DCSS_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_vf_dist_flname=TCC_dist_base_flname_ss+ "_vf_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_vf_distance_flname=TCC_distance_base_flname_ss+ "_vf_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_lc_dist_flname=TCC_dist_base_flname_ss+ "_lc_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_lc_distance_flname=TCC_distance_base_flname_ss+ "_lc_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_id_dist_flname=TCC_dist_base_flname_ss+ "_id_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    # TCC_id_distance_flname=TCC_distance_base_flname_ss+ "_id_" +str(k)+'_' +str(epsilon).replace('.', '')+ ".dat"
    TCC_svd = scipy.sparse.linalg.svds(TCC, k)
    V=pd.DataFrame(TCC_svd[2].T)
    #(TCC_dls, TCC_dls_dist, D_dls, theta_TCC, index_keep_TCC, tau_sorted)= make_leverage_norm_jsdistance(TCC, V, k, epsilon, TCC_dls_flname, TCC_dls_dist_flname, TCC_dls_distance_flname, num_processes, plot_loc)
    (TCC_dls, TCC_dls_dist, D_dls, theta_TCC, index_keep_TCC, tau_sorted)= make_leverage_norm_jsdistance(TCC, V, k, epsilon, TCC_flnames[0], TCC_dist_flnames[0], TCC_distance_flnames[0], num_processes, plot_loc)
    (nsamp, columns_dls)=TCC_dls.shape
    #powerlaw plot
    pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted[0:10000])), tau_sorted[0:10000], k, plot_loc)
    #umi depth for dls
    dls_umi_depth = TCC_dls.sum()/TCC.sum()
    with open(analysis_dir + 'TCC_umi_depth_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write(str(dls_umi_depth))
    (TCC_vf, TCC_vf_dist, D_vf, vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[1], TCC_distance_flnames[1], num_processes, 'std')
    (TCC_lc, TCC_lc_dist, D_lc, counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[2], TCC_distance_flnames[2], num_processes, 'counts')
    (TCC_id, TCC_id_dist, D_id, id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[3], TCC_distance_flnames[3], num_processes, 'index_disp')
    # (TCC_vf, TCC_vf_dist, D_vf, vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_vf_dist_flname, TCC_vf_distance_flname, num_processes, 'std')
    # (TCC_lc, TCC_lc_dist, D_lc, counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_lc_dist_flname, TCC_lc_distance_flname, num_processes, 'counts')
    # (TCC_id, TCC_id_dist, D_id, id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_id_dist_flname, TCC_id_distance_flname, num_processes, 'index_disp')
    #mean_var_plot
    mean_var_plot(TCC, V, k, epsilon, id_min_keep, vf_min_keep, 'TCC', plot_loc)
    nonzero_bool=(np.array(TCC.sum(axis=0))[0,:] >0)
    index_keep_TCC_vf= (np.array(range(TCC.shape[1]))[nonzero_bool])[vf_keep_bool]
    index_keep_TCC_lc= (np.array(range(TCC.shape[1]))[nonzero_bool])[counts_keep_bool]
    index_keep_TCC_id= (np.array(range(TCC.shape[1]))[nonzero_bool])[id_keep_bool]
    #files of indices for comparison
    #method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
    for i,item in enumerate([index_keep_TCC, index_keep_TCC_vf, index_keep_TCC_lc, index_keep_TCC_id]):
        with open(analysis_dir+'index_keep_'+ str(columns_dls)+ '_'+ method[i] + '.txt', 'w' ) as writefile:
            for it in list(item):
                writefile.write(str(it) + '\n')


    # ##for go analysis of DCSS gene set
    index_dir= analysis_dir[:-9]
    with open(index_dir+ 'eq_dict.dat', 'rb') as infile:
        eq_dict_tr= pickle.load(infile)

    transcript_set=set() 
    for item in index_keep_TCC:
        trans = eq_dict_tr[item].split(',')
        for tran in trans:
            transcript_set.add(tran)
            
    if '' in transcript_set:
        transcript_set.remove('') 

    with open(analysis_dir + 'TCC_transcript_set_size_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write(str(len(transcript_set)))

    tr_names_dict={}
    with open(index_dir+'kallisto_index/Zeisel_index.idx_tr_id_names.txt') as infile:
        for line in infile:
            line=line.replace('\n', '')
            splits= line.split('\t')
            tr_names_dict[splits[0]]=splits[1]

    tr_gene_names_dict={}
    with open(index_dir+ 'transcript_to_gene_map.txt') as infile:
        for line in infile:
            line=line.replace('\n', '')
            splits = line.split(' ')
            tr_gene_names_dict[splits[0]]= splits[1]

    # with open(analysis_dir+'EC_ENSG_DICT.dat') as infile:
    #     eq_dict_genes =pickle.load(infile)

    gene_set=set() 
    for item in transcript_set:
        gene_set.add(tr_gene_names_dict[tr_names_dict[item]])

    with open(analysis_dir + 'TCC_gene_set_size_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write(str(len(gene_set)))

    with open(analysis_dir+'go_gene_list_' +str(k) +'_' +str(epsilon).replace('.', '') +'.txt', 'wr') as outfile:
        for gene in gene_set:
            outfile.write(gene + '\n')

def make_zeisel_error(D, TCC, k, epsilon, analysis_dir, plot_loc, num_processes, TCC_flname, TCC_dist_flname, TCC_distance_flname):
#making a bunch of plots and calculating the umi depth
    method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
    TCC_flnames=[]
    TCC_dist_flnames=[]
    TCC_distance_flnames=[]

    for i, item in enumerate(method):
        TCC_flnames.append( TCC_flname+"_" + method[i]+ ".dat" )
        TCC_dist_flnames.append( TCC_dist_flname+"_" + method[i]+  ".dat" )
        TCC_distance_flnames.append( TCC_distance_flname+"_" + method[i]+  ".dat" )

    TCC_svd = scipy.sparse.linalg.svds(TCC, k)
    V=pd.DataFrame(TCC_svd[2].T)
    #(TCC_dls, TCC_dls_dist, D_dls, theta_TCC, index_keep_TCC, tau_sorted)= make_leverage_norm_jsdistance(TCC, V, k, epsilon, TCC_dls_flname, TCC_dls_dist_flname, TCC_dls_distance_flname, num_processes, plot_loc)
    (TCC_dls, TCC_dls_dist, D_dls, theta_TCC, index_keep_TCC, tau_sorted)= make_leverage_norm_jsdistance(TCC, V, k, epsilon, TCC_flnames[0], TCC_dist_flnames[0], TCC_distance_flnames[0], num_processes, plot_loc)
    (nsamp, columns_dls)=TCC_dls.shape
    #powerlaw plot
    #pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted[0:10000])), tau_sorted[0:10000], k, plot_loc)
    #umi depth for dls
    #dls_umi_depth = TCC_dls.sum()/TCC.sum()
    # with open(analysis_dir + 'TCC_umi_depth_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        # f.write(str(dls_umi_depth))
    (TCC_vf, TCC_vf_dist, D_vf, vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[1], TCC_distance_flnames[1], num_processes, 'std')
    (TCC_lc, TCC_lc_dist, D_lc, counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[2], TCC_distance_flnames[2], num_processes, 'counts')
    (TCC_id, TCC_id_dist, D_id, id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(TCC, columns_dls, TCC_dist_flnames[3], TCC_distance_flnames[3], num_processes, 'index_disp')
    #mean_var_plot
    #mean_var_plot(TCC, V, k, epsilon, id_min_keep, vf_min_keep, 'TCC', plot_loc)
    nonzero_bool=(np.array(TCC.sum(axis=0))[0,:] >0)
    index_keep_TCC_vf= (np.array(range(TCC.shape[1]))[nonzero_bool])[vf_keep_bool]
    index_keep_TCC_lc= (np.array(range(TCC.shape[1]))[nonzero_bool])[counts_keep_bool]
    index_keep_TCC_id= (np.array(range(TCC.shape[1]))[nonzero_bool])[id_keep_bool]
    nrep=10
    error=np.empty((nrep, 8))
    for rs in range(nrep):
        tcc_spectral_labels2 = spectral(2,D, rs)
        tcc_spectral_labels9 = spectral(9,D, rs)
        tcc_spectral_labels2_dls = spectral(2,D_dls, rs)
        tcc_spectral_labels9_dls = spectral(9,D_dls, rs)
        tcc_spectral_labels2_vf = spectral(2,D_vf, rs)
        tcc_spectral_labels9_vf = spectral(9,D_vf, rs)
        tcc_spectral_labels2_lc = spectral(2,D_lc, rs)
        tcc_spectral_labels9_lc = spectral(9,D_lc, rs)
        tcc_spectral_labels2_id = spectral(2,D_id, rs)
        tcc_spectral_labels9_id = spectral(9,D_id, rs)
        c2_dls_tcc = compute_clustering_accuracy(tcc_spectral_labels2_dls, tcc_spectral_labels2)
        c9_dls_tcc = compute_clustering_accuracy(tcc_spectral_labels9_dls, tcc_spectral_labels9)
        c2_vf_tcc = compute_clustering_accuracy(tcc_spectral_labels2_vf, tcc_spectral_labels2)
        c9_vf_tcc = compute_clustering_accuracy(tcc_spectral_labels9_vf,tcc_spectral_labels9)
        c2_lc_tcc = compute_clustering_accuracy(tcc_spectral_labels2_lc,tcc_spectral_labels2)
        c9_lc_tcc = compute_clustering_accuracy(tcc_spectral_labels9_lc,tcc_spectral_labels9)
        c2_id_tcc = compute_clustering_accuracy(tcc_spectral_labels2_id,tcc_spectral_labels2)
        c9_id_tcc = compute_clustering_accuracy(tcc_spectral_labels9_id,tcc_spectral_labels9)
        error[rs, 0] = c2_dls_tcc
        error[rs, 1] = c9_dls_tcc
        error[rs, 2] = c2_vf_tcc
        error[rs, 3] = c9_vf_tcc
        error[rs, 4] = c2_lc_tcc
        error[rs, 5] = c9_lc_tcc
        error[rs, 6] = c2_id_tcc
        error[rs, 7] = c9_id_tcc
    return(error)


def zeisel_time_report(TCC, k, epsilon, analysis_dir):
    start_time, start_resources = time(), resource_usage(RUSAGE_SELF)
    TCC_svd = scipy.sparse.linalg.svds(TCC, k)
    V=pd.DataFrame(TCC_svd[2].T)
    theta_TCC, index_keep_TCC, tau_sorted, index_drop = det_leverage(V, k, epsilon)
    end_resources, end_time = resource_usage(RUSAGE_SELF), time()
    with open(analysis_dir + 'TCC_time_report_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write('real:' +str(end_time - start_time) + '\n')
        f.write('sys:' +str(end_resources.ru_stime - start_resources.ru_stime)+ '\n')
        f.write('user:' + str(end_resources.ru_utime - start_resources.ru_utime)+ '\n')

def wishbone_time_report(scdata_raw, k, epsilon, analysis_dir):
    start_time, start_resources = time(), resource_usage(RUSAGE_SELF)###dls before norm
    U, lamb, Vt_raw = scipy.sparse.linalg.svds(scipy.sparse.csr_matrix(scdata_raw.data), k)
    V_raw=pd.DataFrame(Vt_raw.T)
    theta_dls, index_keep_dls,tau_sorted, index_drop = det_leverage(V_raw, k, epsilon)
    end_resources, end_time = resource_usage(RUSAGE_SELF), time()
    with open(analysis_dir + 'wishbone_time_report_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write('real:' +str(end_time - start_time) + '\n')
        f.write('sys:' +str(end_resources.ru_stime - start_resources.ru_stime)+ '\n')
        f.write('user:' + str(end_resources.ru_utime - start_resources.ru_utime)+ '\n')

def wishbone_my_setup(setup_dir):
    # install GSEA, diffusion components, and download data.
    tools_dir = setup_dir + '/tools'
    if not os.path.exists(tools_dir + '/DiffusionGeometry/'):
        shutil.unpack_archive(tools_dir + '/DiffusionGeometry.zip', tools_dir +
                              '/DiffusionGeometry/')
    if not os.path.exists(tools_dir + '/mouse/'):
        shutil.unpack_archive(tools_dir + '/mouse_gene_sets.tar.gz', tools_dir)
    if not os.path.exists(tools_dir + '/human/'):
        shutil.unpack_archive(tools_dir + '/human_gene_sets.tar.gz', tools_dir)
    if not os.path.exists( setup_dir +'/data/GSE72857_umitab.txt.gz'):
        # downloads mouse UMI from GSE72857 Transcriptional heterogeneity and lineage commitment in myeloid progenitors [single cell RNA-seq]
        os.system("wget -m -nH -nd -P "+ setup_dir + "/data/ ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72857/suppl/GSE72857%5Fumitab%2Etxt%2Egz")
    x=pd.read_csv(setup_dir + '/data/GSE72857_umitab.txt.gz', sep = '\t', compression="gzip")
    y=pd.read_csv(setup_dir + '/data/sample_scseq_data.csv', index_col=[0])
    scdata_raw= x.T.loc[y.index]
    scdata_raw = wishbone.wb.SCData(scdata_raw.astype('float'), data_type='sc-seq')
    return(scdata_raw)


def wishbone_powerlaw_meanvar_index_comparision_gene_set(scdata_raw, k, epsilon, analysis_dir, plot_loc):
    ###dls before norm
    U, lamb, Vt_raw = scipy.sparse.linalg.svds(scipy.sparse.csr_matrix(scdata_raw.data), k)
    V_raw=pd.DataFrame(Vt_raw.T)
    theta_dls, index_keep_dls,tau_sorted, index_drop = det_leverage(V_raw, k, epsilon)
    scdata_raw_dls = wishbone.wb.SCData(scdata_raw.data.iloc[:, index_keep_dls], 'sc-seq')
    (nsamp, columns_dls)=scdata_raw_dls.data.shape
    #powerlaw plot
    pl_amp, pl_index=powerlaw_fit(range(1, 1+ len(tau_sorted[0:5000])), tau_sorted[0:5000], k, plot_loc)
    #umi depth for dls
    scdata_umi_depth=scdata_raw_dls.data.sum().sum()/scdata_raw.data.sum().sum()
    with open(analysis_dir + 'scdata_umi_depth_'+str(k)+ '_'+ str(epsilon).replace('.', '')+ '.txt' ,'w') as f:
        f.write(str(scdata_umi_depth))
    #num_processes is a dummy variable
    num_processes=1
    (scdata_raw_vf,  vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'std', distance=False)
    (scdata_raw_lc,  counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'counts', distance=False)
    (scdata_raw_id,  id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'index_disp', distance=False)
    #mean_var_plot
    mean_var_plot(scdata_raw.data.values, V_raw, k, epsilon, id_min_keep, vf_min_keep, 'wishbone', plot_loc)
    nonzero_bool=(np.array(scdata_raw.data.sum(axis=0))>0)
    index_keep_vf= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[vf_keep_bool]
    index_keep_lc= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[counts_keep_bool]
    index_keep_id= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[id_keep_bool]
    #files of indices for comparison
    method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
    for i,item in enumerate([index_keep_dls, index_keep_vf, index_keep_lc, index_keep_id]):
        with open(analysis_dir + 'index_keep_' +str(columns_dls)+ '_'+ method[i] + '.txt', 'w' ) as writefile:
            for it in list(item):
                writefile.write(str(it) + '\n')
    ##for go analysis of DCSS gene set
    gene_set=set() 
    for item in scdata_raw.data.iloc[:, index_keep_dls].columns:
        genes=item.split(';')
        for gene in genes:
            gene_set.add(gene) 

    with open(analysis_dir + 'go_gene_list_DCSS_'+str(k)+ '_' + str(epsilon).replace('.', '')+ '.txt', 'w') as outfile:
        for gene in gene_set:
            outfile.write(gene + '\n')


def wishbone_pipeline(scdata, data_type, analysis_dir):
    #pipeline
    if not os.path.exists(analysis_dir+ 'mouse_marrow_scdata_' + data_type+ '.p'):
        #change case of column names
        scdata.data.columns = scdata.data.columns.str.upper()
        #normalize dls
        if scdata._normalized==False:
            scdata = scdata.normalize_scseq_data()
            print('normalized data', scdata._normalized) 
        #run pca (reduces to 100 dim)
        if scdata.pca is None:
            scdata.run_pca()
        # Run tsne, reduces to no_cmpnts, then does 2-d tsne
        NO_CMPNTS = 15
        # scdata.run_tsne(n_components=NO_CMPNTS, perplexity=30)
        # scdata_dls.run_tsne(n_components=NO_CMPNTS, perplexity=30)
        # Run diffusion 
        # first do PCA projection of data
        # then do sklearn.NearestNeighbors with default for metric, which is L_p metric, and default p=2, which is euclidean.
        # use the k nearest neighbors to do diffusion map stuff.  this is deterministic
        NO_CMPNTS_D=10
        if scdata._diffusion_eigenvalues is None:
            scdata.run_diffusion_map(n_diffusion_components=NO_CMPNTS_D,n_pca_components=NO_CMPNTS)
        # run correlations
        if scdata._diffusion_map_correlations is None:
            scdata.run_diffusion_map_correlations()
    else:
        NO_CMPNTS_D=10
        scdata=wishbone.wb.SCData.load(analysis_dir+'mouse_marrow_scdata_' + data_type+ '.p')
    #run gene set enrichement
    if not os.path.exists(analysis_dir+'sc_mouse_reports/'):
        os.makedirs(analysis_dir+'sc_mouse_reports/')
    if not os.path.exists(analysis_dir+'sc_mouse_reports/'+data_type+'/'):
        reports = scdata.run_gsea(output_stem=analysis_dir+ 'sc_mouse_reports/'+data_type+'/', gmt_file=('mouse', 'gofat.bp.v1.0.gmt.txt'))
        #examine report and pick directions
        # defense response, taxis, phagocytosis, antigen processing, immune response
        with open(analysis_dir+'sc_mouse_reports/'+data_type+'/reports.dat', 'wb') as f:
            pickle.dump(reports, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(analysis_dir+'sc_mouse_reports/'+data_type+'/reports.dat', 'rb') as f:
            reports=pickle.load(f)
    components_list=[]
    for comp in range(1, NO_CMPNTS_D+ 1):
        try:
            for corr in ['pos', 'neg']:
                for item in reports[comp][corr].index:
                    b= [re.search(r'DEFENSE RESPONSE', item), re.search(r'TAXIS', item), re.search(r'PHAGOCYTOSIS', item), re.search(r'ANTIGEN', item),re.search(r'IMMUN', item)]
                    if b!=[None, None, None, None, None] and comp not in components_list:
                        components_list.append(comp)
        except KeyError:
            print('key error', comp)
    if not os.path.exists(analysis_dir +'mouse_marrow_scdata_' + data_type+ '.p'):
        scdata.save(analysis_dir + 'mouse_marrow_scdata_' + data_type+ '.p')
    return(components_list)

# def make_wishbone_assignments_all_random(scdata, components, analysis_dir, plot_loc, nrep=10):
#     """
#     this returns a list of wishbone objects, where each wishbone instance has been randomized differently
#     """
#     scdata_wb_list=[]
#     for rs in range(nrep):
#         np.random.seed(rs)
#         if os.path.exists(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p'):
#             wb_item= wishbone.wb.Wishbone.load(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p')
#         else:
#             wb_item = wishbone.wb.Wishbone(scdata)
#             wb_item.run_wishbone(start_cell='W30258', components_list=components, num_waypoints=250)
#             wb_item.save(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p')
#         scdata_wb_list.append(wb_item)
#         if rs==0:
#             vals, fig, ax = wb_item.plot_marker_trajectory(['CD34', 'GATA1', 'GATA2', 'MPO'])
#             plt.savefig(plot_loc+ 'plot_marker_trajectory_all_' + str(rs) +'.pdf')
#     return(scdata_wb_list)


def all_multi(scdata, components, analysis_dir, plot_loc, rs):
    np.random.seed(rs)
    if os.path.exists(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p'):
        wb_item= wishbone.wb.Wishbone.load(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p')
    else:
        wb_item = wishbone.wb.Wishbone(scdata)
        wb_item.run_wishbone(start_cell='W30258', components_list=components, num_waypoints=250)
        wb_item.save(analysis_dir + 'mouse_marrow_scdata_wb_all_' +str(rs) + '.p')
    if rs==0:
        vals, fig, ax = wb_item.plot_marker_trajectory(['CD34', 'GATA1', 'GATA2', 'MPO'])
        plt.savefig(plot_loc+ 'plot_marker_trajectory_all_' + str(rs) +'.pdf')
    return(wb_item)

def all_multi_wrapper(args):
    (scdata, components, analysis_dir, plot_loc, rs)=args
    try:
        wb_item=all_multi(scdata, components, analysis_dir, plot_loc, rs)
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        with open(analysis_dir+'assertion_error_all_'+str(rs)+'_'+str(time())+'.txt', 'w') as f:
            f.write('An error occurred on line {} in statement {}'.format(line, text))
        wb_item=np.nan
    return(wb_item)



def make_wishbone_assignments_all_random_multi(scdata, components, analysis_dir, plot_loc, nrep=10, num_processes=20):
    """
    this returns a list of wishbone objects, where each wishbone instance has been randomized differently
    """
    pool = multiprocessing.Pool(num_processes)
    results = pool.map_async(all_multi_wrapper,
                                         [(scdata, components, analysis_dir, plot_loc, rs) for rs in range(nrep)])
    # set the pool to work
    pool.close()
    # party's over, kids
    pool.join()
    # wait for all tasks to finish
    results = results.get()
    return(results)


# def make_wishbone_error(scdata_raw, k, epsilon, scdata_wb_list, analysis_dir, plot_loc, nrep=10):
#     """
#     need to have run make_wishbone_assignments_all_random with nrep random starts first.  this makes scdata_wb_list
#     """
#     assert len(scdata_wb_list)==nrep
#     U, lamb, Vt_raw = scipy.sparse.linalg.svds(scipy.sparse.csr_matrix(scdata_raw.data), k)
#     V_raw=pd.DataFrame(Vt_raw.T)
#     theta_dls, index_keep_dls,tau_sorted, index_drop = det_leverage(V_raw, k, epsilon)
#     scdata_raw_dls = wishbone.wb.SCData(scdata_raw.data.iloc[:, index_keep_dls], 'sc-seq')
#     (nsamp, columns_dls)=scdata_raw_dls.data.shape
#     #num_processes is a dummy variable
#     num_processes=1
#     (scdata_raw_vf,  vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'std', distance=False)
#     (scdata_raw_lc,  counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'counts', distance=False)
#     (scdata_raw_id,  id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', num_processes, 'index_disp', distance=False)
#     nonzero_bool=(np.array(scdata_raw.data.sum(axis=0))>0)
#     index_keep_vf= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[vf_keep_bool]
#     index_keep_lc= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[counts_keep_bool]
#     index_keep_id= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[id_keep_bool]
#     #making a list of the raw data for four different methods 
#     scdata_raw_list=[]
#     method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
#     for i,item in enumerate([index_keep_dls, index_keep_vf, index_keep_lc, index_keep_id]):
#         scdata_raw_list.append(wishbone.wb.SCData(scdata_raw.data.iloc[:, item], 'sc-seq'))
#     #making a list of the components chosen for the four different methods
#     components_lists=[]
#     for i, item in enumerate(scdata_raw_list):
#         components_lists.append(wishbone_pipeline(item, method[i], analysis_dir))
#     #loading the piplelined data four the four methods
#     scdata_list=[]
#     for i, item in enumerate(scdata_raw_list):
#         scdata_list.append(wishbone.wb.SCData.load(analysis_dir + 'mouse_marrow_scdata_' + method[i]+ '.p'))
#     #performing the wishbone analysis nrep times (randomness) and saving the errors
#     error=np.zeros((nrep, 4))
#     for i, item in enumerate(scdata_list):
#         for rs in range(nrep):
#             np.random.seed(rs)
#             if components_lists[i]==[]:
#                 print('no meaningful components')
#                 error[rs, i]=np.nan
#             else:
#                 if os.path.exists(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p'):
#                     wb_item= wishbone.wb.Wishbone.load(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p')
#                 else:
#                     wb_item = wishbone.wb.Wishbone(item)
#                     wb_item.run_wishbone(start_cell='W30258', components_list=components_lists[i], num_waypoints=250)
#                     wb_item.save(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p')
#                 # if rs==0:
#                 #     vals, fig, ax = wb_item.plot_marker_trajectory(['CD34', 'GATA1', 'GATA2', 'MPO'])
#                 #     plt.savefig(plot_loc+ 'plot_marker_trajectory_' +method[i]+ '_' + str(rs) +'.pdf')
#                 error[rs, i] = compute_clustering_accuracy(scdata_wb_list[i].branch.values, wb_item.branch.values)
#     return(error)

# def error_multi(item, i, rs, components_lists, scdata_wb_list, analysis_dir, method):
#     np.random.seed(rs)
#     if components_lists[i]==[]:
#         print('no meaningful components')
#         return(np.nan)
#     else:
#         if os.path.exists(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p'):
#             wb_item= wishbone.wb.Wishbone.load(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p')
#         else:
#             wb_item = wishbone.wb.Wishbone(item)
#             wb_item.run_wishbone(start_cell='W30258', components_list=components_lists[i], num_waypoints=250)
#             wb_item.save(analysis_dir + 'mouse_marrow_scdata_wb_' + method[i]+'_' +str(rs) + '.p')
#         # if rs==0:
#         #     vals, fig, ax = wb_item.plot_marker_trajectory(['CD34', 'GATA1', 'GATA2', 'MPO'])
#         #     plt.savefig(plot_loc+ 'plot_marker_trajectory_' +method[i]+ '_' + str(rs) +'.pdf')
#         error =compute_clustering_accuracy(scdata_wb_list[i].branch.values, wb_item.branch.values)
#         return(error)

def error_multi(item, i, rs, components_list, scdata_wb, analysis_dir, method):
    np.random.seed(rs)
    if components_list==[]:
        print('no meaningful components')
        return(np.nan)
    else:
        if os.path.exists(analysis_dir + 'mouse_marrow_scdata_wb_' + method+'_' +str(rs) + '.p'):
            wb_item= wishbone.wb.Wishbone.load(analysis_dir + 'mouse_marrow_scdata_wb_' + method+'_' +str(rs) + '.p')
        else:
            wb_item = wishbone.wb.Wishbone(item)
            wb_item.run_wishbone(start_cell='W30258', components_list=components_list, num_waypoints=250)
            wb_item.save(analysis_dir + 'mouse_marrow_scdata_wb_' + method+'_' +str(rs) + '.p')
        # if rs==0:
        #     vals, fig, ax = wb_item.plot_marker_trajectory(['CD34', 'GATA1', 'GATA2', 'MPO'])
        #     plt.savefig(plot_loc+ 'plot_marker_trajectory_' +method[i]+ '_' + str(rs) +'.pdf')
        error =compute_clustering_accuracy(scdata_wb.branch.values, wb_item.branch.values)
        return(error)

def error_multi_wrapper(args):
    (item, i, rs, components_list, scdata_wb, analysis_dir, method)=args
    try:
        error=error_multi(item, i, rs, components_list, scdata_wb, analysis_dir, method)
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        with open(analysis_dir+'assertion_error_'+ method+ '_'+str(rs)+'_'+str(time())+'.txt', 'w') as f:
            f.write('An error occurred on line {} in statement {}'.format(line, text))
        error=np.nan
    return(error)

# def error_multi_wrapper(args):
#     (item, i, rs, components_lists, scdata_wb_list, analysis_dir, method)=args
#     try:
#         error=error_multi(item, i, rs, components_lists, scdata_wb_list, analysis_dir, method)
#     except AssertionError:
#         _, _, tb = sys.exc_info()
#         traceback.print_tb(tb)
#         # Fixed format
#         tb_info = traceback.extract_tb(tb)
#         filename, line, func, text = tb_info[-1]
#         with open(analysis_dir+'assertion_error_'+ method[i]+ '_'+str(rs)+'_'+str(time())+'.txt', 'w') as f:
#             f.write('An error occurred on line {} in statement {}'.format(line, text))
#         error=np.nan
#     return(error)

def make_wishbone_error_multi(scdata_raw, k, epsilon, scdata_wb_list, analysis_dir, plot_loc, nrep=10, num_processes=40):
    """
    need to have run make_wishbone_assignments_all_random with nrep random starts first.  this makes scdata_wb_list
    """
    assert len(scdata_wb_list)==nrep
    U, lamb, Vt_raw = scipy.sparse.linalg.svds(scipy.sparse.csr_matrix(scdata_raw.data), k)
    V_raw=pd.DataFrame(Vt_raw.T)
    theta_dls, index_keep_dls,tau_sorted, index_drop = det_leverage(V_raw, k, epsilon)
    scdata_raw_dls = wishbone.wb.SCData(scdata_raw.data.iloc[:, index_keep_dls], 'sc-seq')
    (nsamp, columns_dls)=scdata_raw_dls.data.shape
    #num_processes is a dummy variable here
    (scdata_raw_vf,  vf_keep_bool, vf_umi_depth, vf_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', 1, 'std', distance=False)
    (scdata_raw_lc,  counts_keep_bool, lc_umi_depth, lc_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', 1, 'counts', distance=False)
    (scdata_raw_id,  id_keep_bool, id_umi_depth, id_min_keep)=make_norm_jsdistance(scdata_raw.data.values, columns_dls, 'none', 'none', 1, 'index_disp', distance=False)
    nonzero_bool=(np.array(scdata_raw.data.sum(axis=0))>0)
    index_keep_vf= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[vf_keep_bool]
    index_keep_lc= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[counts_keep_bool]
    index_keep_id= (np.array(range(scdata_raw.data.shape[1]))[nonzero_bool])[id_keep_bool]
    #making a list of the raw data for four different methods 
    scdata_raw_list=[]
    method=['DCSS_' + str(k)+ '_' + str(epsilon).replace('.', '') , 'vf_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'lc_' + str(k)+ '_' + str(epsilon).replace('.', ''), 'id_' + str(k)+ '_' + str(epsilon).replace('.', '')]
    for i,item in enumerate([index_keep_dls, index_keep_vf, index_keep_lc, index_keep_id]):
        scdata_raw_list.append(wishbone.wb.SCData(scdata_raw.data.iloc[:, item], 'sc-seq'))
    #making a list of the components chosen for the four different methods
    components_lists=[]
    for i, item in enumerate(scdata_raw_list):
        components_lists.append(wishbone_pipeline(item, method[i], analysis_dir))
    #loading the piplelined data four the four methods
    scdata_list=[]
    for i, item in enumerate(scdata_raw_list):
        scdata_list.append(wishbone.wb.SCData.load(analysis_dir + 'mouse_marrow_scdata_' + method[i]+ '.p'))
    #performing the wishbone analysis nrep times (randomness) and saving the errors
    error=np.zeros((nrep, 4))
    # for i, item in enumerate(scdata_list):
    #     for rs in range(nrep):
    #         error[rs, i]= error_multi_wrapper(item, i, rs, components_lists, analysis_dir, method)
    #multiprocessing
    pool = multiprocessing.Pool(num_processes)
    # results = pool.map_async(error_multi_wrapper,
    #                                      [(item, i, rs, components_lists, scdata_wb_list, analysis_dir, method)
    #                                       for i, item in enumerate(scdata_list) for rs in range(nrep)])
    results = pool.map_async(error_multi_wrapper,
                                         [(item, i, rs, components_lists[i], scdata_wb_list[i], analysis_dir, method[i])
                                          for i, item in enumerate(scdata_list) for rs in range(nrep)])
    # set the pool to work
    pool.close()
    # party's over, kids
    pool.join()
    # wait for all tasks to finish
    results = results.get()
    for i, item in enumerate(scdata_list):
        for rs in range(nrep):
            error[rs, i] = results[i*nrep + rs] 
    return(error)


def plot_error_rate(err, kset, eork_fixed, k, epsilon, num_clust='2', data_type='TCC', plot_loc=''):
    colors = ['black','gray','silver','gainsboro']
    #colors = ['red','deeppink','darkblue','cornflowerblue']
    ms= ['s','^', 'o', '8']
    lws= [1, 1, 2, 3]
    fig, ax = plt.subplots()
    if data_type=='TCC':
        for nc, column in enumerate( ['DCSS '+num_clust+' cluster', 'Var. '+num_clust+' cluster', 'Count '+num_clust+' cluster', 'I.D. '+num_clust+' cluster']):
            # ax.scatter(kset, err.loc[:, column], c=colors[nc], marker=ms[nc], s=36, edgecolors='gray', lw=0.5, label=column)
            ax.plot(kset, err.loc[:, column], c=colors[nc], marker=ms[nc], ls='solid', lw=lws[nc], label=column)

        ax.set_xticks(kset)
        if eork_fixed=='epsilon':
            ax.legend(loc='lower right')
            ax.set_xlabel(r'k for DCSS, $\epsilon$=' + str(epsilon))
        elif eork_fixed=='k':
            ax.legend(loc='lower left')
            ax.set_xlabel(r'$\epsilon$, for k=' + str(k)+' DCSS')
        ax.set_ylabel('Clustering adjusted Rand index')
        #ax.set_ylabel('Clustering error rate (%)')
        #plt.title('Zeisel et al. Clustering Error Rate Compared to Full Data')
        fig.tight_layout()
        if eork_fixed=='epsilon':
            plt.savefig(plot_loc+ 'plot_clustering_error_'+ num_clust+'cluster_' +str(epsilon).replace('.', '')+'.pdf')
        elif eork_fixed=='k':
            plt.savefig(plot_loc+ 'plot_clustering_error_'+num_clust+'cluster_epsilon_' + str(k)+'.pdf')
    elif data_type=='wishbone':
        for nc, column in enumerate( ['DCSS', 'Var.', 'Count', 'I.D.']):
            # ax.scatter(kset, err.loc[:, column], c=colors[nc], marker= ms[nc], s=36, edgecolors='gray', lw=0.5, label=column)
            ax.plot(kset, err.loc[:, column], c=colors[nc], marker= ms[nc], ls='solid', lw=lws[nc], label=column)

        lgd= ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_xticks(kset)
        if eork_fixed=='epsilon':
            ax.set_xlabel(r'k for DCSS, $\epsilon$=' +str(epsilon))
        elif eork_fixed=='k':
            ax.set_xlabel(r'$\epsilon$, for k=' + str(k)+' DCSS')
        ax.set_ylabel('Branch assignment adjusted Rand index')
        #ax.set_ylabel('Branch assignment error rate (%)')
        #plt.title('Zeisel et al. Clustering Error Rate Compared to Full Data')
        if eork_fixed=='epsilon':
            plt.savefig(plot_loc+'plot_clustering_error_' +str(epsilon).replace('.', '')+'.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
        elif eork_fixed=='k':
            plt.savefig(plot_loc+'plot_clustering_error_epsilon_' + str(k)+'.pdf' ,  bbox_extra_artists=(lgd,), bbox_inches='tight')