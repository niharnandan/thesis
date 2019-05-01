import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmsl_model
import seaborn as sns
from fortran_model import doeclim_gmsl

pamnames = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0' \
, 'sigma_T', 'rho_T', 'sigmatsl', 'sigma_O', 'rho_O', 'sigmaslo', 'sigmaot']

def diagnostic(mcmc_chains):
	N = len(mcmc_chains[0])
	m = len(mcmc_chains)
	s_j = [np.var(mcmc_chains[i]) for i in range(m)]
	W = 1 / m * np.sum(s_j)
	global_mean = np.mean([np.mean(mcmc_chains[i]) for i in range(m)])
	B = N / (m - 1) * np.sum([(np.mean(mcmc_chains[i]) - global_mean)**2
			      for i in range(m)])
	Var = (1 - 1/N)*W + 1/N*B
	return np.sqrt(Var/W)

def correlation(mcmc_chains):
    alpha = mcmc_chains[:,0]
    beta = mcmc_chains[:,1]
    cora, ra, flaga = [], 1, True
    corb, rb, flagb = [], 1, True
    for lag in range(1, len(mcmc_chains)):
        Xa,Xb = alpha[:len(alpha) - lag], beta[:len(beta) - lag]
        Ya,Yb = alpha[lag:], beta[lag:]
        cora.append(np.corrcoef(Xa, Ya, rowvar=False)[0][1])
        corb.append(np.corrcoef(Xb, Yb, rowvar=False)[0][1])
        if flaga and cora[-1] <= 0.05:
            flaga = False
            ra = lag
        if flagb and corb[-1] <= 0.05:
            flagb = False
            rb = lag
    return ra if ra > rb else rb

temp = pd.read_csv('array.csv')
indices = [str(i) for i in range(0,16)]
temp = temp[indices]
mcmc_chain = temp.values
NUMBER = len(mcmc_chain)

temp_chain1, temp_chain2 = [], []

if NUMBER >= 50000:
	temp_chain1 = mcmc_chain[100000:150000]
	temp_chain2 = mcmc_chain[150000:]
	conv = diagnostic([temp_chain1, temp_chain2])

mcmc_chain = mcmc_chain[100000:]
mcmc_chains = [temp_chain1, temp_chain2]




#jump = correlation(mcmc_chain)
temp = []
jump = 457
for i in range(0,len(mcmc_chain),jump):
	temp.append(mcmc_chain[i])
mcmc_chain = np.array(temp)
print(jump)


for i in range(16):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.plot(mcmc_chain[: ,i])
	#ax.set_title(pamnames[i])
	fig.savefig('image/plot_'+pamnames[i]+'.png')   # save the figure to file
	plt.close(fig)

for i in range(16):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	sns.distplot(mcmc_chain[: ,i], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws ={'linewidth':4})
	ax.set_title(pamnames[i])
	fig.savefig('image/hist_'+pamnames[i]+'.png')   # save the figure to file
	plt.close(fig)

'''
R = [(diagnostic(mcmc_chains[:,0,:]))]
burn_in = 1
while burn_in < len(mcmc_chain1):
	R.append(diagnostic(mcmc_chains[:,:burn_in,:]))
	burn_in += len(mcmc_chain1)//100
#R.append(diagnostic(mcmc_chains[:,burn_in:,:]))
'''
low = np.percentile(mcmc_chain, 5, axis = 0)
high = np.percentile(mcmc_chain, 95, axis = 0)
med = np.mean(mcmc_chain, axis = 0)
#print(low, high)
_,_,_ ,gmsl_outl = doeclim_gmsl(asc = low[7], t2co_in = low[5], kappa_in=low[6], alphasl_in = low[0], Teq = low[1], SL0 = low[2], forcing='forcing_hindcast')
_,_,_ ,gmsl_outh = doeclim_gmsl(asc = high[7], t2co_in = high[5], kappa_in=high[6], alphasl_in = high[0], Teq = high[1], SL0 = high[2], forcing='forcing_hindcast')
_,_,_ ,gmsl_outm = doeclim_gmsl(asc = med[7], t2co_in = med[5], kappa_in=med[6], alphasl_in = med[0], Teq = med[1], SL0 = med[2], forcing='forcing_hindcast')

x = list(range(1880, 1880+len(gmsl_outl)))
plt.plot(x, gmsl_outm, 'k', color='#CC4F1B')


#plt.fill_between(x, gmsl_outl, gmsl_outh,
#    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.yticks([])
plt.savefig('gmsl.png')
plt.show()
