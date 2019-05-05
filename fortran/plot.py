import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmsl_model
import seaborn as sns
from fortran_model import doeclim_gmsl
import gmsl_model
import doeclimF
import forcing_total
plt.style.use("fivethirtyeight")

COLORS = ["skyblue", "steelblue", "gray"]
ALPHAS = [1.0,1.0,0.45]

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
'''
mcmc_big = temp.values
mcmc_big = mcmc_big[:25000]
NUMBER = len(mcmc_chain)

temp = pd.read_csv('array_uncorr.csv')
indices = [str(i) for i in range(0,13)]
temp = temp[indices]
mcmc_chain_uncorr = temp.values
mcmc_big_uncorr = temp.values

temp_chain1, temp_chain2 = [], []

if NUMBER >= 50000:
	temp_chain1 = mcmc_chain[100000:150000]
	temp_chain2 = mcmc_chain[150000:]
	conv = diagnostic([temp_chain1, temp_chain2])
mcmc_chain_1 = mcmc_chain[100000:150000]
mcmc_chain__1_uncorr = mcmc_chain_uncorr[100000:150000]
mcmc_chain = mcmc_chain[100000:]

mcmc_chains = [temp_chain1, temp_chain2]




#jump = correlation(mcmc_chain)
temp = []
jump = 257
for i in range(0,len(mcmc_chain),jump):
	temp.append(mcmc_chain[i])
temp = np.array(temp)
print(jump)'''


for i in range(16):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))  # create figure & 1 axis
	ax.plot(mcmc_chain[: ,i])
	ax.set_title(pamnames[i])
	fig.savefig('image/plot_'+pamnames[i]+'.png')   # save the figure to file
	plt.close(fig)

for i in range(16):
	if i == 5: continue
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	sns.distplot(mcmc_chain[: ,i], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws ={'linewidth':4, 'alpha':1.0})
	#ax.set_title(pamnames[i])
	fig.savefig('image/hist_'+pamnames[i]+'.png')   # save the figure to file
	plt.close(fig)

'''
fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
sns.distplot(mcmc_chain_1[: ,5], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black', 'alpha':0.2}, kde_kws ={'linewidth':4}, label='Correlated Model')
sns.distplot(mcmc_chain__1_uncorr[: ,5], hist=True, kde=True, color = 'darkgreen', hist_kws={'edgecolor':'black', 'alpha':0.2}, kde_kws ={'linewidth':4}, label='Uncorrelated Model')
#ax.set_title(pamnames[i])
ax.legend()
fig.savefig('image/hist_'+pamnames[5]+'.png')   # save the figure to file
plt.close(fig)


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
forcing = pd.read_csv( 'data/forcing_rcp45.csv')	
mod_time = np.array(range(1880,2100))

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=med[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=med[5], kappa=med[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= med[8]
gmsl_out = gmsl_model.gmsl_model(med, temp_out, 1)
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))
plt.plot(x, gmsl_out, 'k', color='#CC4F1B', linewidth=5)
#noise = [np.random.uniform(i-20, i+20) for i in gmsl_outm]
#plt.scatter(x, noise)

#plt.fill_between(x, gmsl_outl, gmsl_outh,
#    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.yticks([])
plt.savefig('gmsl.png')
plt.show()
