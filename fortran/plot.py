import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmsl_model
import seaborn as sns
from fortran_model import doeclim_gmsl
import gmsl_model
import doeclimF
import forcing_total

COLORS = ["skyblue", "steelblue", "gray","black"]
ALPHAS = [1.0,1.0,0.45,1.0]

pamnames = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0' \
, 'sigma_T', 'rho_T', 'sigmatsl', 'sigma_O', 'rho_O', 'sigmaslo', 'sigmaot']

pamnamesc = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0' \
, 'sigma_T', 'rho_T', 'sigma_O', 'rho_O']

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

temp = pd.read_csv('array_correlated.csv')
indices = [str(i) for i in range(0,16)]
temp = temp[indices]
mcmc_chain = temp.values
mcmc_big = temp.values
NUMBER = len(mcmc_chain)

temp = pd.read_csv('array_uncorr.csv')
indices = [str(i) for i in range(0,13)]
temp = temp[indices]
mcmc_chain_uncorr = temp.values
mcmc_chain = mcmc_chain_uncorr[:11000]
mcmc_big_uncorr = temp.values




#jump = correlation(mcmc_chain)
'''
temp = []
jump = 453
for i in range(0,len(mcmc_chain),jump):
	temp.append(mcmc_chain[i])
temp = np.array(temp)
print(jump)
'''
for i in range(13):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,4))  # create figure & 1 axis
	ax.plot(mcmc_chain[: ,i])
	ax.set_title(pamnamesc[i])
	fig.savefig('image/plot_'+pamnames[i]+'.png')   # save the figure to file
	plt.close(fig)

for i in range(13):
	#if i == 5: continue
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	sns.distplot(mcmc_chain[: ,i], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws ={'linewidth':4, 'alpha':1.0})

	ax.set_title(pamnamesc[i])
	fig.savefig('image/hist_'+pamnamesc[i]+'.png')   # save the figure to file
	plt.close(fig)
exit()
mcmc_chain_uncorr = mcmc_chain_uncorr[100000:150000]
fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
sns.distplot(mcmc_chain[: ,5], hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black', 'alpha':0.2}, kde_kws ={'linewidth':4}, label='Correlated Model')
sns.distplot(mcmc_big_uncorr[: ,5], hist=True, kde=True, color = 'darkgreen', hist_kws={'edgecolor':'black', 'alpha':0.2}, kde_kws ={'linewidth':4}, label='Uncorrelated Model')
#ax.set_title(pamnames[i])
ax.legend()
fig.savefig('image/hist_'+pamnames[5]+'.png')   # save the figure to file
plt.close(fig)

'''
R = [(diagnostic(mcmc_chains[:,0,:]))]
burn_in = 1
while burn_in < len(mcmc_chain1):
	R.append(diagnostic(mcmc_chains[:,:burn_in,:]))
	burn_in += len(mcmc_chain1)//100
#R.append(diagnostic(mcmc_chains[:,burn_in:,:]))
'''
low = np.percentile(mcmc_chain[150000:], 5, axis = 0)
high = np.percentile(mcmc_chain[150000:], 95, axis = 0)
med = np.mean(mcmc_chain, axis = 0)
forcing = pd.read_csv( 'data/forcing_rcp45.csv')	
mod_time = np.array(range(1880,2100))

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=med[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=med[5], kappa=med[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= med[8]
gmsl_out = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_out = np.interp(gmsl_out, (np.min(gmsl_out), np.max(gmsl_out)), (-100, 150))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))
plt.plot(x, gmsl_out, 'k', color='#3498DB', linewidth=5, label="RCP 4.5")


forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=low[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=low[5], kappa=low[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= low[8]
gmsl_outl = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_outl = np.interp(gmsl_outl, (np.min(gmsl_outl), np.max(gmsl_outl)), (-100, 185))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=high[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=high[5], kappa=high[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= high[8]
gmsl_outh = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_outh = np.interp(gmsl_outh, (np.min(gmsl_outh), np.max(gmsl_outh)), (-100, 100))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))

plt.fill_between(x, gmsl_outl, gmsl_outh,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#3498DB')



forcing = pd.read_csv( 'data/forcing_rcp85.csv')	
mod_time = np.array(range(1880,2100))

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=med[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=med[5], kappa=med[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= med[8]
gmsl_out = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_out = np.interp(gmsl_out, (np.min(gmsl_out), np.max(gmsl_out)), (-100, 250))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))
plt.plot(x, gmsl_out, 'k', color='#CC4F1B', linewidth=5, label="RCP 8.5")
#noise = [np.random.uniform(i-20, i+20) for i in gmsl_outm]
#plt.scatter(x, noise)
forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=low[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=low[5], kappa=low[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= low[8]
gmsl_outl = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_outl = np.interp(gmsl_outl, (np.min(gmsl_outl), np.max(gmsl_outl)), (-100, 300))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=high[7], l_project=True, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=high[5], kappa=high[6])
temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2100), "temp"]).tolist())
temp_out -= high[8]
gmsl_outh = gmsl_model.gmsl_model(med, temp_out, 1)
gmsl_outh = np.interp(gmsl_outh, (np.min(gmsl_outh), np.max(gmsl_outh)), (-100, 180))
#plt.rcParams.update(plt.rcParamsDefault)
x = list(range(1880, 1880+len(gmsl_out)))

plt.fill_between(x, gmsl_outl, gmsl_outh,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
#plt.fill_between(x, gmsl_outl, gmsl_outh,
#    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.grid(True)
plt.ylabel('Sea-level (mm)')
plt.xlabel('Year')
plt.legend()
plt.savefig('gmsl.png')
plt.show()
