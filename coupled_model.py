import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use("seaborn")
#%config InlineBackend.figure_format='retina'
from pprint import pprint
COLORS = ["skyblue", "steelblue", "gray"]
ALPHAS = [1.0, 1.0, 0.45]

import gmsl_model
import os
import forcing_total
import doeclimF
from tqdm import tqdm
import sys
from scipy.interpolate import make_interp_spline, BSpline

NUMBER = int(sys.argv[1])

if not os.path.exists('data/'):
    print("FATAL ERROR: No data directory")
    exit()

if not os.path.exists('output/'):
    os.makedirs('output/')

if not os.path.exists('image/'):
    os.makedirs('image/')

climate_sensitivity = 3.1
ocean_vertical_diffusivity = 3.5
aerosol_scaling = 1.1
T_0 = -0.06

dfSealevel = pd.read_csv('GMSL_ChurchWhite2011_yr_2015.csv')
dfTemperature = pd.read_csv('NOAA_IPCC_RCPtempsscenarios.csv')
sealevel = np.array((dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).tolist()) - (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).mean()
year = (dfSealevel.loc[(dfSealevel["year"] >= 1880) & (dfSealevel["year"]<=2009), "year"]).tolist()
sealevel_sigma = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "uncertainty"]).tolist()
dfTemperature = dfTemperature.loc[(dfTemperature["Time"] <= 2008) & (dfTemperature["Time"] >= 1880), "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"] - dfTemperature.loc[(dfTemperature["Time"] <= 1990) & (dfTemperature["Time"] >= 1961), "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"].mean()
forcing = pd.read_csv( 'data/forcing_hindcast.csv')
mod_time = forcing['year']

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=climate_sensitivity, kappa=ocean_vertical_diffusivity)
temperatures = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
	
def update_cov(X, s_d, size):
	a = []
	for i in range(size):
		a.append(X[:,i])
	a = np.array(a)
	cov = np.cov(a)
	eps = 0.0001
	I_d = np.identity(size)
	return s_d*cov + I_d*eps*s_d
#Online update 

def prior(theta, sealevel_0, unc_0, pamnames):
	log_prior = 0
	#unpacking the variables individually for clarity

	alpha, Teq, S0, rho, sigma_ar = (
		theta[0], theta[1], theta[2], theta[3], theta[4])

	if ('alpha' in pamnames):
		log_prior += stats.uniform.logpdf(alpha, loc = 0, scale = 5) #lb and ub?
	if ('Teq' in pamnames):
		log_prior += stats.uniform.logpdf(Teq, loc=-3, scale = 4)
	if ('S0' in pamnames):
		log_prior += stats.norm.logpdf(S0, loc = sealevel_0, scale = unc_0)
	if ('rho' in pamnames):
		log_prior += stats.uniform.logpdf(rho, loc = 0, scale = 1)
	if ('sigma_ar' in pamnames):
		log_prior += stats.uniform.logpdf(sigma_ar, loc = 0, scale = 5)
	if ('climate_sensitivity' in pamnames):
		log_prior += stats.uniform.logpdf(theta[5], loc = 0.1, scale = 9.9)
	if ('ocean_vertical_diffusivity' in pamnames):
		log_prior += stats.uniform.logpdf(theta[6], loc = 0.1, scale = 3.9)
	if ('aerosol_scaling' in pamnames):
		log_prior += stats.uniform.logpdf(theta[7], loc = 0, scale = 2)
	if ('T_0' in pamnames):
		log_prior += stats.uniform.logpdf(theta[8], loc = -0.3, scale = 0.6)
	if ('sigma_T' in pamnames):
		log_prior += stats.uniform.logpdf(theta[9], loc = 0.05, scale = 5.05)
	if ('rho_T' in pamnames):
		log_prior += stats.uniform.logpdf(theta[10], loc =  0, scale = 0.999)
	return log_prior

def logp(theta, sealevel, deltat, temperatures, model, pamnames, sigma=sealevel_sigma):
	if model:
		N = len(sealevel)
		alpha, Teq, S0, rho, sigma_ar = (
		theta[0], theta[1], theta[2], theta[3], theta[4])
		resid = np.array([sealevel[i] - model[i] for i in range(len(model))])

		sigma_obs = np.diag([i**2 for i in sigma])
		sigma_ar1 = build_ar1(rho, sigma_ar, N) if ('sigma_ar' in pamnames) else []

	t_residual = dfTemperature - temperatures

	sigma_ar1_T = build_ar1(theta[10], theta[9], N) if ('sigma_T' in pamnames) else []
	sigma_ar1_T = np.multiply((np.transpose(sigma_ar1_T) + sigma_ar1_T), 1/2)
	log_prior = prior(theta, sealevel[0], sealevel_sigma[0], pamnames)
	if np.isinf(log_prior): return -np.inf
	
	cov = np.add(sigma_obs,sigma_ar1)
	cov = np.multiply((np.transpose(cov) + cov), 1/2)
	
	log_likelihood = stats.multivariate_normal.logpdf(resid, cov=cov) if model else 0
	
	log_likelihood_T = stats.multivariate_normal.logpdf(t_residual, cov=sigma_ar1_T) if ('sigma_T' in pamnames) else 0
	log_posterior = log_likelihood + log_prior + log_likelihood_T
	return log_posterior

def build_ar1(rho, sigma_ar, length):
    ar1 = []
    for i in range(length):
        temp1 = [rho**j for j in range(i,0,-1)]
        temp2 = [rho**j for j in range(length-i)]
        ar1.append(np.array(temp1+temp2))
    ar1 = np.multiply(np.array(ar1), (sigma_ar**2)/(1-rho**2))
    return ar1


def update_mean(m, X):
    N = len(X[0])
    n = []
    for i in range(len(m)):
        n.append([(m[i][0]*(N-1) + X[i][-1])/N])
    return np.array(n)

def online_update_cov(X, m, Ct, Sd, eps=0.0001):
    I_d = np.identity(9)
    m1 = update_mean(m, X)
    t = len(X[0])-1
    part1 = ((t-1)/t)*Ct
    part2 = t*np.matmul(m, np.transpose(m))
    part3 = (t+1)*np.matmul(m1, np.transpose(m1))
    Xt = []
    Xt.append(X[:,-1])
    part4 = np.matmul(np.transpose(Xt), Xt)
    part5 = eps*Id
    cov = part1 + (Sd/t)*(part2 - part3 + part4 + part5)
    return (cov + np.transpose(cov))/2, m1

def coupled_model(forcingtotal, doeclim_out, temperatures, model, parameters, mod_time, T_0, pamnames):
	if 'aerosol_scaling' in pamnames:
		forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=parameters[7], l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
	if 'climate_sensitivity' in pamnames and 'ocean_vertical_diffusivity' in pamnames:
		doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=parameters[5], kappa=parameters[6])
	temperatures = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
	temperatures = temperatures + T_0
	model = gmsl_model.gmsl_model(parameters, temperatures, deltat)

def chain(parameters, temperatures, deltat, sealevel, sealevel_sigma, pamnames, N=10000):
	if 'aerosol_scaling' in pamnames:
		forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
	if 'climate_sensitivity' in pamnames and 'ocean_vertical_diffusivity' in pamnames:
		doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=climate_sensitivity, kappa=ocean_vertical_diffusivity)
		temperatures = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
	temperatures = temperatures + T_0
	theta = parameters
	print('Initial estimate for parameters -', theta)

	model = gmsl_model.gmsl_model(theta, temperatures, deltat) 	if 'alpha' in pamnames else []

	lp = logp(theta, sealevel, deltat, temperatures, model, pamnames, sigma=sealevel_sigma)
	theta_best = theta
	lp_max = lp
	theta_new = [0.] * len(parameters)
	accepts = 0
	mcmc_chains = np.array([np.zeros(len(parameters))] * N)
	stepsizes = {'alpha': .01, 'Teq': .05, 'S0': 5, 'rho': .001, 'sigma_ar': .1, 'climate_sensitivity': 0.16, 'ocean_vertical_diffusivity': 0.17, 'aerosol_scaling': 0.025, 'T_0': 0.03, 'sigma_T': 5e-4, 'rho_T': 0.007}
	step = []
	count = 0
	for i in pamnames:
		temp = [0]*len(pamnames)
		temp[count] = stepsizes[i]
		count += 1
		step.append(temp)
	step = np.array(step)
	#step = np.array([[.01,0,0,0,0,0,0,0,0], [0,.05,0,0,0,0,0,0,0], [0,0,5,0,0,0,0,0,0], [0,0,0,0.001,0,0,0,0,0],[0,0,0,0,0.1,0,0,0,0],
	#				 [0,0,0,0,0,0.16,0,0,0], [0,0,0,0,0,0,0.17,0,0], [0,0,0,0,0,0,0,0.025,0],[0,0,0,0,0,0,0,0,0.03]])
	sd = 2.38**2 / len(theta)
	#Check if converged. If not keep running. 
	print(N)
	print(theta)
	for i in tqdm(range(N)):
		if i > 500: step = update_cov(mcmc_chains[:i], sd, len(parameters))
		theta_new = list(np.random.multivariate_normal(theta, step))
		if len(pamnames) > 10: 
			coupled_model(forcingtotal, doeclim_out, temperatures, model, parameters, mod_time, T_0, pamnames)
		model = gmsl_model.gmsl_model(theta, temperatures, deltat) 	if 'alpha' in pamnames else []

		#Separate above 3 lines + T_0 into a new file. 
		lp_new = logp(theta_new, sealevel, deltat, temperatures, model, pamnames, sigma=sealevel_sigma )
		lq = lp_new - lp
		
		lr = np.math.log(np.random.uniform(0, 1))
		#print(lr, lq)
		if (lr < lq):
			theta = theta_new
			lp = lp_new
			accepts += 1
			if lp > lp_max:
				theta_best = theta
				lp_max = lp
		mcmc_chains[i] = theta

	return mcmc_chains,accepts/N*100

pamnames = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar'] #, 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0', 'sigma_T', 'rho_T']
#pamnames = ['climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0', 'sigma_T', 'rho_T']
parameters = [3.4, -0.5, sealevel[0], 0.5, 3]#, climate_sensitivity, ocean_vertical_diffusivity, aerosol_scaling, T_0, 0.1, 0.55]
deltat = 1
mcmc_chain,accept_rate = chain(parameters, temperatures, deltat, sealevel, sealevel_sigma, pamnames, N=NUMBER)

for i in range(len(parameters)):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.plot(mcmc_chain[int(NUMBER/2): ,i])
	ax.set_title(pamnames[i])
	fig.savefig('image/plot'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)

for i in range(len(pamnames)):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.hist(mcmc_chain[int(NUMBER/2): ,i], density = True, facecolor='green', alpha=0.5, bins = 20, edgecolor = 'white')
	ax.set_title(pamnames[i])
	fig.savefig('image/hist'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)
