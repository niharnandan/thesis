import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use("seaborn")
#%config InlineBackend.figure_format='retina'
#from pprint import pprint
COLORS = ["skyblue", "steelblue", "gray"]
ALPHAS = [1.0, 1.0, 0.45]
#from tqdm import tqdm
from fortran_model import doeclim_gmsl
import sys
import os

NUMBER = int(sys.argv[1])

if not os.path.exists('output/'):
    os.makedirs('output/')

if not os.path.exists('image/'):
    os.makedirs('image/')

'''asc = 1.1
t2co_in= 3.1
kappa_in= 3.5
alphasl_in= 3.4
Teq= -0.5
SL0=0

temp_out, heatflux_mixed_out, heatflux_interior_out, gmsl_out = \
doeclim_gmsl(asc, t2co_in, kappa_in, alphasl_in, Teq, SL0) '''

class parameter:
	
	def __init__(self, name, value = 0):
		self.name = name
		self.value = value

class CoupledModel:

	def __init__(self, values):
		
		self.cs = 3.1
		self.ovd = 3.5
		self.asc = 1.1
		self.T_0 = -0.06
		self.num = 0

		dfSealevel = pd.read_csv('GMSL_ChurchWhite2011_yr_2015.csv')
		dfTemperature = pd.read_csv('NOAA_IPCC_RCPtempsscenarios.csv')
		self.sealevel = np.array((dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).tolist()) - (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).mean()
		self.year = (dfSealevel.loc[(dfSealevel["year"] >= 1880) & (dfSealevel["year"]<=2009), "year"]).tolist()
		self.sealevel_sigma = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "uncertainty"]).tolist()
		self.dfTemperature = dfTemperature.loc[(dfTemperature["Time"] <= 2008) & (dfTemperature["Time"] >= 1880), "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"] - dfTemperature.loc[(dfTemperature["Time"] <= 1990) & (dfTemperature["Time"] >= 1961), "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"].mean()
		forcing = pd.read_csv( 'data/forcing_hindcast.csv')
		self.mod_time = forcing['year']
		
		self.alpha = parameter('alpha')
		self.alpha.value = values[self.alpha.name]
		self.Teq = parameter('Teq')
		self.Teq.value = values[self.Teq.name]
		self.S0 = parameter('S0')
		self.S0.value = values[self.S0.name]
		self.rho = parameter('rho')
		self.rho.value = values[self.rho.name]
		self.sigma_ar = parameter('sigma_ar')
		self.sigma_ar.value = values[self.sigma_ar.name]
		self.climate_sensitivity = parameter('climate_sensitivity')
		self.climate_sensitivity.value = values[self.climate_sensitivity.name]
		self.ocean_vertical_diffusivity = parameter('ocean_vertical_diffusivity')
		self.ocean_vertical_diffusivity.value = values[self.ocean_vertical_diffusivity.name]
		self.aerosol_scaling = parameter('aerosol_scaling')
		self.aerosol_scaling.value = values[self.aerosol_scaling.name]
		self.T_0 = parameter('T_0')
		self.T_0.value = values[self.T_0.name]
		self.sigma_T = parameter('sigma_T')
		self.sigma_T.value = values[self.sigma_T.name]
		self.rho_T = parameter('rho_T')
		self.rho_T.value = values[self.rho_T.name]
		self.values = values
		self.stepsizes = {'alpha': .01, 'Teq': .05, 'S0': 5, 'rho': .001, 'sigma_ar': .1, 'climate_sensitivity': 0.16, 'ocean_vertical_diffusivity': 0.17, 'aerosol_scaling': 0.025, 'T_0': 0.03, 'sigma_T': 5e-4, 'rho_T': 0.007}
		
	def update_cov(self, X, s_d, size):
		a = []
		for i in range(size):
			a.append(X[:,i])
		a = np.array(a)
		cov = np.cov(a)
		eps = 0.0001
		I_d = np.identity(size)
		return s_d*cov + I_d*eps*s_d

	def prior(self, theta, sealevel_0, unc_0):

		#unpacking the variables individually for clarity

		alpha, Teq, S0, rho, sigma_ar = (
			theta[0], theta[1], theta[2], theta[3], theta[4])
		log_prior = 0
		if (self.alpha.value): log_prior += stats.uniform.logpdf(alpha, loc = 0, scale = 5) #lb and ub?
		if (self.Teq.value): log_prior += stats.uniform.logpdf(Teq, loc=-3, scale = 4)
		if (self.S0.value): log_prior += stats.norm.logpdf(S0, loc = sealevel_0, scale = unc_0)
		if (self.rho.value): log_prior += stats.uniform.logpdf(rho, loc = 0, scale = 1)
		if (self.sigma_ar.value): log_prior += stats.uniform.logpdf(sigma_ar, loc = 0, scale = 5)
		if (self.climate_sensitivity.value): log_prior += stats.uniform.logpdf(theta[5], loc = 0.1, scale = 9.9)
		if (self.ocean_vertical_diffusivity.value): log_prior += stats.uniform.logpdf(theta[6], loc = 0.1, scale = 3.9)
		if (self.aerosol_scaling.value): log_prior += stats.uniform.logpdf(theta[7], loc = 0, scale = 2)
		if (self.T_0.value): log_prior += stats.uniform.logpdf(theta[8], loc = -0.3, scale = 0.6)
		if (self.sigma_T.value): log_prior += stats.uniform.logpdf(theta[9], loc = 0.05, scale = 5.05)
		if (self.rho_T.value): log_prior += stats.uniform.logpdf(theta[10], loc =  0, scale = 0.999)
		return log_prior

	def build_ar1(self, rho, sigma_ar, length):
		ar1 = []
		for i in range(length):
			temp1 = [rho**j for j in range(i,0,-1)]
			temp2 = [rho**j for j in range(length-i)]
			ar1.append(np.array(temp1+temp2))
		ar1 = np.multiply(np.array(ar1), (sigma_ar**2)/(1-rho**2))
		return ar1

	def logp(self, theta, deltat, temperatures, model):

		N = len(self.sealevel)
		alpha, Teq, S0, rho, sigma_ar = (
		theta[0], theta[1], theta[2], theta[3], theta[4])
		resid = np.array([self.sealevel[i] - model[i] for i in range(len(model))])

		sigma_obs = np.diag([i**2 for i in self.sealevel_sigma])
		sigma_ar1 = self.build_ar1(rho, sigma_ar, N) if self.sigma_ar.value else []

		t_residual = self.dfTemperature - temperatures

		sigma_ar1_T = self.build_ar1(theta[10], theta[9], N) if self.sigma_T.value else []
		sigma_ar1_T = np.multiply((np.transpose(sigma_ar1_T) + sigma_ar1_T), 1/2)
		log_prior = self.prior(theta, self.sealevel[0], self.sealevel_sigma[0])
		if np.isinf(log_prior): return -np.inf
		
		cov = np.add(sigma_obs,sigma_ar1)
		cov = np.multiply((np.transpose(cov) + cov), 1/2)
		
		log_likelihood = stats.multivariate_normal.logpdf(resid, cov=cov) if model else 0
		
		log_likelihood_T = stats.multivariate_normal.logpdf(t_residual, cov=sigma_ar1_T) if self.sigma_T.value else 0
		log_posterior = log_likelihood + log_prior + log_likelihood_T
		return log_posterior
	
	def chain(self, deltat, N=10000):
		
		parameters = self.values.values()
		parameters = list(filter(lambda a: a != 0, parameters))
		self.num = len(parameters)
		theta = parameters
		print('Initial estimate for parameters -', theta)
		temp_out, heatflux_mixed_out, heatflux_interior_out, gmsl_out = \
		doeclim_gmsl(asc = theta[7], t2co_in = theta[5], kappa_in=theta[6], alphasl_in = theta[0], Teq = theta[1], SL0 = theta[2]) 
		temp_out += self.T_0.value

		lp = self.logp(theta, deltat, temp_out, gmsl_out)
		theta_best = theta
		lp_max = lp
		theta_new = [0.] * len(parameters)
		accepts = 0
		mcmc_chains = np.array([np.zeros(len(parameters))] * N)
		step = []
		count = 0
		temp = self.stepsizes.values()
		for i in range(len(parameters)):
			temp = [0]*(len(parameters))
			temp[count] = temp[i]
			count += 1
			step.append(temp)
		step = np.array(step)
		sd = 2.38**2 / len(theta)
		#Check if converged. If not keep running. 
		print(N)
		print(theta)
		for i in (range(N)):
			if i > 500: step = self.update_cov(mcmc_chains[:i], sd, len(parameters))
			theta_new = list(np.random.multivariate_normal(theta, step))
			if len(parameters) > 10: 
				temp_out, heatflux_mixed_out, heatflux_interior_out, gmsl_out = \
				doeclim_gmsl(asc = theta[7], t2co_in = theta[5], kappa_in=theta[6], alphasl_in = theta[0], Teq = theta[1], SL0 = theta[2]) 

			lp_new = self.logp(theta_new, deltat, temp_out, gmsl_out)
			lq = lp_new - lp
			
			lr = np.log(np.random.uniform(0, 1))
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

values = {'alpha': 3.4, 'Teq': -0.5, 'S0': -100, 'rho': .5, \
'sigma_ar': 3, 'climate_sensitivity': 3.1, 'ocean_vertical_diffusivity': 3.5, \
'aerosol_scaling': 1.1, 'T_0': -0.06, 'sigma_T': 0.1, 'rho_T': 0.55}

CM = CoupledModel(values)

mcmc_chain,accept_rate = CM.chain(1, NUMBER)

pamnames = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0', 'sigma_T', 'rho_T']

for i in range(CM.num):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.plot(mcmc_chain[int(NUMBER/2): ,i])
	ax.set_title(pamnames[i])
	fig.savefig('image/plot'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)

for i in range(CM.num):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.hist(mcmc_chain[int(NUMBER/2): ,i], density = True, facecolor='green', alpha=0.5, bins = 20, edgecolor = 'white')
	ax.set_title(pamnames[i])
	fig.savefig('image/hist'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)

