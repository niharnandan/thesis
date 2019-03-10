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
		#forcing = pd.read_csv( 'data/forcing_hindcast.csv')
		#self.mod_time = np.array(forcing['year'])
		#self.forcingtotal = np.array(forcing_total.forcing_total(forcing=forcing, alpha_doeclim=self.asc, l_project=False, begyear=self.mod_time[0], endyear=np.max(self.mod_time)))

		#self.alpha = parameter('alpha')
		self.alpha = values[0]
		#self.Teq = parameter('Teq')
		self.Teq = values[1]
		#self.S0 = parameter('S0')
		self.S0 = values[2]
		#self.rho = parameter('rho')
		self.rho = values[3]
		#self.sigma_ar = parameter('sigma_ar')
		self.sigma_ar = values[4]
		#self.climate_sensitivity = parameter('climate_sensitivity')
		self.climate_sensitivity = values[5]
		#self.ocean_vertical_diffusivity = parameter('ocean_vertical_diffusivity')
		self.ocean_vertical_diffusivity = values[6]
		#self.aerosol_scaling = parameter('aerosol_scaling')
		self.aerosol_scaling = values[7]
		#self.T_0 = parameter('T_0')
		self.T_0 = values[8]
		#self.sigma_T = parameter('sigma_T')
		self.sigma_T = values[9]
		#self.rho_T = parameter('rho_T')
		self.rho_T = values[10]
		self.values = values
		#self.stepsizes = {'alpha': .01, 'Teq': .05, 'S0': 5, 'rho': .001, 'sigma_ar': .1, 'climate_sensitivity': 0.16, 'ocean_vertical_diffusivity': 0.17, 'aerosol_scaling': 0.025, 'T_0': 0.03, 'sigma_T': 5e-4, 'rho_T': 0.007}
		self.stepsizes = [ .01,  .05,  5,  .001, .1,  0.16,  0.17,  0.025,0.03, 5e-4, 0.007]		
		#self.stepsizes = {'alpha': .01, 'Teq': .05, 'S0': 5, 'climate_sensitivity': 0.16, 'ocean_vertical_diffusivity': 0.17, 'aerosol_scaling': 0.025, 'T_0': 0.03}
		
	def update_cov(self, X, s_d, size):
		a = np.array(X)
		a = a.T
		cov = np.cov(a)
		eps = 0.0001
		I_d = np.identity(size)
		return s_d*cov + I_d*eps*s_d

	def prior(self, theta):

		#unpacking the variables individually for clarity

		alpha, Teq, S0, rho, sigma_ar = (
			theta[0], theta[1], theta[2], theta[3], theta[4])
		log_prior = 0
		if (self.alpha): log_prior += stats.uniform.logpdf(alpha, loc = 0, scale = 5) #lb and ub?
		if (self.Teq): log_prior +=  stats.uniform.logpdf(Teq, loc=-3, scale = 4)
		if (self.S0): log_prior +=  stats.norm.logpdf(S0, loc = self.sealevel[0], scale = self.sealevel_sigma[0])
		if (self.rho): log_prior +=  stats.uniform.logpdf(rho, loc = 0, scale = 1)
		if (self.sigma_ar): log_prior +=  stats.uniform.logpdf(sigma_ar, loc = 0, scale = 5)
		if (self.climate_sensitivity): log_prior +=  stats.uniform.logpdf(theta[5], loc = 0.1, scale = 9.9)
		if (self.ocean_vertical_diffusivity): log_prior +=  stats.uniform.logpdf(theta[6], loc = 0.1, scale = 3.9)
		if (self.aerosol_scaling): log_prior +=  stats.uniform.logpdf(theta[7], loc = 0, scale = 2)
		if (self.T_0): log_prior +=  stats.uniform.logpdf(theta[8], loc = -0.3, scale = 0.6)
		if (self.sigma_T): log_prior +=  stats.uniform.logpdf(theta[9], loc = 0.05, scale = 4.95)
		if (self.rho_T): log_prior +=  stats.uniform.logpdf(theta[10], loc =  0, scale = 0.999)
		#print(log_prior)
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


		log_prior = self.prior(theta)
		return log_prior 
		N = len(self.sealevel)
		alpha, Teq, S0 = (
		theta[0], theta[1], theta[2])
		rho, sigma_ar = 0.5, 3
		rho_t, sigma_t = 0.55, 1
		resid = np.array([self.sealevel[i] - model[i] for i in range(len(model))])

		sigma_obs = np.diag([i**2 for i in self.sealevel_sigma])
		sigma_ar1 = self.build_ar1(rho, sigma_ar, N) 

		t_residual = self.dfTemperature - temperatures

		sigma_ar1_T = self.build_ar1(rho_t, sigma_t, N)
		sigma_ar1_T = np.multiply((np.transpose(sigma_ar1_T) + sigma_ar1_T), 1/2)
		log_prior = self.prior(theta)
		if np.isinf(log_prior): return -np.inf
		
		cov = np.add(sigma_obs,sigma_ar1)
		cov = np.multiply((np.transpose(cov) + cov), 1/2)
		#cov = sigma_obs
		log_likelihood = stats.multivariate_normal.logpdf(resid, mean = None, cov=cov)
		
		cov_T = np.multiply((np.transpose(sigma_ar1_T) + sigma_ar1_T), 1/2)

		log_likelihood_T = stats.multivariate_normal.logpdf(t_residual, mean=None, cov=cov_T)
		log_posterior = log_likelihood + log_prior + log_likelihood_T
		return log_posterior

	def update_mean(self, m, X):
		N = len(X[0])
		n = []
		for i in range(len(m)):
			n.append([(m[i][0]*(N-1) + X[i][-1])/N])
		return np.array(n)

	def update_cov1(self, X, m, Ct, Sd, eps, size, t):
		Id = np.identity(size)
		m1 = update_mean(m, X)
		part1 = ((t-1)/t)*Ct
		part2 = t*np.matmul(m, np.transpose(m))
		part3 = (t+1)*np.matmul(m1, np.transpose(m1))
		Xt = []
		Xt.append(X[:,-1])
		part4 = np.matmul(np.transpose(Xt), Xt)
		part5 = eps*Id
		cov = part1 + (Sd/t)*(part2 - part3 + part4 + part5)
		return 0.5*(cov + np.transpose(cov)), m1

	def diagnostic(self, mcmc_chains):
		N = len(mcmc_chains[0])
		m = len(mcmc_chains)
		s_j = [np.var(mcmc_chains[i]) for i in range(m)]
		W = 1 / m * np.sum(s_j)
		global_mean = np.mean([np.mean(mcmc_chains[i]) for i in range(m)])
		B = N / (m - 1) * np.sum([(np.mean(mcmc_chains[i]) - global_mean)**2
				      for i in range(m)])
		Var = (1 - 1/N)*W + 1/N*B
		return np.sqrt(Var/W)
	
	def chain(self, deltat, N=10000):
		theta = np.array(self.values)
		print('Initial estimate for parameters -', theta)
		temp_out, heatflux_mixed_out, heatflux_interior_out, gmsl_out = \
		doeclim_gmsl(asc = theta[7], t2co_in = theta[5], kappa_in=theta[8], alphasl_in = theta[0], Teq = theta[1], SL0 = theta[2]) 
		temp_out += self.T_0

		lp = self.logp(theta, deltat, temp_out, gmsl_out)
		theta_best = theta
		lp_max = lp
		theta_new = [0.] * len(theta)
		accepts = 0
		mcmc_chains = np.zeros((N, (len(theta))))
		step = np.diag(self.stepsizes)
		sd = 2.38**2 / len(theta)
		#step = 0.5 * step

		#Check if converged. If not keep running. 
		print(N)
		for i in (range(N)):
			if i > 500: step = self.update_cov(mcmc_chains[:i], sd, len(theta))
			theta_new = list(np.random.multivariate_normal(theta, step))
			#theta_new[3] = 0.5			
			#theta_new[4] = 3
			#theta_new[9] = 0.1			
			#theta_new[10] = 0.55
			temp_out, heatflux_mixed_out, heatflux_interior_out, gmsl_out = \
			doeclim_gmsl(asc = theta[7], t2co_in = theta[5], kappa_in=theta[8], alphasl_in = theta[0], Teq = theta[1], SL0 = theta[2]) 
			temp_out += theta[8]
			lp_new = self.logp(theta_new, deltat, temp_out, gmsl_out)
			if np.isinf(lp_new):
				mcmc_chains[i,:] = theta
				continue
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
			mcmc_chains[i,:] = theta

		return mcmc_chains,accepts/N*100

#values = {'alpha': 3.4, 'Teq': -0.5, 'S0': -100, 'rho': .5, \
#'sigma_ar': 3, 'climate_sensitivity': 3.1, 'ocean_vertical_diffusivity': 3.5, \
#'aerosol_scaling': 1.1, 'T_0': -0.06, 'sigma_T': 0.1, 'rho_T': 0.55}

values = [3.4, -0.5, -100, .5, 3, 3.1, 3.5, 1.1, -0.06, 0.1, 0.55]
#values = [3.4, -0.5, -100, 3.1, 3.5, 1.1, -0.06]

CM = CoupledModel(values)

mcmc_chain,accept_rate = CM.chain(1, NUMBER)
print(accept_rate)
pamnames = ['alpha', 'Teq', 'S0', 'rho', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0', 'sigma_T', 'rho_T']
#pamnames = ['alpha', 'Teq', 'sigma_ar', 'climate_sensitivity', 'ocean_vertical_diffusivity', 'aerosol_scaling', 'T_0']

#print(mcmc_chain[:200,0])

if NUMBER >= 50000:
	temp_chain1 = mcmc_chain[30000:40000]
	temp_chain2 = mcmc_chain[40000:]
	conv = CM.diagnostic([temp_chain1, temp_chain2])
	mcmc_chain = mcmc_chain[40000:]

print(conv)

for i in range(11):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.plot(mcmc_chain[: ,i])
	ax.set_title(pamnames[i])
	fig.savefig('image/plot'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)

for i in range(11):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.hist(mcmc_chain[: ,i], density = True, facecolor='green', alpha=0.5, bins = 20, edgecolor = 'white')
	ax.set_title(pamnames[i])
	fig.savefig('image/hist'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)
