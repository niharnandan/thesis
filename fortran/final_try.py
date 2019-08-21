import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import stats
import matplotlib.pyplot as plt
import doeclimF
import forcing_total

dfSealevel = pd.read_csv('https://media.githubusercontent.com/media/MUSSLES/tutorials/master/data/GMSL_ChurchWhite2011_yr_2015.csv')
dfTemperature = pd.read_csv('https://media.githubusercontent.com/media/MUSSLES/tutorials/master/data/NOAA_IPCC_RCPtempsscenarios.csv')
dfOcean_heat = pd.read_csv('gouretski_ocean_heat_3000m.txt', sep= ' ')
offset = (1952-1880, 2009-1996)
ocean_heat = dfOcean_heat['heat-anomaly(10^22J)'].tolist()
ocean_sigma = dfOcean_heat['std.dev.(10^22J)'].tolist()
forcing = pd.read_csv( 'data/forcing_hindcast.csv')
year = dfSealevel["year"].tolist()
sealevel = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).tolist()
sealevel_sigma = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "uncertainty"]).tolist()
temperatures = dfTemperature.loc[dfTemperature["Time"] <= 2008, "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"].tolist()
temp = pd.read_csv("temp.txt", header=None, sep=',')
temp_unc = temp.drop(temp.columns[[1,3,4,5]], axis = 1)
temp_unc.columns = ["year", "uncertainty"]
temp_unc = (temp_unc.loc[(temp_unc["year"]>=1880) & (temp_unc["year"]<=2008), "uncertainty"]).tolist()
ocean_heat = dfOcean_heat['heat-anomaly(10^22J)'].tolist()
ocean_sigma = dfOcean_heat['std.dev.(10^22J)'].tolist()

temperatures = temperatures - np.mean(temperatures[81:111])
sealevel = sealevel - np.mean(sealevel[81:111])
forcing = pd.read_csv( 'data/forcing_hindcast.csv')

def generate_level(parameters, temperatures, deltat):
    alpha, Teq, S0 = parameters[0], parameters[1], parameters[2]
    S = [0]*(len(temperatures))
    S[0] = S0
    for i in range(1,len(temperatures)):
        S[i] = S[i-1] + deltat * alpha * (temperatures[i-1] - Teq)
    return S
	
def build_ar1(rho, sigma_ar, length):
    ar1 = []
    for i in range(length):
        temp1 = [rho**j for j in range(i,0,-1)]
        temp2 = [rho**j for j in range(length-i)]
        ar1.append(np.array(temp1+temp2))
    ar1 = np.multiply(np.array(ar1), (sigma_ar**2)/(1-rho**2))
    return ar1
	
def update_cov(X, s_d):
    cov = np.cov([X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7],X[:,8]
	,X[:,9],X[:,10],X[:,11],X[:,12]])
    eps = 0.0001
    I_d = np.identity(13)
    return s_d*cov + I_d*eps*s_d

def prior(theta):
	log_prior = 0
	#unpacking the variables individually for clarity
	alpha, Teq, S0, rho, sigma_ar = (
		theta[0], theta[1], theta[2], theta[3], theta[4])
	log_prior += stats.uniform.logpdf(alpha, loc = -2, scale = 7) #lb and ub?
	log_prior += stats.uniform.logpdf(Teq, loc=-1, scale = 2)
	log_prior += stats.uniform.logpdf(S0, loc = -175, scale = 50)
	log_prior += stats.uniform.logpdf(rho, loc = 0, scale = 2)
	log_prior += stats.uniform.logpdf(sigma_ar, loc = 0, scale = 7)
	log_prior +=  stats.uniform.logpdf(theta[5], loc = 0.1, scale = 9.9)
	log_prior +=  stats.uniform.logpdf(theta[6], loc = -2, scale = 6)
	log_prior += stats.uniform.logpdf(theta[7], loc = 0.1, scale = 2)
	log_prior +=  stats.uniform.logpdf(theta[8], loc = -0.3, scale = 0.6)
	log_prior +=  stats.uniform.logpdf(theta[9], loc = -2, scale = 7)
	log_prior +=  stats.uniform.logpdf(theta[10], loc =  0.1, scale = 2)
	log_prior +=  stats.uniform.logpdf(theta[11], loc = 0.05, scale = 4.95)
	log_prior +=  stats.uniform.logpdf(theta[12], loc =  0.1, scale = 0.999)
	return log_prior

def logp(theta, sealevel, ocean_heat, deltat, temperatures, forcing, temp_unc, ocean_sigma, sigma=sealevel_sigma):
	N = len(sealevel)
	alpha, Teq, S0, rho, sigma_ar = (
		theta[0], theta[1], theta[2], theta[3], theta[4])
		
	model = generate_level(theta, temperatures, deltat)
	forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=theta[7], l_project=False, begyear=1880, endyear=2008)
	doeclim_out = doeclimF.doeclimF(forcingtotal, list(range(1880,2009)), S=theta[5], kappa=theta[6])
	temp_out = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
	temp_out += theta[8]
	ocheat = np.array((doeclim_out.loc[(doeclim_out["time"]>=1953) & (doeclim_out["time"]<=1996), 'ocheat.mixed']).tolist())
	
	resid = np.array([sealevel[i] - model[i] for i in range(len(model))])
	o_residual = np.array([ocean_heat[i] - ocheat[i] for i in range(len(ocheat))])
	t_residual = np.array([temperatures[i] - temp_out[i] for i in range(len(temp_out))])
	
	sigma_obs = np.diag([i**2 for i in sigma])
	sigma_obs_T = np.diag([i**2 for i in temp_unc])
	sigma_obs_O = np.diag([i**2 for i in ocean_sigma])
	
	sigma_ar1 = build_ar1(rho, sigma_ar, N)
	sigma_arT = build_ar1(theta[10], theta[9], N)
	sigma_arO = build_ar1(theta[12], theta[11], 44)
	log_prior = prior(theta)
	if np.isinf(log_prior): return -np.inf
	
	cov = np.add(sigma_obs,sigma_ar1)
	min_eig = np.min(np.real(np.linalg.eigvals(cov)))
	if min_eig < 0:	cov -= 10*min_eig * np.eye(*cov.shape)

	covT = np.add(sigma_obs_T,sigma_arT)
	min_eig = np.min(np.real(np.linalg.eigvals(covT)))
	if min_eig < 0:	covT -= 10*min_eig * np.eye(*covT.shape)
	
	covO = np.add(sigma_obs_O,sigma_arO)
	min_eig = np.min(np.real(np.linalg.eigvals(covO)))
	if min_eig < 0:	covO -= 10*min_eig * np.eye(*covO.shape)
	
	
	log_likelihood = stats.multivariate_normal.logpdf(resid, cov=cov)
	log_likelihoodT = stats.multivariate_normal.logpdf(t_residual, cov=covT)
	log_likelihoodO = stats.multivariate_normal.logpdf(o_residual, cov=covO)
	log_posterior = log_likelihood + log_likelihoodO +  log_likelihoodT + log_prior
	return log_posterior

def chain(parameters, ocean_heat, temperatures, deltat, sealevel, temp_unc, ocean_sigma, sealevel_sigma, forcing, N=20000):
	theta = parameters
	print('Initial estimate for parameters -', theta)

	lp = logp(theta, sealevel, ocean_heat, deltat, temperatures, forcing, temp_unc, ocean_sigma, sigma=sealevel_sigma)
	theta_best = theta
	lp_max = lp
	theta_new = [0] * 13
	accepts = 0
	mcmc_chains = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]] * N)    
	#step = np.array([[.01,0,0,0,0], [0,.05,0,0,0], [0,0,5,0,0],[0,0,0,0.001,0],[0,0,0,0,0.1]])
	stepsizes = [ .01,  .05,  5,  .001, .1,  0.16,  0.17,  0.025,0.03, 5e-4, 0.007, 5e-4, 0.007]
	step = np.diag(stepsizes)
	
	sd = 2.38**2 / len(theta)
	print(N)
	for i in range(N):
		if i > 1 and not i %2000: pd.DataFrame(mcmc_chains).to_csv('array_uncorr.csv')
		if i > 500: step = update_cov(mcmc_chains[:i], sd)
		if not i%1000: print(i)
		theta_new = list(np.random.multivariate_normal(theta, step))
		lp_new = logp(theta_new, sealevel, ocean_heat, deltat, temperatures, forcing, temp_unc, ocean_sigma, sigma=sealevel_sigma)
		lq = lp_new - lp
		lr = np.math.log(np.random.uniform(0, 1))
		if (lr < lq):
			theta = theta_new
			lp = lp_new
			accepts += 1
			if lp > lp_max:
				theta_best = theta
				lp_max = lp
		mcmc_chains[i] = theta
	return mcmc_chains,accepts/N*100

values = [3.4, -0.5, -125, .5, 3, 3.1, 3.5, 1.1, -0.06, 3, .5, 3, 0.5]
deltat = 1
mcmc_chain,accept_rate = chain(values, ocean_heat, temperatures, deltat, sealevel, temp_unc, ocean_sigma, sealevel_sigma, forcing, 50000)

pd.DataFrame(mcmc_chain).to_csv('array_uncorr.csv')