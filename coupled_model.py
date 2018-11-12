import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import stats
import matplotlib.pyplot as plt
import gmsl_model
import os
import forcing_total
import doeclimF
from tqdm import tqdm

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

dfSealevel = pd.read_csv('https://media.githubusercontent.com/media/MUSSLES/tutorials/master/data/GMSL_ChurchWhite2011_yr_2015.csv')
dfTemperature = pd.read_csv('https://media.githubusercontent.com/media/MUSSLES/tutorials/master/data/NOAA_IPCC_RCPtempsscenarios.csv')
sealevel = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "sealevel"]).tolist()
year = (dfSealevel.loc[(dfSealevel["year"] >= 1880) & (dfSealevel["year"]<=2009), "year"]).tolist()
sealevel_sigma = (dfSealevel.loc[(dfSealevel["year"]>=1880) & (dfSealevel["year"]<=2009), "uncertainty"]).tolist()
#temperatures = dfTemperature.loc[(dfTemperature["Time"] <= 2009) & (dfTemperature["Time"] >= 1880), "Historical NOAA temp & CNRM RCP 8.5 with respect to 20th century"]
forcing = pd.read_csv( 'data/forcing_hindcast.csv')
mod_time = forcing['year']

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=climate_sensitivity, kappa=ocean_vertical_diffusivity)
temperatures = (doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist()
	
def update_cov(X, s_d):
    cov = np.cov([X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7],X[:,8]])
    eps = 0.0001
    I_d = np.identity(9)
    return s_d*cov + I_d*eps*s_d
#Online update 

def prior(theta, sealevel_0, unc_0):
    log_prior = 0
    #unpacking the variables individually for clarity
    alpha, Teq, S0, rho, sigma_ar = (
        theta[0], theta[1], theta[2], theta[3], theta[4])
    log_prior += stats.uniform.logpdf(alpha, loc = 0, scale = 5) #lb and ub?
    log_prior += stats.uniform.logpdf(Teq, loc=-1, scale = 2)
    log_prior += stats.norm.logpdf(S0, loc = sealevel_0, scale = unc_0)
    log_prior += stats.uniform.logpdf(rho, loc = 0, scale = 1)
    log_prior += stats.uniform.logpdf(sigma_ar, loc = 0, scale = 5)
    log_prior += stats.uniform.logpdf(theta[5], loc = 0.1, scale = 9.9)
    log_prior += stats.uniform.logpdf(theta[6], loc = 0.1, scale = 3.9)
    log_prior += stats.uniform.logpdf(theta[7], loc = 0, scale = 2)
    log_prior += stats.uniform.logpdf(theta[8], loc = -0.3, scale = 0.6)

    return log_prior

def logp(theta, sealevel, deltat, temperatures, model, sigma=sealevel_sigma):
    N = len(sealevel)
    alpha, Teq, S0, rho, sigma_ar = (
    theta[0], theta[1], theta[2], theta[3], theta[4])
    resid = np.array([sealevel[i] - model[i] for i in range(len(model))])
    sigma_obs = np.diag([i**2 for i in sigma])
    sigma_ar1 = build_ar1(rho, sigma_ar, N)
    log_prior = prior(theta, sealevel[0], sealevel_sigma[0])
    if np.isinf(log_prior): return -np.inf
    cov = np.add(sigma_obs,sigma_ar1)
    cov = np.multiply((np.transpose(cov) + cov), 1/2)
    log_likelihood = stats.multivariate_normal.logpdf(resid, cov=cov)
    log_posterior = log_likelihood + log_prior
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

def coupled_model(forcingtotal, doeclim_out, temperatures, model, parameters, mod_time, T_0):
    forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=parameters[7], l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
    doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=parameters[5], kappa=parameters[6])
    temperatures = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
    temperatures = temperatures + T_0
    model = gmsl_model.gmsl_model(parameters, temperatures, deltat)

def chain(parameters, temperatures, deltat, sealevel, sealevel_sigma, N=10000):
    
    forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
    doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=climate_sensitivity, kappa=ocean_vertical_diffusivity)
    temperatures = np.array((doeclim_out.loc[(doeclim_out["time"]>=1880) & (doeclim_out["time"]<=2008), "temp"]).tolist())
    temperatures = temperatures + T_0
    alpha, Teq, S0 = parameters[0], parameters[1], parameters[2]
    theta = parameters
    print('Initial estimate for parameters -', theta)
    
    model = gmsl_model.gmsl_model(theta, temperatures, deltat)

    lp = logp(theta, sealevel, deltat, temperatures, model, sigma=sealevel_sigma)
    theta_best = theta
    lp_max = lp
    theta_new = [0.] * 9
    accepts = 0
    mcmc_chains = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]] * N)
    step = np.array([[.01,0,0,0,0,0,0,0,0], [0,.05,0,0,0,0,0,0,0], [0,0,5,0,0,0,0,0,0], [0,0,0,0.001,0,0,0,0,0],[0,0,0,0,0.1,0,0,0,0],
					 [0,0,0,0,0,0.16,0,0,0], [0,0,0,0,0,0,0.17,0,0], [0,0,0,0,0,0,0,0.025,0],[0,0,0,0,0,0,0,0,0.03]])
    sd = 2.38**2 / len(theta)
    initial_burn = 20000
	#Check if converged. If not keep running. 
    print(N)
    for i in range(N):
        if i > 500: step = update_cov(mcmc_chains[:i], sd)
        theta_new = list(np.random.multivariate_normal(theta, step))
        coupled_model(forcingtotal, doeclim_out, temperatures, model, parameters, mod_time, T_0)
		#Separate above 3 lines + T_0 into a new file. 
        lp_new = logp(theta_new, sealevel, deltat, temperatures, model, sigma=sealevel_sigma)
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

parameters = [3.4, -0.5, sealevel[0], 0.5, 3, climate_sensitivity, ocean_vertical_diffusivity, aerosol_scaling, T_0]
deltat = 1
mcmc_chain,accept_rate = chain(parameters, temperatures, deltat, sealevel, sealevel_sigma, N=1000)

for i in range(9):
	fig, ax = plt.subplots(nrows=1, ncols=1 )  # create figure & 1 axis
	ax.plot(mcmc_chain[:,i])
	fig.savefig('image/plot'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig)
	
'''TO-DO
	Switch for parameters
	Make histogram for each parameter (USE SUPERCOMPUTER)
	Compare to paper sent from Tony
	Use PYMC3 package.
	Multiprocessing for Python 
'''