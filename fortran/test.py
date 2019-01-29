import forcing_total
import pandas as pd
import numpy as np
import thing

forcing = pd.read_csv( '../data/forcing_hindcast.csv')
asc = 1.1
mod_time = forcing['year']

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=asc, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))

n = len(mod_time)

thing.run_doeclim_gmsl(ns = n,
forcing_in = forcingtotal,
t2co_in = 3.1, 
kappa_in = 3.5,
alphasl_in = 3.4,
teq_in = -0.5,
sl0_in = 0)
