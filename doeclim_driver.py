import pandas as pd
import numpy as np
import os

import forcing_total
import doeclimF

if not os.path.exists('data/'):
    print("FATAL ERROR: No data directory")
    exit()

if not os.path.exists('output/'):
    os.makedirs('output/')

forcing = pd.read_csv( 'data/forcing_hindcast.csv')
mod_time = forcing['year']

climate_sensitivity = 3.1
ocean_vertical_diffusivity = 3.5
aerosol_scaling = 1.1

forcingtotal = forcing_total.forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))

doeclim_out = doeclimF.doeclimF(forcingtotal, mod_time, S=climate_sensitivity, kappa=ocean_vertical_diffusivity)

doeclim_out.to_csv('output/doeclim_output.csv')