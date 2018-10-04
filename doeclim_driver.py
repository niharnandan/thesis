import pandas as pd
import numpy as np
import os

#import 'doeclimF.py'
#import 'forcing_total.py'

forcing = pd.read_csv( 'data/forcing_hindcast.csv')
mod_time = forcing['year']

climate_sensitivity = 3.1
ocean_vertical_diffusivity = 3.5
aerosol_scaling = 1.1

if not os.path.exists('output/'):
    os.makedirs('output/')

forcingtotal = forcing_total(forcing=forcing, alpha_doeclim=aerosol_scaling, l_project=False, begyear=mod_time[0], endyear=np.max(mod_time))
							   
doeclim_out = doeclimF(S=climate_sensitivity, kappa=ocean_vertical_diffusivity, forcingtotal, mod_time)

doeclim_out.to_csv('output/doeclim_output.csv')