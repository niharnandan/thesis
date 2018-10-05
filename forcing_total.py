import numpy as np

def forcing_total (forcing, alpha_doeclim, l_project, begyear, endyear,flnd = 0.29):
	
	if(not l_project):

		## Hindcasts
		forcing_land = forcing['co2'] + forcing['nonco2.land'] + alpha_doeclim*forcing['aerosol.land'] + forcing['solar.land'] + forcing['volc.land']
		forcing_ocean = forcing['co2'] + forcing['nonco2.ocean'] + alpha_doeclim*forcing['aerosol.ocean'] + forcing['solar.ocean'] + forcing['volc.ocean']
		forcing_total = flnd*forcing_land + (1-flnd)*forcing_ocean
	else:
		## Projections
		forcing_total = forcing['co2'] + forcing['nonco2'] + alpha_doeclim*forcing['aerosol.direct'] + alpha_doeclim*forcing['aerosol_indirect'] +forcing['solar'] + forcing['volcanic'] + forcing['other']


	## Clip forcing at the beginning and end of the model simulation
	ibeg = -1
	iend = -1
	ibeg=forcing.index[forcing['year'] == begyear][0]
	iend=forcing.index[forcing['year'] == endyear][0]
	
	if(ibeg == iend):
		print("ERROR - begyear/endyear not within forcing data")
	forcing_total = forcing_total[ibeg:iend+1]

	return(forcing_total)