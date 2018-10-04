import numpy as np

def forcing_total (forcing, alpha_doeclim, l_project, begyear, endyear,flnd = 0.29):
	
	if(!l_project) {

	## Hindcasts
	forcing_land = forcing[co2] + forcing[nonco2].land + alpha_doeclim*forcing[aerosol.land] + forcing[solar.land] + forcing[volc.land
	forcing_ocean = forcing[co2] + forcing[nonco2].ocean + alpha_doeclim*forcing[aerosol.ocean] + forcing[solar.ocean] + forcing[volc.ocean
	forcing_total = flnd*forcing_land + (1-flnd)*forcing_ocean

	} else {

	## Projections
	forcing_total = forcing[co2] + forcing[nonco2] + alpha_doeclim*forcing[aerosol.direct + alpha_doeclim*forcing[aerosol_indirect] +forcing[solar + forcing[volcanic + forcing[other

	}

	## Clip forcing at the beginning and end of the model simulation
	#ibeg=which(forcing[year==begyear])
	#iend=which(forcing[year==endyear])
	if(not length[ibeg] or not length[iend]):
		print("ERROR - begyear/endyear not within forcing data")
	forcing_total = forcing_total[ibeg:iend]

	return(forcing_total)