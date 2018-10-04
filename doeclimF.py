import numpuy as np

def flux_to_heat(heatflux_mixed, heatflux_interior):
	flnd = 0.29 # area land fraction
	fso = 0.95 # ocean area fraction of interior
	secs_per_year = 31556926
	earth_area = 510065600 * 10**6
	ocean_area = (1-flnd)*earth_area
	powtoheat = ocean_area*secs_per_year / 10^22 # in 10^22 J/yr

	heat_mixed = np.sum(heatflux_mixed) * powtoheat
	heat_interior = fso * np.sum(heatflux_interior) * powtoheat
	ocean_heat = heat_mixed + heat_interior

	return(list(ocean_heat, heat_mixed, heat_interior))

## load DOECLIM model shared library
dyn.load("../fortran/doeclim.so")

def doeclimF (forcing_total,mod_time, S=3.1, kappa = 3.5):
	n = length(mod_time)

	# call Fortran DOECLIM
	# doeclim.so must be already dynamically loaded (see above this function)
	fout = .Fortran( "run_doeclim",
			ns = n,
			time_out = as.double(mod_time),
			forcing_in = as.double(forcing_total),
			t2co_in = as.double(S),
			kappa_in = as.double(kappa),
			temp_out = as.double(rep(0,n)),
			heatflux_mixed_out = as.double(rep(0,n)),
			heatflux_interior_out = as.double(rep(0,n))
		)

	ocheat = flux.to.heat(fout$heatflux_mixed, fout$heatflux_interior)

	model_output = list(time=mod_time, temp=fout$temp_out, ocheat=ocheat$ocean.heat,
											ocheat.mixed=ocheat$heat.mixed, ocheat.interior=ocheat$heat.interior,
											ocheatflux.mixed = fout$heatflux_mixed, ocheatflux.interior = fout$heatflux_interior)

	return(model_output)