import numpy as np
import run_doeclim
import pandas as pd

def flux_to_heat(heatflux_mixed, heatflux_interior):
	flnd = 0.29 # area land fraction
	fso = 0.95 # ocean area fraction of interior
	secs_per_year = 31556926
	earth_area = 510065600 * (10**6)
	ocean_area = (1-flnd)*earth_area
	powtoheat = ocean_area*secs_per_year / 10**22 # in 10^22 J/yr

	heat_mixed = np.cumsum(heatflux_mixed) * powtoheat
	heat_interior = fso * np.cumsum(heatflux_interior) * powtoheat
	ocean_heat = heat_mixed + heat_interior
	return([ocean_heat, heat_mixed, heat_interior])

## load DOECLIM model shared library
#dyn.load("../fortran/doeclim.so")

def doeclimF (forcing_total,mod_time, S=3.1, kappa = 3.5):
	n = len(mod_time)

	# call Fortran DOECLIM
	# doeclim.so must be already dynamically loaded (see above this function)
	temp_out = np.array([0.0]*n)
	fout =  run_doeclim.run_doeclim(
			ns = n,
			time_out = mod_time,
			forcing_in = forcing_total,
			t2co_in = S,
			kappa_in = kappa,
			temp_out = temp_out,
			heatflux_mixed_out = np.array([0]*n),
			heatflux_interior_out = np.array([0]*n)
		)

	ocheat = flux_to_heat(fout[0], fout[1])
	model = {'time': mod_time, 'temp': temp_out, 'ocheat': ocheat[0],'ocheat.mixed': ocheat[1], \
	'ocheat.interior':ocheat[2], 'ocheatflux.mixed' : fout[0], 'ocheatflux.interior' : fout[1]}

	model_output = pd.DataFrame(data=model)
	#model_output = list(time=mod_time, temp=fout$temp_out, ocheat=ocheat$ocean.heat,
	#										ocheat.mixed=ocheat$heat.mixed, ocheat.interior=ocheat$heat.interior,
	#										ocheatflux.mixed = fout$heatflux_mixed, ocheatflux.interior = fout$heatflux_interior)

	return(model_output)