import doeclim
import numpy as np

def run_doeclim(ns, time_out, forcing_in, t2co_in, kappa_in, temp_out, heatflux_mixed_out, heatflux_interior_out):

	#USE global
	#USE doeclim

	#implicit none

	time_out = np.array([0]*ns)

	start_year = 1850

	# Assign global variables.
	deltat = 1.0
	temp = doeclim.doeclim(deltat = deltat, nsteps = ns)
	#init_doeclim_arrays()
	temp.init_doeclim_parameters(t2co_in, kappa_in)
	#print(vars(temp))
	for i in range(ns):
		x = [0.0]
		temp.doeclimtimestep_simple(i, forcing_in[i], x)
		temp_out[i] = x[0]
		time_out[i] = start_year + (i-1)*deltat
	
	heatflux_mixed_out = temp.heatflux_mixed
	heatflux_interior_out = temp.heatflux_interior
	return (heatflux_mixed_out,heatflux_interior_out)