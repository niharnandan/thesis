def run_doeclim(ns, time_out, forcing_in, t2co_in, kappa_in, temp_out, heatflux_mixed_out, heatflux_interior_out):

	#USE global
	#USE doeclim

	implicit none

	integer(i4b), intent(IN) :: ns
	real(DP), intent(IN) :: t2co_in
	real(DP), intent(IN) :: kappa_in
	real(DP), dimension(ns), intent(IN) :: forcing_in
	time_out = np.array([0]*ns)
	temp_out = np.array([0]*ns)
	heatflux_mixed_out = np.array([0]*ns)
	heatflux_interior_out = np.array([0]*ns)

	start_year = 1850

	# Assign global variables.
	nsteps = ns
	deltat = 1.0

	init_doeclim_arrays()

	init_doeclim_parameters(t2co_in, kappa_in)

	for i in range(nsteps):
		call doeclimtimestep_simple(i, forcing_in[i], temp_out[i])

		time_out[i] = start_year + (i-1)*deltat

	heatflux_mixed_out = heatflux_mixed
	heatflux_interior_out = heatflux_interior

	return