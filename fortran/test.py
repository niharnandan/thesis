from ctypes import CDLL, POINTER, c_int, c_float, c_void_p, c_double
import numpy as np
import time
import forcing_total
import pandas as pd
import numpy as np

fortran = CDLL('./doeclim_gmsl.so')
fortran.run_doeclim_gmsl.argtypes = [
POINTER(c_int),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double),
POINTER(c_double)
]

forcing = pd.read_csv('../data/forcing_hindcast.csv')
asc = 1.1
mod_time = np.array(forcing['year'])
forcingtotal = np.array(forcing_total.forcing_total(forcing=forcing, alpha_doeclim=asc, l_project = False, begyear = mod_time[0], endyear = np.max(mod_time)))
n = len(mod_time)

mod_time = mod_time.astype(c_double)
forcingtotal = forcingtotal.astype(c_double)
t2co_in = 3.1
kappa_in = 3.5
alphasl_in = 3.4
Teq = -0.5
SL0 = 0
temp_out = np.empty((1,n), dtype = c_double)
heatflux_mixed_out = np.empty((1,n), dtype = c_double)
heatflux_interior_out = np.empty((1,n), dtype = c_double)
gmsl_out = np.empty((1,n), dtype = c_double)

fortran.run_doeclim_gmsl(
c_int(n),
mod_time.ctypes.data_as(POINTER(c_double)),
forcingtotal.ctypes.data_as(POINTER(c_double)),
c_double(t2co_in),
c_double(kappa_in),
c_double(alphasl_in),
c_double(Teq),
c_double(SL0),
temp_out.ctypes.data_as(POINTER(c_double)),
heatflux_mixed_out.ctypes.data_as(POINTER(c_double)),
heatflux_interior_out.ctypes.data_as(POINTER(c_double)),
gmsl_out.ctypes.data_as(POINTER(c_double))
)

print(temp_out, heatflux_mixed_out)
print("SUCCESS!")
