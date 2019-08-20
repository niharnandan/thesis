For correlated model - 
1. Run "python coupled_model.py 'iterations'" where 'iterations' are how long the chain should run for
2. Before plotting the results, open plot.py, and change the slice of the array at line 53 to plot the necessary values.
3. To plot the histogram and history plots, run "python plot.py"

For uncorrelated model - 
1. Run "python coupled_model_1.py 'iterations'" where 'iterations' are how long the chain should run for
2. Before plotting the results, open plot.py, and change the slice of the array at line 61 to plot the necessary values.
3. Uncomment line 61
4. To plot the histogram and history plots, run "python plot.py"

The plot program will exit midway without plotting the ocean_vertical_diffusivity for comparison.

