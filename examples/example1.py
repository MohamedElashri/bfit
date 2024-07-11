############################################### Example 1 ###############################################
################## Fit a Gaussian signal with Gaussian noise using a custom fit model. ##################
############################################### Example 1 ###############################################

import numpy as np
from bfit import GaussianComponent, FitBase, PlotConfig

# Generate example data
np.random.seed(42)
bins = np.linspace(-5, 5, 100)
x = bins[:-1]

# Gaussian signal
gaussian_amplitude = 1000
gaussian_mean = 0
gaussian_std = 1
gaussian_signal = gaussian_amplitude * np.exp(-0.5 * ((x - gaussian_mean) / gaussian_std)**2)

# Add Gaussian noise to the signal
noise_std = 50
counts = gaussian_signal + np.random.normal(0, noise_std, size=gaussian_signal.shape)
counts = np.maximum(0, counts)  # Ensure non-negative counts

# Create a custom fit model
class GaussianFit(FitBase):
    def __init__(self, bins, counts):
        super().__init__(bins, counts)
        self.add_component(GaussianComponent())

    def fit_function(self, x, amplitude, mean, std_dev):
        return amplitude * self.components[0][0](x, mean, std_dev)

# Create an instance of the custom fit model
model = GaussianFit(bins, counts)

# Set initial parameter values and limits
initial_params = [800, 0, 1.5]
param_limits = {
    'amplitude': (0, None),  # Gaussian amplitude
    'mean': (-10, 10),       # Gaussian mean
    'std_dev': (0, None),    # Gaussian standard deviation
}

# Perform the fit
param_names = ['amplitude', 'mean', 'std_dev']
fit_result = model.fit(initial_params, param_names=param_names, param_limits=param_limits)

# Print the fit summary
model.summary()

# Plot the fit results
plot_config = PlotConfig(
    title='Gaussian Fit',
    xlabel='X',
    ylabel='Counts',
    data_label='Data',
    fit_label='Fit'
)
model.plot(config=plot_config)