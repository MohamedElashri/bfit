# BFit: B-Physics Fitting Library

BFit is a Python library designed for fitting various probability distribution functions commonly used in B-physics analyses. It provides a flexible and extensible framework for fitting histogrammed data with a variety of signal and background models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Usage Guide](#usage-guide)
   - [Creating a Fit Object](#creating-a-fit-object)
   - [Performing a Fit](#performing-a-fit)
   - [Visualizing Results](#visualizing-results)
   - [Accessing Fit Results](#accessing-fit-results)
4. [Available Fit Models](#available-fit-models)
5. [API Reference](#api-reference)
   - [FitBase Class](#fitbase-class)
   - [Derived Fit Classes](#derived-fit-classes)
   - [FitModels Class](#fitmodels-class)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/MohamedElashri/bfit.git
cd bfit
pip install -r requirements.txt
```

## Quick Start

Here's a basic example to get you started:

```python
import numpy as np
from bfit import GaussianExpFit

# Generate example data
x = np.random.normal(5, 1, 1000)
bins = np.linspace(0, 10, 100)
hist, bin_edges = np.histogram(x, bins=bins)

# Create fit object and perform fit
fit = GaussianExpFit(bins, hist)
result = fit.fit([800, 200, 5, 1, 1, 0.5], 
                 param_names=['n_signal', 'n_bkg', 'mean', 'std_dev', 'amplitude', 'decay'])

# Plot and summarize results
fit.plot(title='Gaussian + Exponential Fit', xlabel='X', ylabel='Counts')
fit.summary()
```

## Usage Guide

### Creating a Fit Object

To create a fit object, choose the appropriate model class and initialize it with your histogram data:

```python
from bfit import GaussianExpFit

fit = GaussianExpFit(bins, hist_counts, param_limits={'mean': (4, 6)})
```

The `param_limits` parameter is optional and allows you to set limits on fit parameters.

### Performing a Fit

Use the `fit` method to perform the fit:

```python
result = fit.fit(initial_params, param_names=['n_signal', 'n_bkg', 'mean', 'std_dev', 'amplitude', 'decay'])
```

`initial_params` is a list of initial parameter values. `param_names` is an optional list of parameter names.

### Visualizing Results

Plot the fit results:

```python
fit.plot(title='My Fit', xlabel='Mass (GeV)', ylabel='Events / 50 MeV',
         data_color='black', fit_color='red', show_plot=True)
```

### Accessing Fit Results

Access fit parameters, errors, and chi-squared value:

```python
params = fit.fit_params
errors = fit.fit_errors
chi_sq = fit.chi_squared_value
```

## Available Fit Models

- `GaussianExpFit`
- `DoubleGaussianExpFit`
- `DoubleGaussianParabolaFit`
- `DoubleGaussianLinearFit`
- `GaussianArgusFit`
- `GaussianLinearFit`
- `BreitWignerExpFit`
- `BreitWignerLinearFit`
- `CrystalBallExpFit`

## API Reference

### FitBase Class

Base class for all fit models.

Methods:
- `__init__(self, bins, counts, param_limits=None)`
- `fit(self, initial_params, param_names=None)`
- `plot(self, title='Fit plot', xlabel='X', ylabel='Y', ...)`
- `summary(self)`

Properties:
- `fit_params`
- `fit_errors`
- `chi_squared_value`

### Derived Fit Classes

All derived classes (e.g., `GaussianExpFit`) inherit from `FitBase` and implement their specific `fit_function`.

### FitModels Class

Contains implementations of various probability distribution functions and background models.

## Examples

### Fitting a Double Gaussian with Linear Background

```python
from bfit import DoubleGaussianLinearFit
import numpy as np

# Generate example data
x = np.concatenate([np.random.normal(5, 0.5, 800), np.random.normal(5.5, 1, 200)])
x = np.concatenate([x, np.random.uniform(0, 10, 500)])  # Add uniform background
bins = np.linspace(0, 10, 100)
hist, bin_edges = np.histogram(x, bins=bins)

# Create fit object and perform fit
fit = DoubleGaussianLinearFit(bins, hist)
result = fit.fit([800, 0.7, 500, 5, 5.5, 0.5, 1, 1, 10],
                 param_names=['n_signal', 'frac', 'n_bkg', 'mean1', 'mean2', 'std_dev1', 'std_dev2', 'slope', 'intercept'])

# Plot and summarize results
fit.plot(title='Double Gaussian + Linear Fit', xlabel='X', ylabel='Counts')
fit.summary()
```

### Fitting with Parameter Limits

```python
from bfit import GaussianExpFit

fit = GaussianExpFit(bins, hist, param_limits={'mean': (4, 6), 'std_dev': (0, 2)})
result = fit.fit([800, 200, 5, 1, 1, 0.5])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 