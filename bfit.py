import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad
from scipy.stats import rv_continuous

class FitModels:
    """Contains various probability distribution functions and background models."""
    
    def __init__(self, bin_width=1, x_min=0, x_max=1):
        self.bin_width = bin_width
        self.x_min = x_min
        self.x_max = x_max

    def gaussian(self, x, mean, std_dev):
        """Gaussian distribution."""
        return np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / np.sqrt(2 * np.pi * std_dev ** 2)

    def exponential(self, x, amplitude, decay):
        """Exponential distribution."""
        return amplitude * np.exp(-decay * (x - self.x_min))

    def linear(self, x, slope, intercept):
        """Linear function."""
        return slope * x + intercept

    def parabola(self, x, a, b, c):
        """Parabolic function."""
        return a * (x ** 2) + b * x + c

    def breit_wigner(self, x, mass, width):
        """Breit-Wigner distribution."""
        return (1 / np.pi) * (0.5 * width) / ((x - mass) ** 2 + (0.5 * width) ** 2)

    def crystal_ball(self, x, alpha, n, mean, std_dev):
        """Crystal Ball function."""
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mean) / std_dev > -alpha,
                        np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)),
                        A * (B - (x - mean) / std_dev) ** (-n))

    def argus_bg(self, x, m0, c, p):
        """ARGUS background function."""
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, x * np.sqrt(z) * np.exp(c * z ** p), 0)

class Component(rv_continuous):
    """Base class for fit components."""
    def __init__(self, function, pdf, n_params):
        super().__init__()
        self.function = function
        self._pdf = pdf
        self.n_params = n_params

    def __call__(self, x, *params):
        return self.function(x, *params)

    def pdf(self, x, *params):
        return self._pdf(x, *params)

class FitBase:
    """Base class for all binned fit models."""

    def __init__(self, bins, counts, param_limits=None):
        self.bins = bins
        self.x_min, self.x_max, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self._setup_fit(counts, bins)
        self.param_limits = param_limits or {}
        self.fit_models = FitModels(self.bin_width, self.x_min, self.x_max)
        self.fit_result = None

    def _setup_fit(self, counts, bins):
        x_min, x_max = bins[0], bins[-1]
        bin_width = bins[1] - bins[0]
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        y_vals, y_errs = counts, np.sqrt(counts)
        return x_min, x_max, bin_width, x_vals, y_vals, y_errs

    def chi_squared(self, *params):
        """Calculate chi-squared value."""
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        residuals_squared = ((y_vals_masked - prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

    def fit(self, initial_params, param_names=None):
        """Perform the fit."""
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.chi_squared, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, title='Fit plot', xlabel='X', ylabel='Y', vlines=None, plot_range=None,
             data_color='b', data_label='Data', fit_color='r', fit_label='Fit', show_plot=True):
        """Plot the fit results."""
        fig, ax = plt.subplots()
        ax.set_title(title)

        visual_bins = self.bins
        if plot_range is not None:
            visual_bins = visual_bins[(visual_bins >= plot_range[0]) & (visual_bins <= plot_range[1] + self.bin_width)]
            ax.set_xlim(plot_range)

        ax.hist(self.x_vals, bins=visual_bins, weights=self.y_vals, color=data_color, label=data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.fit_function(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=fit_color, label=fit_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15 * max(np.append(self.y_vals, predictions)))
        ax.grid(True)
        
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6 * max(self.y_vals), colors='yellow')
        
        ax.legend()

        if show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """Print fit summary."""
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """Get fit parameters."""
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """Get fit parameter errors."""
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def chi_squared_value(self):
        """Get chi-squared value of the fit."""
        if self.fit_result:
            return self.fit_result.fval
        else:
            raise RuntimeError("Fit has not been performed yet.")

class CompositeModel(FitBase):
    """Composite model class for combining multiple fit components."""

    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """Add a component to the composite model."""
        self.components.append((component, weight))

    def fit_function(self, x_vals, *params):
        """Composite fit function."""
        result = np.zeros_like(x_vals)
        param_index = 0
        total_weight = 0
        for component, weight in self.components:
            n_params = component.n_params
            result += weight * component(x_vals, *params[param_index:param_index+n_params])
            param_index += n_params
            total_weight += weight
        
        # Normalize the fit function
        return result / total_weight

    def chi_squared(self, *params):
        """Calculate chi-squared value."""
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        
        # Scale the prediction to match the total number of events
        scale_factor = np.sum(y_vals_masked) / np.sum(prediction)
        scaled_prediction = prediction * scale_factor
        
        residuals_squared = ((y_vals_masked - scaled_prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

class UnbinnedFitBase:
    """Base class for all unbinned fit models."""

    def __init__(self, data, param_limits=None):
        self.data = data
        self.x_min, self.x_max = self._setup_fit(data)
        self.param_limits = param_limits or {}
        self.fit_models = FitModels(x_min=self.x_min, x_max=self.x_max)
        self.fit_result = None

    def _setup_fit(self, data):
        return np.min(data), np.max(data)

    def log_likelihood(self, *params):
        """Calculate log-likelihood value."""
        raise NotImplementedError("Subclass must implement abstract method")

    def fit(self, initial_params, param_names=None):
        """Perform the fit."""
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.log_likelihood, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, title='Fit plot', xlabel='X', ylabel='Y', vlines=None, plot_range=None,
             data_color='b', data_label='Data', fit_color='r', fit_label='Fit', show_plot=True):
        """Plot the fit results."""
        fig, ax = plt.subplots()
        ax.set_title(title)

        if plot_range is not None:
            ax.set_xlim(plot_range)
        else:
            ax.set_xlim(self.x_min, self.x_max)

        ax.hist(self.data, bins=100, density=True, color=data_color, label=data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.pdf(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=fit_color, label=fit_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6 * ax.get_ylim()[1], colors='yellow')
        
        ax.legend()

        if show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """Print fit summary."""
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """Get fit parameters."""
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """Get fit parameter errors."""
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def log_likelihood_value(self):
        """Get log-likelihood value of the fit."""
        if self.fit_result:
            return -self.fit_result.fval  # Negative because we minimize -log(L)
        else:
            raise RuntimeError("Fit has not been performed yet.")

class UnbinnedCompositeModel(UnbinnedFitBase):
    """Composite model class for combining multiple unbinned fit components."""

    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """Add a component to the composite model."""
        self.components.append((component, weight))

    def pdf(self, x_vals, *params):
        """Composite probability density function."""
        result = np.zeros_like(x_vals)
        param_index = 0
        total_weight = 0
        for component, weight in self.components:
            n_params = component.n_params
            result += weight * component.pdf(x_vals, *params[param_index:param_index+n_params])
            param_index += n_params
            total_weight += weight
        
        # Normalize the PDF
        return result / total_weight

    def log_likelihood(self, *params):
        """Calculate log-likelihood value."""
        return -np.sum(np.log(self.pdf(self.data, *params)))

# Pre-defined components
class GaussianComponent(Component):
    def __init__(self):
        super().__init__(FitModels().gaussian, FitModels().gaussian, 2)  # mean, std_dev

class ExponentialComponent(Component):
    def __init__(self):
        super().__init__(FitModels().exponential, FitModels().exponential, 2)  # amplitude, decay

class LinearComponent(Component):
    def __init__(self):
        super().__init__(FitModels().linear, FitModels().linear, 2)  # slope, intercept

class ParabolaComponent(Component):
    def __init__(self):
        super().__init__(FitModels().parabola, FitModels().parabola, 3)  # a, b, c

class BreitWignerComponent(Component):
    def __init__(self):
        super().__init__(FitModels().breit_wigner, FitModels().breit_wigner, 2)  # mass, width

class CrystalBallComponent(Component):
    def __init__(self):
        super().__init__(FitModels().crystal_ball, FitModels().crystal_ball, 4)  # alpha, n, mean, std_dev

class ArgusComponent(Component):
    def __init__(self):
        super().__init__(FitModels().argus_bg, FitModels().argus_bg, 3)  # m0, c, p

# Binned fit models
class GaussianExpFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianExpFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianParabolaFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class DoubleGaussianLinearFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class GaussianArgusFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class GaussianLinearFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class BreitWignerExpFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class BreitWignerLinearFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class CrystalBallExpFit(CompositeModel):
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())

# Unbinned fit models
class UnbinnedGaussianExpFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianExpFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianParabolaFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class UnbinnedDoubleGaussianLinearFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedGaussianArgusFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class UnbinnedGaussianLinearFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedBreitWignerExpFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class UnbinnedBreitWignerLinearFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class UnbinnedCrystalBallExpFit(UnbinnedCompositeModel):
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())
