import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad

class FitModels:
    """Contains various probability distribution functions and background models."""
    
    def __init__(self, bin_width, x_min, x_max):
        self.bin_width = bin_width
        self.x_min = x_min
        self.x_max = x_max

    def gaussian(self, x, mean, std_dev):
        """Gaussian distribution."""
        return self.bin_width * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / np.sqrt(2 * np.pi * std_dev ** 2)

    def exponential(self, x, amplitude, decay):
        """Exponential distribution."""
        integral = (amplitude / decay) * (1.0 - np.exp(-decay * (self.x_max - self.x_min)))
        norm = 1. / integral
        return self.bin_width * norm * amplitude * np.exp(-decay * (x - self.x_min))

    def double_gaussian_exp(self, x_vals, n_signal, frac, n_bkg, mean1, mean2, std_dev1, std_dev2, amplitude, decay):
        """Double Gaussian with exponential background."""
        n_signal1 = n_signal * frac
        n_signal2 = n_signal * (1 - frac)
        return (n_signal1 * self.gaussian(x_vals, mean1, std_dev1) +
                n_signal2 * self.gaussian(x_vals, mean2, std_dev2) +
                n_bkg * self.exponential(x_vals, amplitude, decay))

    def double_gaussian_parabola(self, x_vals, n_signal, frac, n_bkg, mean1, mean2, std_dev1, std_dev2, a, b, c):
        """Double Gaussian with parabolic background."""
        n_signal1 = n_signal * frac
        n_signal2 = n_signal * (1 - frac)
        return (n_signal1 * self.gaussian(x_vals, mean1, std_dev1) +
                n_signal2 * self.gaussian(x_vals, mean2, std_dev2) +
                n_bkg * self.parabola(x_vals, a, b, c))

    def parabola(self, x, a, b, c):
        """Parabolic function."""
        return self.bin_width * (a * (x ** 2) + b * x + c)

    def double_gaussian_linear(self, x_vals, n_signal, frac, n_bkg, mean1, mean2, std_dev1, std_dev2, slope, intercept):
        """Double Gaussian with linear background."""
        n_signal1 = n_signal * frac
        n_signal2 = n_signal * (1 - frac)
        return (n_signal1 * self.gaussian(x_vals, mean1, std_dev1) +
                n_signal2 * self.gaussian(x_vals, mean2, std_dev2) +
                n_bkg * self.linear(x_vals, slope, intercept))

    def linear(self, x, slope, intercept):
        """Linear function."""
        integral = slope * (self.x_max - self.x_min) + 0.5 * intercept * (self.x_max ** 2 - self.x_min ** 2)
        norm = 1. / integral
        return self.bin_width * norm * (slope * x + intercept)

    def argus_bg_integral(self, m0, c, p):
        """ARGUS background integral."""
        def integrand(x):
            z = 1 - (x / m0) ** 2
            return x * np.sqrt(z) * np.exp(c * z ** p) if z > 0 else 0
        integral, _ = quad(integrand, 0, m0, limit=10000)
        return integral

    def argus_bg(self, x, m0, c, p):
        """ARGUS background function."""
        normalization = self.argus_bg_integral(m0, c, p)
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, (x * np.sqrt(z) * np.exp(c * z ** p)) / normalization, 0)

    def gaussian_argus(self, x_vals, n_signal, n_bkg, mean, std_dev, m0, c, p):
        """Gaussian with ARGUS background."""
        return n_signal * self.gaussian(x_vals, mean, std_dev) + n_bkg * self.argus_bg(x_vals, m0, c, p)

    def gaussian_exp(self, x_vals, n_signal, n_bkg, mean, std_dev, amplitude, decay):
        """Gaussian with exponential background."""
        return n_signal * self.gaussian(x_vals, mean, std_dev) + n_bkg * self.exponential(x_vals, amplitude, decay)

    def gaussian_linear(self, x_vals, n_signal, n_bkg, mean, std_dev, slope, intercept):
        """Gaussian with linear background."""
        return n_signal * self.gaussian(x_vals, mean, std_dev) + n_bkg * self.linear(x_vals, slope, intercept)

    def breit_wigner(self, x, mass, width):
        """Breit-Wigner distribution."""
        return (1 / np.pi) * (0.5 * width) / ((x - mass) ** 2 + (0.5 * width) ** 2)

    def breit_wigner_exp(self, x_vals, n_signal, n_bkg, mass, width, amplitude, decay):
        """Breit-Wigner with exponential background."""
        return n_signal * self.breit_wigner(x_vals, mass, width) + n_bkg * self.exponential(x_vals, amplitude, decay)

    def breit_wigner_linear(self, x_vals, n_signal, n_bkg, mass, width, slope, intercept):
        """Breit-Wigner with linear background."""
        return n_signal * self.breit_wigner(x_vals, mass, width) + n_bkg * self.linear(x_vals, slope, intercept)

    def crystal_ball(self, x, alpha, n, mean, std_dev):
        """Crystal Ball function."""
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mean) / std_dev > -alpha,
                        np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)),
                        A * (B - (x - mean) / std_dev) ** (-n))

    def crystal_ball_exp(self, x_vals, n_signal, n_bkg, alpha, n, mean, std_dev, amplitude, decay):
        """Crystal Ball with exponential background."""
        return n_signal * self.crystal_ball(x_vals, alpha, n, mean, std_dev) + n_bkg * self.exponential(x_vals, amplitude, decay)

class FitBase:
    """Base class for all fit models."""

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

# Derived fit classes
class GaussianExpFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.gaussian_exp(x_vals, *params)

class DoubleGaussianExpFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.double_gaussian_exp(x_vals, *params)

class DoubleGaussianParabolaFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.double_gaussian_parabola(x_vals, *params)

class DoubleGaussianLinearFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.double_gaussian_linear(x_vals, *params)

class GaussianArgusFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.gaussian_argus(x_vals, *params)

class GaussianLinearFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.gaussian_linear(x_vals, *params)

class BreitWignerExpFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.breit_wigner_exp(x_vals, *params)

class BreitWignerLinearFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.breit_wigner_linear(x_vals, *params)

class CrystalBallExpFit(FitBase):
    def fit_function(self, x_vals, *params):
        return self.fit_models.crystal_ball_exp(x_vals, *params)