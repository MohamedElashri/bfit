import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad
from scipy.stats import rv_continuous

class PlotConfig:
    """
    Configuration class for plot customization.

    This class holds various parameters for customizing plots, including
    titles, labels, colors, and display options.

    Attributes:
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        vlines (list): Vertical lines to be drawn on the plot.
        plot_range (tuple): The range of x-values to be plotted.
        data_color (str): The color of the data points.
        data_label (str): The label for the data in the legend.
        fit_color (str): The color of the fit line.
        fit_label (str): The label for the fit in the legend.
        show_plot (bool): Whether to display the plot immediately.
    """

    def __init__(self, **kwargs):
        """
        Initialize the PlotConfig with customizable options.

        Args:
            **kwargs: Keyword arguments for customizing plot attributes.
        """
        self.title = kwargs.get('title', 'Fit plot')
        self.xlabel = kwargs.get('xlabel', 'X')
        self.ylabel = kwargs.get('ylabel', 'Y')
        self.vlines = kwargs.get('vlines', None)
        self.plot_range = kwargs.get('plot_range', None)
        self.data_color = kwargs.get('data_color', 'b')
        self.data_label = kwargs.get('data_label', 'Data')
        self.fit_color = kwargs.get('fit_color', 'r')
        self.fit_label = kwargs.get('fit_label', 'Fit')
        self.show_plot = kwargs.get('show_plot', True)

class FitModels:
    """
    Contains various probability distribution functions and background models.

    This class provides a collection of common statistical distributions and
    background models used in fitting procedures.

    Attributes:
        bin_width (float): The width of histogram bins.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
    """
    
    def __init__(self, bin_width=1, x_min=0, x_max=1):
        """
        Initialize the FitModels instance.

        Args:
            bin_width (float): The width of histogram bins.
            x_min (float): The minimum x-value of the fit range.
            x_max (float): The maximum x-value of the fit range.
        """
        self.bin_width = bin_width
        self.x_min = x_min
        self.x_max = x_max

    def gaussian(self, x, mean, std_dev):
        """
        Gaussian distribution function.

        Args:
            x (array-like): The x-values.
            mean (float): The mean of the distribution.
            std_dev (float): The standard deviation of the distribution.

        Returns:
            array-like: The y-values of the Gaussian distribution.
        """
        return np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / np.sqrt(2 * np.pi * std_dev ** 2)

    def exponential(self, x, amplitude, decay):
        """
        Exponential distribution function.

        Args:
            x (array-like): The x-values.
            amplitude (float): The amplitude of the exponential.
            decay (float): The decay rate of the exponential.

        Returns:
            array-like: The y-values of the exponential distribution.
        """
        return amplitude * np.exp(-decay * (x - self.x_min))

    def linear(self, x, slope, intercept):
        """
        Linear function.

        Args:
            x (array-like): The x-values.
            slope (float): The slope of the line.
            intercept (float): The y-intercept of the line.

        Returns:
            array-like: The y-values of the linear function.
        """
        return slope * x + intercept

    def parabola(self, x, a, b, c):
        """
        Parabolic function.

        Args:
            x (array-like): The x-values.
            a (float): The coefficient of x^2.
            b (float): The coefficient of x.
            c (float): The constant term.

        Returns:
            array-like: The y-values of the parabolic function.
        """
        return a * (x ** 2) + b * x + c

    def breit_wigner(self, x, mass, width):
        """
        Breit-Wigner distribution function.

        Args:
            x (array-like): The x-values.
            mass (float): The mass parameter.
            width (float): The width parameter.

        Returns:
            array-like: The y-values of the Breit-Wigner distribution.
        """
        return (1 / np.pi) * (0.5 * width) / ((x - mass) ** 2 + (0.5 * width) ** 2)

    def crystal_ball(self, x, alpha, n, mean, std_dev):
        """
        Crystal Ball function.

        Args:
            x (array-like): The x-values.
            alpha (float): The alpha parameter.
            n (float): The n parameter.
            mean (float): The mean of the Gaussian core.
            std_dev (float): The standard deviation of the Gaussian core.

        Returns:
            array-like: The y-values of the Crystal Ball function.
        """
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mean) / std_dev > -alpha,
                        np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)),
                        A * (B - (x - mean) / std_dev) ** (-n))

    def argus_bg(self, x, m0, c, p):
        """
        ARGUS background function.

        Args:
            x (array-like): The x-values.
            m0 (float): The kinematic limit.
            c (float): The curvature parameter.
            p (float): The power parameter.

        Returns:
            array-like: The y-values of the ARGUS background function.
        """
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, x * np.sqrt(z) * np.exp(c * z ** p), 0)

class Component(rv_continuous):
    """
    Base class for fit components.

    This class extends scipy's rv_continuous to create custom probability
    distributions that can be used as components in composite fit models.

    Attributes:
        function (callable): The function defining the component.
        _pdf (callable): The probability density function of the component.
        n_params (int): The number of parameters in the component.
        normalization (float): The normalization factor for the component.
    """

    def __init__(self, function, pdf, n_params):
        """
        Initialize the Component instance.

        Args:
            function (callable): The function defining the component.
            pdf (callable): The probability density function of the component.
            n_params (int): The number of parameters in the component.
        """
        super().__init__()
        self.function = function
        self._pdf = pdf
        self.n_params = n_params
        self.normalization = 1.0

    def __call__(self, x, *params):
        """
        Call the component's function.

        Args:
            x (array-like): The x-values.
            *params: The parameters for the function.

        Returns:
            array-like: The y-values of the function.
        """
        return self.function(x, *params)

    def pdf(self, x, *params):
        """
        Probability density function of the component.

        Args:
            x (array-like): The x-values.
            *params: The parameters for the PDF.

        Returns:
            array-like: The y-values of the PDF.
        """
        return self._pdf(x, *params) / self.normalization

    def normalize(self, x_min, x_max, *params):
        """
        Normalize the component over a given range.

        Args:
            x_min (float): The minimum x-value of the range.
            x_max (float): The maximum x-value of the range.
            *params: The parameters for the PDF.

        Returns:
            float: The normalization factor.
        """
        self.normalization, _ = quad(self._pdf, x_min, x_max, args=params)
        return self.normalization

class FitBase:
    """
    Base class for all binned fit models.

    This class provides the fundamental structure and methods for performing
    binned fits on histogram data.

    Attributes:
        bins (array-like): The bin edges of the histogram.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
        bin_width (float): The width of each bin.
        x_vals (array-like): The center of each bin.
        y_vals (array-like): The count in each bin.
        y_errs (array-like): The error on the count in each bin.
        param_limits (dict): Optional limits on the fit parameters.
        fit_models (FitModels): An instance of the FitModels class.
        fit_result (Minuit): The result of the fit, once performed.
    """

    def __init__(self, bins, counts, param_limits=None):
        """
        Initialize the FitBase instance.

        Args:
            bins (array-like): The bin edges of the histogram.
            counts (array-like): The counts in each bin.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        self.bins = bins
        self.x_min, self.x_max, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self._setup_fit(counts, bins)
        self.param_limits = param_limits or {}
        self.fit_models = FitModels(self.bin_width, self.x_min, self.x_max)
        self.fit_result = None

    def _setup_fit(self, counts, bins):
        """
        Set up the fit by calculating necessary values from the input data.

        Args:
            counts (array-like): The counts in each bin.
            bins (array-like): The bin edges of the histogram.

        Returns:
            tuple: x_min, x_max, bin_width, x_vals, y_vals, y_errs
        """
        x_min, x_max = bins[0], bins[-1]
        bin_width = bins[1] - bins[0]
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        y_vals, y_errs = counts, np.sqrt(counts)
        return x_min, x_max, bin_width, x_vals, y_vals, y_errs

    def chi_squared(self, *params):
        """
        Calculate chi-squared value for the current fit parameters.

        Args:
            *params: The current fit parameters.

        Returns:
            float: The chi-squared value.
        """
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        residuals_squared = ((y_vals_masked - prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

    def fit(self, initial_params, param_names=None):
        """
        Perform the fit using Minuit.

        Args:
            initial_params (list): Initial values for the fit parameters.
            param_names (list, optional): Names of the fit parameters.

        Returns:
            Minuit: The Minuit object containing the fit results.
        """
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.chi_squared, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config=None):
        """
        Plot the fit results.

        Args:
            config (PlotConfig, optional): Configuration for the plot.

        Returns:
            tuple: Figure and Axes objects of the plot.
        """
        if config is None:
            config = PlotConfig()

        fig, ax = plt.subplots()
        ax.set_title(config.title)

        visual_bins = self.bins
        if config.plot_range is not None:
            visual_bins = visual_bins[(visual_bins >= config.plot_range[0]) & (visual_bins <= config.plot_range[1] + self.bin_width)]
            ax.set_xlim(config.plot_range)

        ax.hist(self.x_vals, bins=visual_bins, weights=self.y_vals, color=config.data_color, label=config.data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.fit_function(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=config.fit_color, label=config.fit_label)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_ylim(0, 1.15 * max(np.append(self.y_vals, predictions)))
        ax.grid(True)
        
        if config.vlines:
            for vl in config.vlines:
                ax.vlines(vl, 0., 0.6 * max(self.y_vals), colors='yellow')
        
        ax.legend()

        if config.show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """
        Print a summary of the fit results.
        """
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """
        Get the fit parameters.

        Returns:
            dict: The fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """
        Get the errors on the fit parameters.

        Returns:
            dict: The errors on the fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

@property
    def chi_squared_value(self):
        """
        Get the chi-squared value of the fit.

        Returns:
            float: The chi-squared value.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.fval
        else:
            raise RuntimeError("Fit has not been performed yet.")

class CompositeModel(FitBase):
    """
    Composite model class for combining multiple fit components.

    This class allows the creation of complex models by combining multiple
    simpler components.

    Attributes:
        components (list): List of (component, weight) tuples.
    """

    def __init__(self, bins, counts, param_limits=None):
        """
        Initialize the CompositeModel instance.

        Args:
            bins (array-like): The bin edges of the histogram.
            counts (array-like): The counts in each bin.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        super().__init__(bins, counts, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """
        Add a component to the composite model.

        Args:
            component (Component): The component to add.
            weight (float, optional): The weight of the component in the model.
        """
        self.components.append((component, weight))

    def fit_function(self, x_vals, *params):
        """
        Composite fit function combining all components.

        Args:
            x_vals (array-like): The x-values to evaluate the function at.
            *params: The parameters for all components.

        Returns:
            array-like: The y-values of the composite function.
        """
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component(x_vals, *component_params)
            param_index += n_params
        return result

    def chi_squared(self, *params):
        """
        Calculate chi-squared value for the composite model.

        This method overrides the base class method to include scaling.

        Args:
            *params: The current fit parameters.

        Returns:
            float: The chi-squared value.
        """
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
    """
    Base class for all unbinned fit models.

    This class provides the fundamental structure and methods for performing
    unbinned fits on raw data points.

    Attributes:
        data (array-like): The raw data points.
        x_min (float): The minimum x-value of the fit range.
        x_max (float): The maximum x-value of the fit range.
        param_limits (dict): Optional limits on the fit parameters.
        fit_models (FitModels): An instance of the FitModels class.
        fit_result (Minuit): The result of the fit, once performed.
    """

    def __init__(self, data, param_limits=None):
        """
        Initialize the UnbinnedFitBase instance.

        Args:
            data (array-like): The raw data points.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        self.data = data
        self.x_min, self.x_max = self._setup_fit(data)
        self.param_limits = param_limits or {}
        self.fit_models = FitModels(x_min=self.x_min, x_max=self.x_max)
        self.fit_result = None

    def _setup_fit(self, data):
        """
        Set up the fit by calculating the range of the data.

        Args:
            data (array-like): The raw data points.

        Returns:
            tuple: x_min, x_max
        """
        return np.min(data), np.max(data)

    def log_likelihood(self, *params):
        """
        Calculate log-likelihood value.

        This method should be implemented by subclasses.

        Args:
            *params: The current fit parameters.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")
        
    def fit(self, initial_params, param_names=None):
        """
        Perform the fit using Minuit.

        Args:
            initial_params (list): Initial values for the fit parameters.
            param_names (list, optional): Names of the fit parameters.

        Returns:
            Minuit: The Minuit object containing the fit results.
        """
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.log_likelihood, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config=None):
        """
        Plot the fit results.

        Args:
            config (PlotConfig, optional): Configuration for the plot.

        Returns:
            tuple: Figure and Axes objects of the plot.
        """
        if config is None:
            config = PlotConfig()

        fig, ax = plt.subplots()
        ax.set_title(config.title)

        if config.plot_range is not None:
            ax.set_xlim(config.plot_range)
        else:
            ax.set_xlim(self.x_min, self.x_max)

        ax.hist(self.data, bins=100, density=True, color=config.data_color, label=config.data_label)

        if self.fit_result:
            dense_x_vals = np.linspace(self.x_min, self.x_max, 1000)
            predictions = self.pdf(dense_x_vals, *self.fit_result.values)
            ax.plot(dense_x_vals, predictions, color=config.fit_color, label=config.fit_label)

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.grid(True)
        
        if config.vlines:
            for vl in config.vlines:
                ax.vlines(vl, 0., 0.6 * ax.get_ylim()[1], colors='yellow')
        
        ax.legend()

        if config.show_plot:
            plt.show()

        return fig, ax

    def summary(self):
        """
        Print a summary of the fit results.
        """
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self):
        """
        Get the fit parameters.

        Returns:
            dict: The fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self):
        """
        Get the errors on the fit parameters.

        Returns:
            dict: The errors on the fit parameters.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def log_likelihood_value(self):
        """
        Get the log-likelihood value of the fit.

        Returns:
            float: The log-likelihood value.

        Raises:
            RuntimeError: If the fit has not been performed.
        """
        if self.fit_result:
            return -self.fit_result.fval  # Negative because we minimize -log(L)
        else:
            raise RuntimeError("Fit has not been performed yet.")

class UnbinnedCompositeModel(UnbinnedFitBase):
    """
    Composite model class for combining multiple unbinned fit components.

    This class allows the creation of complex unbinned models by combining
    multiple simpler components.

    Attributes:
        components (list): List of (component, weight) tuples.
    """

    def __init__(self, data, param_limits=None):
        """
        Initialize the UnbinnedCompositeModel instance.

        Args:
            data (array-like): The raw data points.
            param_limits (dict, optional): Limits on the fit parameters.
        """
        super().__init__(data, param_limits)
        self.components = []

    def add_component(self, component, weight=1.0):
        """
        Add a component to the composite model.

        Args:
            component (Component): The component to add.
            weight (float, optional): The weight of the component in the model.
        """
        self.components.append((component, weight))

    def pdf(self, x_vals, *params):
        """
        Composite probability density function combining all components.

        Args:
            x_vals (array-like): The x-values to evaluate the PDF at.
            *params: The parameters for all components.

        Returns:
            array-like: The y-values of the composite PDF.
        """
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component.pdf(x_vals, *component_params)
            param_index += n_params
        return result

    def log_likelihood(self, *params):
        """
        Calculate log-likelihood value for the composite model.

        Args:
            *params: The current fit parameters.

        Returns:
            float: The negative log-likelihood value.
        """
        epsilon = 1e-10  # Small value to prevent log(0)
        return -np.sum(np.log(self.pdf(self.data, *params) + epsilon))

# Pre-defined components
class GaussianComponent(Component):
    """Gaussian distribution component."""
    def __init__(self):
        super().__init__(FitModels().gaussian, FitModels().gaussian, 2)  # mean, std_dev

class ExponentialComponent(Component):
    """Exponential distribution component."""
    def __init__(self):
        super().__init__(FitModels().exponential, FitModels().exponential, 2)  # amplitude, decay

class LinearComponent(Component):
    """Linear function component."""
    def __init__(self):
        super().__init__(FitModels().linear, FitModels().linear, 2)  # slope, intercept

class ParabolaComponent(Component):
    """Parabolic function component."""
    def __init__(self):
        super().__init__(FitModels().parabola, FitModels().parabola, 3)  # a, b, c

class BreitWignerComponent(Component):
    """Breit-Wigner distribution component."""
    def __init__(self):
        super().__init__(FitModels().breit_wigner, FitModels().breit_wigner, 2)  # mass, width

class CrystalBallComponent(Component):
    """Crystal Ball function component."""
    def __init__(self):
        super().__init__(FitModels().crystal_ball, FitModels().crystal_ball, 4)  # alpha, n, mean, std_dev

class ArgusComponent(Component):
    """ARGUS background function component."""
    def __init__(self):
        super().__init__(FitModels().argus_bg, FitModels().argus_bg, 3)  # m0, c, p

# Binned fit models
class GaussianExpFit(CompositeModel):
    """Gaussian + Exponential binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianExpFit(CompositeModel):
    """Double Gaussian + Exponential binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianParabolaFit(CompositeModel):
    """Double Gaussian + Parabola binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class DoubleGaussianLinearFit(CompositeModel):
    """Double Gaussian + Linear binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class GaussianArgusFit(CompositeModel):
    """Gaussian + ARGUS binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class GaussianLinearFit(CompositeModel):
    """Gaussian + Linear binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class BreitWignerExpFit(CompositeModel):
    """Breit-Wigner + Exponential binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class BreitWignerLinearFit(CompositeModel):
    """Breit-Wigner + Linear binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class CrystalBallExpFit(CompositeModel):
    """Crystal Ball + Exponential binned fit model."""
    def __init__(self, bins, counts, param_limits=None):
        super().__init__(bins, counts, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())

# Unbinned fit models
class UnbinnedGaussianExpFit(UnbinnedCompositeModel):
    """Gaussian + Exponential unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianExpFit(UnbinnedCompositeModel):
    """Double Gaussian + Exponential unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianParabolaFit(UnbinnedCompositeModel):
    """Double Gaussian + Parabola unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class UnbinnedDoubleGaussianLinearFit(UnbinnedCompositeModel):
    """Double Gaussian + Linear unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedGaussianArgusFit(UnbinnedCompositeModel):
    """Gaussian + ARGUS unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class UnbinnedGaussianLinearFit(UnbinnedCompositeModel):
    """Gaussian + Linear unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedBreitWignerExpFit(UnbinnedCompositeModel):
    """Breit-Wigner + Exponential unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class UnbinnedBreitWignerLinearFit(UnbinnedCompositeModel):
    """Breit-Wigner + Linear unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class UnbinnedCrystalBallExpFit(UnbinnedCompositeModel):
    """Crystal Ball + Exponential unbinned fit model."""
    def __init__(self, data, param_limits=None):
        super().__init__(data, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())
