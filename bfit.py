import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad
from scipy.stats import rv_continuous
from typing import List, Tuple, Optional, Callable, Dict, Any, Union

class PlotConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.title: str = kwargs.get('title', 'Fit plot')
        self.xlabel: str = kwargs.get('xlabel', 'X')
        self.ylabel: str = kwargs.get('ylabel', 'Y')
        self.vlines: Optional[List[float]] = kwargs.get('vlines', None)
        self.plot_range: Optional[Tuple[float, float]] = kwargs.get('plot_range', None)
        self.data_color: str = kwargs.get('data_color', 'b')
        self.data_label: str = kwargs.get('data_label', 'Data')
        self.fit_color: str = kwargs.get('fit_color', 'r')
        self.fit_label: str = kwargs.get('fit_label', 'Fit')
        self.show_plot: bool = kwargs.get('show_plot', True)

class FitModels:
    def __init__(self, bin_width: float = 1, x_min: float = 0, x_max: float = 1) -> None:
        self.bin_width: float = bin_width
        self.x_min: float = x_min
        self.x_max: float = x_max

    def gaussian(self, x: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
        return np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) / np.sqrt(2 * np.pi * std_dev ** 2)

    def exponential(self, x: np.ndarray, amplitude: float, decay: float) -> np.ndarray:
        return amplitude * np.exp(-decay * (x - self.x_min))

    def linear(self, x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return slope * x + intercept

    def parabola(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (x ** 2) + b * x + c

    def breit_wigner(self, x: np.ndarray, mass: float, width: float) -> np.ndarray:
        return (1 / np.pi) * (0.5 * width) / ((x - mass) ** 2 + (0.5 * width) ** 2)

    def crystal_ball(self, x: np.ndarray, alpha: float, n: float, mean: float, std_dev: float) -> np.ndarray:
        A = (n / np.abs(alpha)) ** n * np.exp(-alpha ** 2 / 2)
        B = n / np.abs(alpha) - np.abs(alpha)
        return np.where((x - mean) / std_dev > -alpha,
                        np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)),
                        A * (B - (x - mean) / std_dev) ** (-n))

    def argus_bg(self, x: np.ndarray, m0: float, c: float, p: float) -> np.ndarray:
        z = 1 - (x / m0) ** 2
        return np.where(z > 0, x * np.sqrt(z) * np.exp(c * z ** p), 0)

class Component(rv_continuous):
    def __init__(self, function: Callable, pdf: Callable, n_params: int) -> None:
        super().__init__()
        self.function: Callable = function
        self._pdf: Callable = pdf
        self.n_params: int = n_params
        self.normalization: float = 1.0

    def __call__(self, x: np.ndarray, *params: float) -> np.ndarray:
        return self.function(x, *params)

    def pdf(self, x: np.ndarray, *params: float) -> np.ndarray:
        return self._pdf(x, *params) / self.normalization

    def normalize(self, x_min: float, x_max: float, *params: float) -> float:
        self.normalization, _ = quad(self._pdf, x_min, x_max, args=params)
        return self.normalization

class FitBase:
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.bins: np.ndarray = bins
        self.x_min: float
        self.x_max: float
        self.bin_width: float
        self.x_vals: np.ndarray
        self.y_vals: np.ndarray
        self.y_errs: np.ndarray
        self.x_min, self.x_max, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self._setup_fit(counts, bins)
        self.param_limits: Dict[str, Tuple[float, float]] = param_limits or {}
        self.fit_models: FitModels = FitModels(self.bin_width, self.x_min, self.x_max)
        self.fit_result: Optional[Minuit] = None

    def _setup_fit(self, counts: np.ndarray, bins: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max = bins[0], bins[-1]
        bin_width = bins[1] - bins[0]
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        y_vals, y_errs = counts, np.sqrt(counts)
        return x_min, x_max, bin_width, x_vals, y_vals, y_errs

    def chi_squared(self, *params: float) -> float:
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        residuals_squared = ((y_vals_masked - prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

    def fit(self, initial_params: List[float], param_names: Optional[List[str]] = None) -> Minuit:
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.chi_squared, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config: Optional[PlotConfig] = None) -> Tuple[plt.Figure, plt.Axes]:
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

    def summary(self) -> None:
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self) -> Dict[str, float]:
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self) -> Dict[str, float]:
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def chi_squared_value(self) -> float:
        if self.fit_result:
            return self.fit_result.fval
        else:
            raise RuntimeError("Fit has not been performed yet.")

class CompositeModel(FitBase):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.components: List[Tuple[Component, float]] = []

    def add_component(self, component: Component, weight: float = 1.0) -> None:
        self.components.append((component, weight))

    def fit_function(self, x_vals: np.ndarray, *params: float) -> np.ndarray:
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component(x_vals, *component_params)
            param_index += n_params
        return result

    def chi_squared(self, *params: float) -> float:
        mask = self.y_errs > 0
        x_vals_masked = self.x_vals[mask]
        y_vals_masked = self.y_vals[mask]
        y_errs_masked = self.y_errs[mask]
        prediction = self.fit_function(x_vals_masked, *params)
        
        scale_factor = np.sum(y_vals_masked) / np.sum(prediction)
        scaled_prediction = prediction * scale_factor
        
        residuals_squared = ((y_vals_masked - scaled_prediction) / y_errs_masked) ** 2
        return np.sum(residuals_squared)

class UnbinnedFitBase:
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.data: np.ndarray = data
        self.x_min: float
        self.x_max: float
        self.x_min, self.x_max = self._setup_fit(data)
        self.param_limits: Dict[str, Tuple[float, float]] = param_limits or {}
        self.fit_models: FitModels = FitModels(x_min=self.x_min, x_max=self.x_max)
        self.fit_result: Optional[Minuit] = None

    def _setup_fit(self, data: np.ndarray) -> Tuple[float, float]:
        return np.min(data), np.max(data)

    def log_likelihood(self, *params: float) -> float:
        raise NotImplementedError("Subclass must implement abstract method")
        
    def fit(self, initial_params: List[float], param_names: Optional[List[str]] = None) -> Minuit:
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(initial_params))]
        assert len(param_names) == len(initial_params), "Number of parameter names must match the number of initial parameters."
        
        minuit = Minuit(self.log_likelihood, name=param_names, *initial_params)
        for key, value in self.param_limits.items():
            minuit.limits[key] = value
        minuit.migrad()
        self.fit_result = minuit
        return minuit

    def plot(self, config: Optional[PlotConfig] = None) -> Tuple[plt.Figure, plt.Axes]:
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

    def summary(self) -> None:
        if self.fit_result is not None:
            print(self.fit_result)
        else:
            print("Fit has not been performed yet.")

    @property
    def fit_params(self) -> Dict[str, float]:
        if self.fit_result:
            return self.fit_result.values
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def fit_errors(self) -> Dict[str, float]:
        if self.fit_result:
            return self.fit_result.errors
        else:
            raise RuntimeError("Fit has not been performed yet.")

    @property
    def log_likelihood_value(self) -> float:
        if self.fit_result:
            return -self.fit_result.fval  # Negative because we minimize -log(L)
        else:
            raise RuntimeError("Fit has not been performed yet.")

class UnbinnedCompositeModel(UnbinnedFitBase):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.components: List[Tuple[Component, float]] = []

    def add_component(self, component: Component, weight: float = 1.0) -> None:
        """Add a component to the composite model."""
        self.components.append((component, weight))

    def pdf(self, x_vals: np.ndarray, *params: float) -> np.ndarray:
        """Composite probability density function."""
        result = np.zeros_like(x_vals)
        param_index = 0
        for component, weight in self.components:
            n_params = component.n_params
            component_params = params[param_index:param_index+n_params]
            component.normalize(self.x_min, self.x_max, *component_params)
            result += weight * component.pdf(x_vals, *component_params)
            param_index += n_params
        return result

    def log_likelihood(self, *params: float) -> float:
        """Calculate log-likelihood value."""
        epsilon = 1e-10  # Small value to prevent log(0)
        return -np.sum(np.log(self.pdf(self.data, *params) + epsilon))

# Pre-defined components
class GaussianComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().gaussian, FitModels().gaussian, 2)  # mean, std_dev

class ExponentialComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().exponential, FitModels().exponential, 2)  # amplitude, decay

class LinearComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().linear, FitModels().linear, 2)  # slope, intercept

class ParabolaComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().parabola, FitModels().parabola, 3)  # a, b, c

class BreitWignerComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().breit_wigner, FitModels().breit_wigner, 2)  # mass, width

class CrystalBallComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().crystal_ball, FitModels().crystal_ball, 4)  # alpha, n, mean, std_dev

class ArgusComponent(Component):
    def __init__(self) -> None:
        super().__init__(FitModels().argus_bg, FitModels().argus_bg, 3)  # m0, c, p

# Binned fit models
class GaussianExpFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianExpFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class DoubleGaussianParabolaFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class DoubleGaussianLinearFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class GaussianArgusFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class GaussianLinearFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class BreitWignerExpFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class BreitWignerLinearFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class CrystalBallExpFit(CompositeModel):
    def __init__(self, bins: np.ndarray, counts: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(bins, counts, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())

# Unbinned fit models
class UnbinnedGaussianExpFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianExpFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ExponentialComponent())

class UnbinnedDoubleGaussianParabolaFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(ParabolaComponent())

class UnbinnedDoubleGaussianLinearFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedGaussianArgusFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(ArgusComponent())

class UnbinnedGaussianLinearFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(GaussianComponent())
        self.add_component(LinearComponent())

class UnbinnedBreitWignerExpFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(ExponentialComponent())

class UnbinnedBreitWignerLinearFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(BreitWignerComponent())
        self.add_component(LinearComponent())

class UnbinnedCrystalBallExpFit(UnbinnedCompositeModel):
    def __init__(self, data: np.ndarray, param_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        super().__init__(data, param_limits)
        self.add_component(CrystalBallComponent())
        self.add_component(ExponentialComponent())
