from scipy.stats import rv_continuous
from .utils import FitModels
from scipy.integrate import quad


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