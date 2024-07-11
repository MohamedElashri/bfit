from .binned import CompositeModel
from .unbinned import UnbinnedCompositeModel
from .components import (
    GaussianComponent, ExponentialComponent, LinearComponent,
    ParabolaComponent, BreitWignerComponent, CrystalBallComponent, ArgusComponent
)
import numpy as np

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