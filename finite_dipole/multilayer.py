"""
Finite dipole model (FDM) for predicting contrasts in scanning near-field
optical microscopy (SNOM) measurements.

References
----------
.. [1] B. Hauer, A.P. Engelhardt, T. Taubner,
   Quasi-analytical model for scattering infrared near-field microscopy on
   layered systems,
   Opt. Express. 20 (2012) 13173.
   https://doi.org/10.1364/OE.20.013173.
.. [2] A. Cvitkovic, N. Ocelic, R. Hillenbrand
   Analytical model for quantitative prediction of material contrasts in
   scattering-type near-field optical microscopy,
   Opt. Express. 15 (2007) 8550.
   https://doi.org/10.1364/oe.15.008550.
"""
import numpy as np
from numba import njit
from scipy.special import j0


def _init_phi_integrand(beta_stack, width_stack):
    """
    Derivations here by Carla. Make sure to add a proper citation.
    """
    n_interfaces = len(beta_stack)
    if len(width_stack) + 1 != n_interfaces:
        raise ValueError(
            "`beta_stack` must have length one greater than `width_stack`."
        )

    full_width = np.sum(width_stack)

    # Ensure stacks are immutable
    beta_stack = tuple(beta_stack)
    width_stack = tuple(width_stack)

    if n_interfaces == 2:

        @njit
        def beta_A(k):
            return beta_stack[1]

        @njit
        def beta_B(k):
            return beta_stack[0] * beta_stack[1]

    elif n_interfaces == 3:

        @njit
        def beta_A(k):
            return (
                beta_stack[2]
                + beta_stack[1] * np.exp(2 * k * width_stack[1])
                + beta_stack[0] * beta_stack[1] * beta_stack[2]
            )

        @njit
        def beta_B(k):
            return (
                beta_stack[0] * beta_stack[2]
                + beta_stack[0] * beta_stack[1] * np.exp(2 * k * width_stack[0])
                + beta_stack[1] * beta_stack[2]
            )

    elif n_interfaces == 4:

        @njit
        def beta_A(k):
            return (
                beta_stack[3]
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[3]
                * np.exp(2 * k * width_stack[0])
                + beta_stack[2] * np.exp(2 * k * width_stack[2])
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[2]
                * np.exp(2 * k * (width_stack[0] + width_stack[2]))
                + beta_stack[1] * np.exp(2 * k * (width_stack[1] + width_stack[2]))
                + beta_stack[0]
                * beta_stack[2]
                * beta_stack[3]
                * np.exp(2 * k * (width_stack[0] + width_stack[1]))
                + beta_stack[1]
                * beta_stack[2]
                * beta_stack[3]
                * np.exp(2 * k * width_stack[1])
            )

        @njit
        def beta_B(k):
            return (
                beta_stack[0] * beta_stack[3]
                + beta_stack[1] * beta_stack[3] * np.exp(2 * k * width_stack[0])
                + beta_stack[0] * beta_stack[2] * np.exp(2 * k * width_stack[2])
                + beta_stack[1]
                * beta_stack[2]
                * np.exp(2 * k * (width_stack[0] + width_stack[2]))
                + beta_stack[0]
                * beta_stack[1]
                * np.exp(2 * k * (width_stack[1] + width_stack[2]))
                + beta_stack[2]
                * beta_stack[3]
                * np.exp(2 * k * (width_stack[0] + width_stack[1]))
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[2]
                * beta_stack[3]
                * np.exp(2 * k * width_stack[1])
            )

    elif n_interfaces == 5:

        @njit
        def beta_A(k):
            return (
                beta_stack[4]
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[2]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[0] + width_stack[2]))
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[4]
                * np.exp(2 * k * width_stack[0])
                + beta_stack[1]
                * beta_stack[2]
                * beta_stack[4]
                * np.exp(2 * k * width_stack[1])
                + beta_stack[0]
                * beta_stack[2]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[0] + width_stack[1]))
                + beta_stack[1]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[1] + width_stack[2]))
                + beta_stack[0]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[1] + width_stack[2] + width_stack[3]))
                + beta_stack[3] * np.exp(2 * k * width_stack[3])
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[3]
                * np.exp(2 * k * (width_stack[0] + width_stack[3]))
                + beta_stack[2] * np.exp(2 * k * (width_stack[2] + width_stack[3]))
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[2]
                * np.exp(2 * k * (width_stack[0] + width_stack[2] + width_stack[3]))
                + beta_stack[1]
                * np.exp(2 * k * (width_stack[1] + width_stack[2] + width_stack[3]))
                + beta_stack[2]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * width_stack[2])
            )

        @njit
        def beta_B(k):
            return (
                beta_stack[0] * beta_stack[4]
                + beta_stack[1]
                * beta_stack[2]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[0] + width_stack[2]))
                + beta_stack[1] * beta_stack[4] * np.exp(2 * k * width_stack[0])
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[2]
                * beta_stack[4]
                * np.exp(2 * k * width_stack[1])
                + beta_stack[2]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[0] + width_stack[1]))
                + beta_stack[0]
                * beta_stack[1]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[1] + width_stack[2]))
                + beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * (width_stack[1] + width_stack[2] + width_stack[3]))
                + beta_stack[0] * beta_stack[3] * np.exp(2 * k * width_stack[3])
                + beta_stack[1]
                * beta_stack[3]
                * np.exp(2 * k * (width_stack[0] + width_stack[3]))
                + beta_stack[0]
                * beta_stack[2]
                * np.exp(2 * k * (width_stack[2] + width_stack[3]))
                + beta_stack[1]
                * beta_stack[2]
                * np.exp(2 * k * (width_stack[0] + width_stack[2] + width_stack[3]))
                + beta_stack[0]
                * beta_stack[1]
                * np.exp(2 * k * (width_stack[1] + width_stack[2] + width_stack[3]))
                + beta_stack[0]
                * beta_stack[2]
                * beta_stack[3]
                * beta_stack[4]
                * np.exp(2 * k * width_stack[2])
            )

    else:
        raise NotImplementedError(
            f"Potential cannot be calculated for stacks with {n_interfaces} interfaces"
        )

    @njit
    def _A_and_exp(k, z, z_0):
        """
        Everything in the integrand from Carla's paper except the Bessel
        function (which won't work with numba's JIT compilation).
        """
        return (
            np.exp((z - 2 * z_0) * k)
            * (beta_stack[0] + beta_A(k) * np.exp(-2 * k * full_width))
            / (1 + beta_B(k) * np.exp(-2 * k * full_width))
        )

    def _phi_integrand(k, z, z_0, radius):
        return _A_and_exp(k, z, z_0) * j0(k * radius)

    return _phi_integrand


@njit
def geom_func_ML(z, x, radius, semi_maj_axis, g_factor):
    """
    Function that encapsulates the geometric properties of the tip-sample
    system. Defined as :math:`f_0` or :math:`f_1` in equation (11) of
    reference [1]_, for multilayer samples.

    Parameters
    ----------
    z : float
        Height of the tip above the sample. Defined as :math:`H` in
        reference [1]_.
    x : float
        Position of an induced charge within the tip. Specified relative to
        the tip radius. Defined as :math:`X_0` or :math:`X_1` in equation
        (11) of reference [1]_.
    radius : float
        Radius of curvature of the AFM tip in metres. Defined as
        :math:`\rho` in reference [1]_.
    semi_maj_axis : float
        Semi-major axis in metres of the effective spheroid from the FDM.
        Defined as :math:`L` in reference [1]_.
    g_factor : complex
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample. Defined as :math:`g` in reference [1]_.

    Returns
    -------
    f_n : complex
        A complex number encapsulating geometric properties of the tip-
        sample system.
    """
    return (
        (g_factor - (radius + z + x * radius) / (2 * semi_maj_axis))
        * np.log(4 * semi_maj_axis / (radius + 2 * z + 2 * x * radius))
        / np.log(4 * semi_maj_axis / radius)
    )
