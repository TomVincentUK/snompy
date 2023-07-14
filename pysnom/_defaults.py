import numpy as np


class Defaults:
    """
    Default values (:class:`pysnom.defaults`)
    =========================================

    A class holding default values used by various functions in
    ``pysnom``.

    Sample-related
    --------------
    eps_env : complex, default 1 + 0j
        Dielectric function of the environment

    Tip-related
    -----------
    r_tip : float, default 20e-9
        Radius of curvature of the AFM tip.
    L_tip : float, default 300e-9
        Semi-major axis length of the effective spheroid from the finite
        dipole model.

    Finite dipole model-related
    ---------------------------
    g_factor : complex, default 0.7 * np.exp(0.06j)
        A dimensionless approximation relating the magnitude of charge
        induced in the AFM tip to the magnitude of the nearby charge which
        induced it. A small imaginary component can be used to account for
        phase shifts caused by the capacitive interaction of the tip and
        sample.
    d_Q0 : float, default 1.31 * L_tip / (L_tip + 2 * r_tip))
        Depth of an induced charge 0 within the tip. Specified in units
        of the tip radius.

    .. note::
        When not specified by the user, the default for `d_Q0` is
        calculated from `r_tip` and `L_tip` at execution time. There is
        therefore no value `pysnom.defaults.d_Q0`.

    d_Q1 : float, default 0.5
        Depth of an induced charge 1 within the tip. Specified in units
        of the tip radius.

    Demodulation-related
    --------------------
    n_trapz : int, default 64
        The number of intervals to use for the trapezium-method
        integration used by `pysnom.demodulate.demod`.

    :math:`q` integral-related
    --------------------------
    n_lag : int, default 64
        The order of the Laguerre polynomial approximation used for
        integrating exponentially-weighted, semi-definite integrals.

    Taylor inversion-related
    ------------------------
    n_tayl : int, default 16
        Order of the Taylor approximation to the effective polarizability.
    beta_threshold : float
        The maximum amplitude of returned `beta` values determined to be
        valid when inverting the Taylor approximation for the effective
        polarizability.

    """

    def __init__(
        self,
        eps_env=1 + 0j,
        r_tip=20e-9,
        L_tip=300e-9,
        g_factor=0.7 * np.exp(0.06j),
        d_Q1=0.5,
        n_trapz=64,
        n_lag=64,
        n_tayl=16,
        beta_threshold=1.01,
    ):
        # Sample-related properties
        self.eps_env = eps_env

        # Tip-related properties
        self.r_tip = r_tip
        self.L_tip = L_tip

        # FDM-related properties
        self.g_factor = g_factor
        self.d_Q1 = d_Q1

        # Demodulation-related properties
        self.n_trapz = n_trapz

        # q integral-related properties
        self.n_lag = n_lag

        # Taylor inversion-related properties
        self.n_tayl = n_tayl
        self.beta_threshold = beta_threshold

    def _fdm_defaults(self, r_tip, L_tip, g_factor, d_Q0, d_Q1):
        r_tip = self.r_tip if r_tip is None else r_tip
        L_tip = self.L_tip if L_tip is None else L_tip
        g_factor = self.g_factor if g_factor is None else g_factor
        if d_Q0 is None:
            d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)
        d_Q1 = self.d_Q1 if d_Q1 is None else d_Q1
        return r_tip, L_tip, g_factor, d_Q0, d_Q1


defaults = Defaults()
