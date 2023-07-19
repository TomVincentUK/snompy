"""
Sample properties (:mod:`pysnom.sample`)
========================================

.. currentmodule:: pysnom.sample

This module provides a class to represent layered and bulk samples within
``pysnom``, and functions for converting between reflection coefficients
and permitivitties.

Classes
-------
.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated/

    Sample

Functions
---------
.. autosummary::
    :nosignatures:
    :toctree: generated/

    refl_coef_qs_single
    permitivitty
"""
import warnings

import numpy as np
from numpy.polynomial.laguerre import laggauss

from ._defaults import defaults
from ._utils import _pad_for_broadcasting


class Sample:
    r"""A class representing a layered sample with a semi-infinite
    substrate and superstrate.

    Parameters
    ----------
    eps_stack : array_like
        Dielectric function of each layer in the stack (with the first
        element corresponding to the semi-infinite superstrate, and the
        last to the semi-infinite substrate).
        `eps_stack` should have first dimension size two greater than
        `t_stack`, and all `eps_stack[i, ...]` should be broadcastable with
        all `t_stack[i, ...]`.
        Either `eps_stack` or `beta_stack` must be None.
    beta_stack : array_like
        Quasistatic  reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
        `beta_stack` should have first dimension size one greater than
        `t_stack`, and all `beta_stack[i, ...]` should be broadcastable
        with all `t_stack[i, ...]`.
        Either `eps_stack` or `beta_stack` must be None.
    t_stack : array_like
        Thicknesses of each finite-thickness layer sandwiched between the
        interfaces in the stack. A zero-size array can be used for the case
        of a bulk sample with a single interface.
    eps_env : array_like
        Dielectric function of the environment (equivalent to
        `eps_stack[0]`). This is used to calculate `eps_stack` from
        `beta_stack` if needed.
    k_vac : float
        Vacuuum wavenumber of incident light in inverse meters. Used to
        calculate far-field reflection coefficients via the transfer matrix
        method. Should be broadcastable with all `eps_stack[i, ...]`.

    Attributes
    ----------
    eps_stack :
        Dielectric function of each layer in the stack (with the first
        element corresponding to the semi-infinite superstrate, and the
        last to the semi-infinite substrate).
    beta_stack :
        Quasistatic  reflection coefficients of each interface in the
        stack (with the first element corresponding to the top interface).
    t_stack :
        Thicknesses of each finite-thickness layer sandwiched between the
        interfaces in the stack.
    multilayer :
        True if sample has one or more finite-thickness layer sandwiched
        between the interfaces in the stack, False for bulk samples.
    eps_env : array_like
        Dielectric function of the enviroment (equivalent to
        `eps_stack[0]`).

    """

    def __init__(
        self, eps_stack=None, beta_stack=None, t_stack=None, eps_env=None, k_vac=None
    ):
        # Check input validity
        if (eps_stack is None) == (beta_stack is None):
            raise ValueError(
                " ".join(
                    [
                        "Either `eps_stack` or `beta_stack` must be specified,",
                        "(but not both).",
                    ]
                )
            )

        # Initialise internal variables
        self._t_stack = None
        self._eps_stack = None
        self._eps_env = defaults.eps_env if eps_env is None else eps_env

        # Initialise public variables
        self.t_stack = t_stack
        if eps_stack is not None:
            self.eps_stack = eps_stack
        elif beta_stack is not None:
            self.beta_stack = beta_stack
        self.k_vac = np.asarray(k_vac)

    @property
    def t_stack(self):
        return self._t_stack

    @t_stack.setter
    def t_stack(self, val):
        if val is None:
            self._t_stack = np.array([])
        else:
            self._t_stack = np.asarray(np.broadcast_arrays(*val))

        # Check inputs make sense
        self._check_layers_valid()
        if (self._t_stack == 0).any():
            warnings.warn(
                " ".join(
                    [
                        "`t_stack` contains zeros.",
                        "Zero-thickness dielectric layers are unphysical.",
                        "Results may not be as expected.",
                    ]
                )
            )

    @property
    def eps_stack(self):
        return self._eps_stack

    @eps_stack.setter
    def eps_stack(self, val):
        self._eps_stack = np.asarray(np.broadcast_arrays(*val), dtype=complex)

        # Check inputs make sense
        self._check_layers_valid()

    @property
    def beta_stack(self):
        # Calculate beta from eps
        return refl_coef_qs_single(self.eps_stack[:-1], self.eps_stack[1:])

    @beta_stack.setter
    def beta_stack(self, val):
        # Calculate eps_stack from beta assuming first eps is 1
        beta = np.asarray(np.broadcast_arrays(*val))
        eps_stack = np.ones([beta.shape[0] + 1, *beta.shape[1:]], dtype=complex)
        eps_stack[1:] = np.cumprod(permitivitty(beta), axis=0)
        eps_stack *= self.eps_env
        self.eps_stack = eps_stack

    @property
    def eps_env(self):
        return self._eps_env if self.eps_stack is None else self.eps_stack[0]

    @property
    def multilayer(self):
        # True if more than one interface
        return self._t_stack.shape[0] > 0

    def refl_coef_qs(self, q=0):
        """Return the momentum-dependent quasistatic reflection coefficient
        for the sample.

        Parameters
        ----------
        q : float or array_like
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `beta_stack[i, ...]` and
            `t_stack[i, ...]`.

        Returns
        -------
        beta_total : complex
            Quasistatic reflection coefficient of the sample.

        """
        beta_total = self.beta_stack[0] * np.ones_like(q)
        for i in range(self.t_stack.shape[0]):
            layer_decay = np.exp(-2 * q * self.t_stack[i])
            beta_total = (beta_total + self.beta_stack[i + 1] * layer_decay) / (
                1 + beta_total * self.beta_stack[i + 1] * layer_decay
            )
        return beta_total

    def transfer_matrix(self, theta_in=None, q=None, k_vac=None, polarization="p"):
        """Return the transfer matrix for the sample for incident light
        with a given wavenumber, in-plane momentum and polarization.

        Parameters
        ----------
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        k_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarisation of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        M : complex
            The transfer matrix of the sample.

        """
        if k_vac is None:
            if self.k_vac is None:
                raise ValueError("`k_vac` must not be None.")
            else:
                k_vac = self.k_vac
        else:
            k_vac = np.asarray(k_vac)

        # Get q from theta in if needed
        if theta_in is None:
            if q is None:
                q = 0
            else:
                q = np.asarray(q)
        else:
            if q is None:
                q = k_vac * np.sin(np.abs(theta_in))
            else:
                raise ValueError("Either `theta_in` or `q` must be None.")

        # Wavevector in each layer
        k_z_medium = np.sqrt(self.eps_stack * k_vac**2 - q**2)

        # Transmission matrix depends on polarization
        if polarization == "p":
            trans_factor = np.stack(
                [
                    self.eps_stack[i]
                    * k_z_medium[i + 1]
                    / (self.eps_stack[i + 1] * k_z_medium[i])
                    for i in range(len(self.beta_stack))
                ]
            )
        elif polarization == "s":
            trans_factor = np.stack(
                [k_z_medium[i + 1] / k_z_medium[i] for i in range(len(self.beta_stack))]
            )
        else:
            raise ValueError("`polarization` must be 's' or 'p'")

        # Optical path length of internal layers
        path_length = np.array(
            [k_z * t for k_z, t in zip(k_z_medium[1:-1], self.t_stack)]
        )

        # Transition and propagation matrices
        trans_stack = np.moveaxis(
            np.array(
                [
                    [1 + trans_factor, 1 - trans_factor],
                    [1 - trans_factor, 1 + trans_factor],
                ]
            )
            / 2,
            (0, 1),
            (-2, -1),
        )
        prop_stack = np.moveaxis(
            np.array(
                [
                    [np.exp(-1j * path_length), np.zeros_like(path_length)],
                    [np.zeros_like(path_length), np.exp(1j * path_length)],
                ]
            ),
            (0, 1),
            (-2, -1),
        )

        M = trans_stack[0]
        for T, P in zip(trans_stack[1:], prop_stack):
            M = M @ P @ T

        return M

    def refl_coef(self, theta_in=None, q=None, k_vac=None, polarization="p"):
        """Return the momentum-dependent Fresnel reflection coefficient
        for the sample, using the transfer matrix method.

        Parameters
        ----------
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        k_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarisation of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        r : complex
            Fresnel reflection coefficient of the sample.

        """
        M = self.transfer_matrix(
            theta_in=theta_in, q=q, k_vac=k_vac, polarization=polarization
        )
        return M[..., 1, 0] / M[..., 0, 0]

    def trans_coef(self, theta_in=None, q=None, k_vac=None, polarization="p"):
        """Return the momentum-dependent Fresnel transmission coefficient
        for the sample, using the transfer matrix method.

        Parameters
        ----------
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        k_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarisation of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        t : complex
            Fresnel transmission coefficient of the sample.

        """
        M = self.transfer_matrix(
            theta_in=theta_in, q=q, k_vac=k_vac, polarization=polarization
        )
        return 1 / M[..., 0, 0]

    def _check_layers_valid(self):
        if (self.t_stack is not None) and (self.eps_stack is not None):
            if self.eps_stack.shape[0] != self.t_stack.shape[0] + 2:
                raise ValueError(
                    " ".join(
                        [
                            "Invalid inputs:",
                            "`eps_stack` must be 2 longer than `t_stack`, or",
                            "`beta_stack` must be 1 longer than `t_stack`,",
                            "along the first axis.",
                        ]
                    )
                )

    def refl_coef_qs_above_surf(self, z_Q, n_lag=None):
        r"""Return the effective quasistatic reflection coefficient for a
        charge over the sample surface, evaluated at the position of the
        charge itself.

        This function works by performing integrals over all values of
        in-plane electromagnetic wave momentum `q`, using Gauss-Laguerre
        quadrature.

        Parameters
        ----------
        z_Q : float
            Height of the charge above the sample.
        n_lag : int
            The order of the Laguerre polynomial used to evaluate the integrals
            over all `q`.

        Returns
        -------
        beta_eff : complex
            The effective quasistatic reflection coefficient at position
            `z_Q`.

        See also
        --------
        numpy.polynomial.laguerre.laggauss :
            Laguerre polynomial weights and roots for integration.

        Notes
        -----
        This function evaluates

        .. math::

            \overline{\beta} =\frac
            {\int_0^\infty \beta(q) q e^{-2 z_Q q} dq}
            {\int_0^\infty q e^{-2 z_Q q} dq}

        where :math:`\overline{\beta}` is the effective quasistatic
        reflection coefficient for a charge at height :math:`z_Q`
        (evaluated at the position of the charge itself), :math:`q` is the
        electromagnetic wave momentum, and :math:`\beta(q)` is the
        momentum-dependent effective reflection coefficient for the
        surface  [1]_.

        To do this, the denominator is first solved explicitly as
        :math:`1 / (4 z_Q^2)`.

        The choice of :math:`N`, defined in this function as `n_lag`,
        will affect the accuracy of the approximation, with higher
        :math:`N` values leading to more accurate evaluation of the
        integrals.

        In this function the Laguerre weights and roots are found using
        :func:`numpy.polynomial.laguerre.laggauss` and the
        momentum-dependent reflection coefficient is found using
        :func:`pysnom.sample.Sample.refl_coef_qs`.

        References
        ----------
        .. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
           R. Hillenbrand, “Subsurface chemical nanoidentification by
           nano-FTIR spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359,
           Dec. 2020, doi: 10.1038/s41467-020-17034-6.

        """
        # Set defaults
        n_lag = defaults.n_lag if n_lag is None else n_lag

        # Evaluate integral in terms of x = q * 2 * z_Q
        x_lag, w_lag = [
            _pad_for_broadcasting(a, (self.refl_coef_qs(z_Q),)) for a in laggauss(n_lag)
        ]

        q = x_lag / np.asarray(2 * z_Q)

        beta_q = self.refl_coef_qs(q)

        beta_eff = z_Q * np.sum(w_lag * x_lag * beta_q, axis=0)

        return beta_eff

    def surf_pot_and_field(self, z_Q, n_lag=None):
        r"""Return the electric potential and field at the sample surface,
        induced by a charge above the top interface.

        This function works by performing integrals over all values of
        in-plane electromagnetic wave momentum `q`, using Gauss-Laguerre
        quadrature.

        Parameters
        ----------
        z_Q : float
            Height of the charge above the sample.
        n_lag : int
            The order of the Laguerre polynomial used to evaluate the integrals
            over all `q`.

        Returns
        -------
        phi : complex
            The electric potential induced at the sample surface by the charge.
        E : complex
            The component of the surface electric field perpendicular to the
            surface.

        See also
        --------
        numpy.polynomial.laguerre.laggauss :
            Laguerre polynomial weights and roots for integration.

        Notes
        -----
        This function evaluates the integrals

        .. math::

            \begin{align*}
                \phi \rvert_{z=0} &=
                \int_0^\infty \beta(q) e^{-2 z_Q q} dq,
                \ \text{and}\\
                E_z \rvert_{z=0} &=
                \int_0^\infty \beta(q) q e^{-2 z_Q q} dq,
            \end{align*}

        where :math:`\phi` is the electric potential, :math:`E_z` is the
        vertical component of the electric field, :math:`q` is the
        electromagnetic wave momentum, :math:`\beta(q)` is the
        momentum-dependent effective reflection coefficient for the
        surface, and :math:`z_Q` is the height of the inducing charge above
        the surface [1]_.

        To do this, it first makes the substitution :math:`x = 2 z_Q q`,
        such that the integrals become

        .. math::

            \begin{align*}
                \phi \rvert_{z=0}
                & = \frac{1}{2 z_Q} \int_0^\infty
                \beta\left(\frac{x}{2 z_Q}\right) e^{-x} dx, \ \text{and}\\
                E_z \rvert_{z=0}
                & = \frac{1}{4 z_Q^2} \int_0^\infty
                \beta\left(\frac{x}{2 z_Q}\right) x e^{-x} dx.
            \end{align*}

        It then uses the Gauss-Laguerre approximation [2]_

        .. math::

            \int_0^{\infty} e^{-x} f(x) dx \approx \sum_{n=1}^N w_n f(x_n),

        where :math:`x_n` is the :math:`n^{th}` root of the Laguerre
        polynomial

        .. math::

            L_N(x) = \sum_{n=0}^{N} {N \choose n} \frac{(-1)^n}{n!} x^n,

        and :math:`w_n` is a weight given by

        .. math::

            w_n = \frac{x_n}{\left((N + 1) L_{N+1}(x_n) \right)^2}.

        The integrals can therefore be approximated by the sums

        .. math::

            \begin{align*}
                \phi \rvert_{z=0}
                & \approx \frac{1}{2 z_Q}
                \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_Q}\right),
                \ \text{and}\\
                E_z \rvert_{z=0}
                & \approx \frac{1}{4 z_Q^2}
                \sum_{n=1}^N w_n \beta\left(\frac{x_n}{2 z_Q}\right) x_n.
            \end{align*}

        The choice of :math:`N`, defined in this function as `n_lag`,
        will affect the accuracy of the approximation, with higher
        :math:`N` values leading to more accurate evaluation of the
        integrals.

        In this function the Laguerre weights and roots are found using
        :func:`numpy.polynomial.laguerre.laggauss` and the
        momentum-dependent reflection coefficient is found using
        :func:`pysnom.sample.Sample.refl_coef_qs`.

        References
        ----------
        .. [1] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and
           R. Hillenbrand, “Subsurface chemical nanoidentification by
           nano-FTIR spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359,
           Dec. 2020, doi: 10.1038/s41467-020-17034-6.
        .. [2] S. Ehrich, “On stratified extensions of Gauss-Laguerre and
           Gauss-Hermite quadrature formulas,” J. Comput. Appl. Math., vol.
           140, no. 1-2, pp. 291-299, Mar. 2002,
           doi: 10.1016/S0377-0427(01)00407-1.

        """
        # Set defaults
        n_lag = defaults.n_lag if n_lag is None else n_lag

        # Evaluate integral in terms of x = q * 2 * z_Q
        x_lag, w_lag = [
            _pad_for_broadcasting(a, (self.refl_coef_qs(z_Q),)) for a in laggauss(n_lag)
        ]

        q = x_lag / np.asarray(2 * z_Q)

        beta_q = self.refl_coef_qs(q)

        phi = np.sum(w_lag * beta_q, axis=0) / (2 * z_Q)
        E = np.sum(w_lag * x_lag * beta_q, axis=0) / (4 * z_Q**2)

        return phi, E

    def image_depth_and_charge(self, z_Q, n_lag=None):
        r"""Calculate the depth and relative charge of an image charge
        induced below the top surface of the sample.

        This function works by evaluating the electric potential and field
        induced at the sample surface using :func:`Sample.surf_pot_and_field`.

        Parameters
        ----------
        z_Q : float
            Height of the charge above the sample.
        n_lag : int
            The order of the Laguerre polynomial used by :func:`surf_pot_and_field`.

        Returns
        -------
        d_image : complex
            The effective depth of the image charge induced below the
            surface.
        beta_image : complex
            The relative charge of the image charge induced below the
            surface.

        See also
        --------
        surf_pot_and_field : Surface electric potential and field.

        Notes
        -----

        This function calculates the depth of an image charge induced by a
        charge :math:`q` at height :math:`z_Q` above a sample surface using
        the equation

        .. math::

            z_{image} = \left|
                \frac{\phi \rvert_{z=0}}{E_z \rvert_{z=0}}
            \right| - z_Q,

        and the effective charge of the image, relative to :math:`q`, using
        the equation

        .. math::

            \beta_{image} =
            \frac{ \left( \phi \rvert_{z=0} \right)^2 }
            {E_z \rvert_{z=0}},

        where :math:`\phi` is the electric potential, and :math:`E_z` is
        the vertical component of the electric field. These are based on
        equations (9) and (10) from reference [1]_. The depth :math:`z_Q`
        is converted to a real number by taking the absolute value of the
        :math:`\phi`-:math:`E_z` ratio, as described in reference [2]_.

        References
        ----------
        .. [1] B. Hauer, A. P. Engelhardt, and T. Taubner,
           “Quasi-analytical model for scattering infrared near-field
           microscopy on layered systems,” Opt. Express, vol. 20, no. 12,
           p. 13173, Jun. 2012, doi: 10.1364/OE.20.013173.
        .. [2] C. Lupo et al., “Quantitative infrared near-field imaging of
           suspended topological insulator nanostructures,” pp. 1-23, Dec.
           2021, [Online]. Available: http://arxiv.org/abs/2112.10104

        """
        phi, E = self.surf_pot_and_field(z_Q, n_lag)
        d_image = np.abs(phi / E) - z_Q
        beta_image = phi**2 / E
        return d_image, beta_image


def bulk_sample(eps_sub=None, beta=None, eps_env=None, **kwargs):
    eps_env = defaults.eps_env if eps_env is None else eps_env
    eps_stack = None if eps_sub is None else (eps_env, eps_sub)
    beta_stack = None if beta is None else (beta,)
    return Sample(eps_stack=eps_stack, beta_stack=beta_stack, eps_env=eps_env, **kwargs)


def refl_coef_qs_single(eps_i, eps_j):
    """Return the quasistatic  reflection coefficient for an interface
    between two materials.

    Calculated using ``(eps_j - eps_i) / (eps_j + eps_i)``, where `eps_i`
    and `eps_j` are the permitivitties of two materials `i` and `j`.

    Parameters
    ----------
    eps_i : complex
        Dielectric function of material i.
    eps_j : complex, default 1 + 0j
        Dielectric function of material j.

    Returns
    -------
    beta : complex
        Quasistatic  reflection coefficient of the sample.

    See also
    --------
    refl_coef_qs_ml :
        Momentum-dependent reflection coefficient for multilayer samples.
    permitivitty :
        The inverse of this function.

    Examples
    --------
    Works with real inputs:

    >>> refl_coef_qs_single(1, 3)
    0.5

    Works with complex inputs:

    >>> refl_coef_qs_single(1, 1 + 1j)
    (0.2+0.4j)

    Performs vectorised operations:

    >>> refl_coef_qs_single([1, 3], [3, 5])
    array([0.5 , 0.25])
    >>> refl_coef_qs_single([1, 3], [[1], [3]])
    array([[ 0. , -0.5],
          [ 0.5,  0. ]])

    """
    eps_i = np.asarray(eps_i)
    eps_j = np.asarray(eps_j)
    return (eps_j - eps_i) / (eps_j + eps_i)


def permitivitty(beta, eps_i=1 + 0j):
    """Return the permittivity of a material based on its quasistatic
    reflection coefficient.

    Calculated using ``eps_i * (1 + beta) / (1 - beta)``, where `eps_i` is
    the permitivitty of the preceeding material, and `beta` is the
    quasistatic reflection coefficient of the sample.

    Parameters
    ----------
    beta : complex
        Quasistatic  reflection coefficient of the sample.
    eps_i : complex, default 1.0
        Dielectric function of material i (the preceeding material).

    Returns
    -------
    eps_j : complex
        Dielectric function of material j.

    See also
    --------
    refl_coef_qs_single :
        The inverse of this function.

    """
    beta = np.asarray(beta)
    eps_i = np.asarray(eps_i)
    return eps_i * (1 + beta) / (1 - beta)
