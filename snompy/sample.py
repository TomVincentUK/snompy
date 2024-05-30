"""
Sample properties (:mod:`snompy.sample`)
========================================

.. currentmodule:: snompy.sample

This module provides a class to represent layered and bulk samples within
``snompy``, and functions for converting between reflection coefficients
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

    bulk_sample
    refl_coef_qs_single
    permitivitty
    lorentz_perm
    drude_perm
"""

import functools

import numpy as np

from ._defaults import defaults
from ._utils import _pad_for_broadcasting

# Cache common functions to speed up execution
laggauss = functools.cache(np.polynomial.laguerre.laggauss)

# Maximum floating point argument to np.exp that doesn't overflow
_MAX_EXP_ARG = np.log(np.finfo(float).max)


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
    nu_vac : float
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
        self, eps_stack=None, beta_stack=None, t_stack=None, eps_env=None, nu_vac=None
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
        self.nu_vac = None if nu_vac is None else np.array(nu_vac)

    @property
    def t_stack(self):
        return self._t_stack

    @t_stack.setter
    def t_stack(self, val):
        if val is None:
            self._t_stack = np.array([])
        else:
            self._t_stack = np.asanyarray(np.broadcast_arrays(*val))

        # Check inputs make sense
        self._check_layers_valid()

    @property
    def eps_stack(self):
        return self._eps_stack

    @eps_stack.setter
    def eps_stack(self, val):
        self._eps_stack = np.asanyarray(np.broadcast_arrays(*val), dtype=complex)

        # Check inputs make sense
        self._check_layers_valid()

    @property
    def beta_stack(self):
        # Calculate beta from eps
        return refl_coef_qs_single(self.eps_stack[:-1], self.eps_stack[1:])

    @beta_stack.setter
    def beta_stack(self, val):
        # Calculate eps_stack from beta assuming first eps is 1
        beta = np.asanyarray(np.broadcast_arrays(*val))
        eps_stack = np.ones([beta.shape[0] + 1, *beta.shape[1:]], dtype=complex)
        eps_stack[1:] = np.cumprod(permitivitty(beta), axis=0)
        eps_stack *= self.eps_env
        self.eps_stack = eps_stack

    @property
    def eps_env(self):
        return self._eps_env if self.eps_stack is None else self.eps_stack[0]

    @property
    def n_layers(self):
        return self._eps_stack.shape[0]

    @property
    def multilayer(self):
        # True if more than one interface
        return self.n_layers > 2

    def transfer_matrix_qs(self, q=0.0):
        """Return the transfer matrix for the sample in the quasistatic
        limit.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.

        Returns
        -------
        M_qs : complex
            The quasistatic transfer matrix of the sample.

        Notes
        -----
        This implementation of the transfer matrix method is based on the
        description given in reference [1]_.

        References
        ----------
        .. [1] T. Zhan, X. Shi, Y. Dai, X. Liu, and J. Zi, “Transfer matrix
           method for optics in graphene layers,” J. Phys. Condens. Matter,
           vol. 25, no. 21, p. 215301, May 2013,
           doi: 10.1088/0953-8984/25/21/215301.

        """
        trans_factor = self.eps_stack[:-1] / self.eps_stack[1:]
        trans_matrices = (
            1 + np.array([[1, -1], [-1, 1]]) * trans_factor[..., np.newaxis, np.newaxis]
        ) / 2

        # Convert stack to single transfer matrix (accounting for propagation if needed)
        M_qs = trans_matrices[0]
        if self.multilayer:
            # Optical path length of internal layers
            prop_factor = np.array([q * t for t in self.t_stack])

            # Avoid overflow by clipping (there's probably a more elegant way)
            prop_factor = np.clip(prop_factor, -_MAX_EXP_ARG, _MAX_EXP_ARG)
            prop_matrices = np.exp(
                np.array([1, -1]) * prop_factor[..., np.newaxis, np.newaxis]
            ) * np.eye(2)

            for T, P in zip(trans_matrices[1:], prop_matrices):
                M_qs = M_qs @ P @ T
        else:
            M_qs = M_qs * np.ones_like(q)[..., np.newaxis, np.newaxis]

        return M_qs

    def refl_coef_qs(self, q=0.0):
        """Return the momentum-dependent quasistatic reflection coefficient
        for the sample.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.

        Returns
        -------
        beta : complex
            Quasistatic reflection coefficient of the sample.

        """
        M = self.transfer_matrix_qs(q=q)
        return M[..., 1, 0] / M[..., 0, 0]

    def trans_coef_qs(self, q=0.0):
        """Return the momentum-dependent quasistatic transmission
        coefficient for the sample.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.

        Returns
        -------
        t_qs : complex
            Quasistatic transmission coefficient of the sample.

        """
        M = self.transfer_matrix_qs(q=q)
        return 1 / M[..., 0, 0]

    def transfer_matrix(self, q=None, theta_in=None, nu_vac=None, polarization="p"):
        """Return the transfer matrix for the sample for incident light
        with a given wavenumber, in-plane momentum and polarization.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        nu_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarization of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        M : complex
            The transfer matrix of the sample.

        Notes
        -----
        This implementation of the transfer matrix method is based on the
        description given in reference [1]_.

        References
        ----------
        .. [1] T. Zhan, X. Shi, Y. Dai, X. Liu, and J. Zi, “Transfer matrix
           method for optics in graphene layers,” J. Phys. Condens. Matter,
           vol. 25, no. 21, p. 215301, May 2013,
           doi: 10.1088/0953-8984/25/21/215301.

        """
        # Check nu_vac given for multilayer samples (at init or function call)
        if nu_vac is None:
            if self.nu_vac is None:
                if self.multilayer:
                    raise ValueError(
                        "`nu_vac` must not be None for multilayer samples."
                    )
                else:
                    nu_vac = 1  # nu_vac has no effect for bulk samples
            else:
                nu_vac = self.nu_vac
        else:
            nu_vac = np.asanyarray(nu_vac)

        # Get q from theta in if needed
        if theta_in is None:
            if q is None:
                q = 0
            else:
                q = np.asanyarray(q)
        else:
            if q is None:
                q = nu_vac * np.sin(np.abs(theta_in))
            else:
                raise ValueError("Either `theta_in` or `q` must be None.")

        # Wavevector in each layer
        nu_z_medium = np.stack(
            [np.sqrt(eps * nu_vac**2 - q**2) for eps in self.eps_stack]
        )

        # Transmission matrix depends on polarization
        if polarization == "p":
            trans_factor = np.stack(
                [
                    self.eps_stack[i]
                    * nu_z_medium[i + 1]
                    / (self.eps_stack[i + 1] * nu_z_medium[i])
                    for i in range(self.n_layers - 1)
                ]
            )
        elif polarization == "s":
            trans_factor = np.stack(
                [nu_z_medium[i + 1] / nu_z_medium[i] for i in range(self.n_layers - 1)]
            )
        else:
            raise ValueError("`polarization` must be 's' or 'p'")

        trans_matrices = (
            1 + np.array([[1, -1], [-1, 1]]) * trans_factor[..., np.newaxis, np.newaxis]
        ) / 2

        # Convert stack to single transfer matrix (accounting for propagation if needed)
        M = trans_matrices[0]
        if self.multilayer:
            # Optical path length of internal layers
            prop_factor = np.array(
                [nu_z * t for nu_z, t in zip(nu_z_medium[1:-1], self.t_stack)]
            )

            prop_matrices = np.exp(
                np.array([-1j, 1j]) * prop_factor[..., np.newaxis, np.newaxis]
            ) * np.eye(2)

            for T, P in zip(trans_matrices[1:], prop_matrices):
                M = M @ P @ T

        return M

    def refl_coef(self, q=None, theta_in=None, nu_vac=None, polarization="p"):
        """Return the momentum-dependent Fresnel reflection coefficient
        for the sample, using the transfer matrix method.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        nu_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarization of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        r : complex
            Fresnel reflection coefficient of the sample.

        """
        M = self.transfer_matrix(
            q=q, theta_in=theta_in, nu_vac=nu_vac, polarization=polarization
        )
        return M[..., 1, 0] / M[..., 0, 0]

    def trans_coef(self, q=None, theta_in=None, nu_vac=None, polarization="p"):
        """Return the momentum-dependent Fresnel transmission coefficient
        for the sample, using the transfer matrix method.

        Parameters
        ----------
        q : float, default 0.0
            In-plane electromagnetic wave momentum.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Either `q` or `theta_in` must be None.
        theta_in : float
            Angle of the incident light to the surface normal in radians.
            Must be broadcastable with all `eps_stack[i, ...]` and
            `t_stack[i, ...]`. Used to calculate `q`. Either `q` or
            `theta_in` must be None.
        nu_vac : float
            Vacuuum wavenumber of incident light in inverse meters. Used to
            calculate far-field reflection coefficients via the transfer
            matrix method. Should be broadcastable with all
            `eps_stack[i, ...]`.
        polarization: {"p", "s"}
            The polarization of the incident light. "p" for parallel to the
            plane of incidence, and "s" for perpendicular (from the German
            word *senkrecht*).

        Returns
        -------
        t : complex
            Fresnel transmission coefficient of the sample.

        """
        M = self.transfer_matrix(
            q=q, theta_in=theta_in, nu_vac=nu_vac, polarization=polarization
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
        :math:`1 / (4 z_Q^2)`. Then the substitution :math:`x = 2 z_Q q`,
        is made to give

        .. math::

            \overline{\beta} =
            \int_0^\infty \beta\left(\frac{x}{2 z_Q}\right) x e^{-x} dx.

        It then uses the Gauss-Laguerre approximation [2]_

        .. math::

            \int_0^{\infty} e^{-x} f(x) dx \approx \sum_{j=1}^J w_j f(x_j),

        where :math:`x_j` is the :math:`j^{th}` root of the Laguerre
        polynomial

        .. math::

            L_J(x) = \sum_{j=0}^{J} {J \choose j} \frac{(-1)^j}{j!} x^j,

        and :math:`w_j` is a weight given by

        .. math::

            w_j = \frac{x_j}{\left((J + 1) L_{J + 1}(x_j) \right)^2}.

        The integral can therefore be approximated by the sum

        .. math::

            \overline{\beta} \approx
            \sum_{j=1}^J w_j \beta\left(\frac{x_j}{2 z_Q}\right) x_j.



        The choice of :math:`J`, defined in this function as `n_lag`,
        will affect the accuracy of the approximation, with higher
        :math:`J` values leading to more accurate evaluation of the
        integrals.

        In this function the Laguerre weights and roots are found using
        :func:`numpy.polynomial.laguerre.laggauss` and the
        momentum-dependent reflection coefficient is found using
        :func:`snompy.sample.Sample.refl_coef_qs`.

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

        q = x_lag / np.asanyarray(2 * z_Q)

        beta_q = self.refl_coef_qs(q)

        beta_eff = np.sum(w_lag * x_lag * beta_q, axis=0)

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

            \int_0^{\infty} e^{-x} f(x) dx \approx \sum_{j=1}^J w_j f(x_j),

        where :math:`x_j` is the :math:`j^{th}` root of the Laguerre
        polynomial

        .. math::

            L_J(x) = \sum_{j=0}^{J} {J \choose j} \frac{(-1)^j}{j!} x^j,

        and :math:`w_j` is a weight given by

        .. math::

            w_j = \frac{x_j}{\left((J + 1) L_{J + 1}(x_j) \right)^2}.

        The integrals can therefore be approximated by the sums

        .. math::

            \begin{align*}
                \phi \rvert_{z=0}
                & \approx \frac{1}{2 z_Q}
                \sum_{j=1}^J w_j \beta\left(\frac{x_j}{2 z_Q}\right),
                \ \text{and}\\
                E_z \rvert_{z=0}
                & \approx \frac{1}{4 z_Q^2}
                \sum_{j=1}^J w_j \beta\left(\frac{x_j}{2 z_Q}\right) x_j.
            \end{align*}

        The choice of :math:`J`, defined in this function as `n_lag`,
        will affect the accuracy of the approximation, with higher
        :math:`J` values leading to more accurate evaluation of the
        integrals.

        In this function the Laguerre weights and roots are found using
        :func:`numpy.polynomial.laguerre.laggauss` and the
        momentum-dependent reflection coefficient is found using
        :func:`snompy.sample.Sample.refl_coef_qs`.

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

        q = x_lag / np.asanyarray(2 * z_Q)

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

            d_{image} = \left|
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
        equations (9) and (10) from reference [1]_. The depth
        :math:`d_{image}` is forced to be a real number by taking the
        absolute value of the :math:`\phi`-:math:`E_z` ratio, as described
        in reference [2]_.

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
    r"""Return an object representing a bulk sample with just a
    semi-infinite substrate and superstrate.

    Parameters
    ----------
    eps_sub : array_like
        Dielectric function of the semi-infinite substrate. Either
        `eps_sub` or `beta` must be None.
    beta : array_like
        Quasistatic  reflection coefficients of the interface between the
        substrate and superstrate. Either `eps_stack` or `beta_stack` must
        be None.
    eps_env : array_like
        Dielectric function of the environment.
    **kwargs : dict, optional
        Extra keyword arguments are passed to :func:`snompy.sample.Sample`.

    Returns
    -------
    sample : :class:`snompy.sample.Sample`
        Object representing the sample.

    """
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
    eps_i = np.asanyarray(eps_i)
    eps_j = np.asanyarray(eps_j)
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
        Permitivitty of material j.

    See also
    --------
    refl_coef_qs_single :
        The inverse of this function.

    """
    beta = np.asanyarray(beta)
    eps_i = np.asanyarray(eps_i)
    return eps_i * (1 + beta) / (1 - beta)


def lorentz_perm(nu_vac, nu_j, gamma_j, A_j=None, nu_plasma=None, f_j=1.0, eps_inf=1.0):
    """Return permittivity as a function of wavenumber using a single
    Lorentzian oscillator model.

    This function returns
     `eps_inf + A_j / (nu_j**2 - nu_vac**2 - 1j * gamma_j * nu_vac)`, where
     `A_j = f_j * nu_plasma**2`.

    Parameters
    ----------
    nu_vac : float
        Vacuuum wavenumber of incident light.
    nu_j : float
        Centre wavenumber of the oscillator.
    gamma_j : float
        Width, or damping frequency, of the oscillator (equivalent to the
        reciprocal of the relaxation time).
    A_j : float
        Amplitude of the oscillator, equivalent to `f_j * nu_plasma**2`. As
        this term accounts for the plasma frequency, either `A_j` or
        `nu_plasma` must be None.
    nu_plasma : float
        Plasma wavenumber (wavenumber corresponding to the plasma
        frequency) of the sample. Either `A_j` or `nu_plasma` must be None.
    f_j : float, default 1.0
        Dimensionless constant that modifies the oscillation amplitude when
        used in combination with `nu_plasma`.
    eps_inf : float, default 1.0
        High frequency permitivitty of the sample.

    Returns
    -------
    eps : complex
        Permitivitty.

    """
    if A_j is None:
        if nu_plasma is None:
            raise ValueError("`A_j` and `nu_plasma` cannot both be None")
        else:
            A_j = f_j * nu_plasma**2
    elif nu_plasma is not None:
        raise ValueError("Either `A_j` or `nu_plasma` must be None")

    return eps_inf + A_j / (nu_j**2 - nu_vac**2 - 1j * gamma_j * nu_vac)


def drude_perm(nu_vac, nu_plasma, gamma, eps_inf=1.0):
    """Return permittivity as a function of wavenumber using a Drude model.

    This function returns
     `eps_inf - nu_plasma**2 / (nu_vac**2 + 1j * gamma_j * nu_vac)`.

    Parameters
    ----------
    nu_vac : float
        Vacuuum wavenumber of incident light.
    nu_plasma : float
        Plasma wavenumber (wavenumber corresponding to the plasma
        frequency) of the sample.
    gamma : float
        Damping frequency of the sample (equivalent to the reciprocal of
        the relaxation time).
    eps_inf : float, default 1.0
        High frequency permitivitty of the sample.

    Returns
    -------
    eps : complex
        Permitivitty.

    """
    return lorentz_perm(
        nu_vac, nu_j=0, gamma_j=gamma, nu_plasma=nu_plasma, eps_inf=eps_inf
    )
