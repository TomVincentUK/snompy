"""
Sample properties (:mod:`pysnom.sample`)
========================================

.. currentmodule:: pysnom.sample

This module provides a class to represent layered and bulk samples within
``pysnom``, and functions for converting between reflection coefficients
and permitivitties.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Sample
    refl_coef_qs_single
    permitivitty
"""
import warnings

import numpy as np


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
        Dielectric function of the enviroment (equivalent to
        `eps_stack[0]`). This is used to calculate `eps_stack` from
        `beta_stack` if needed.
    k_vac : array_like
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
    multilayer:
        True if sample has one or more finite-thickness layer sandwiched
        between the interfaces in the stack, False for bulk samples.
    eps_env : array_like
        Dielectric function of the enviroment (equivalent to
        `eps_stack[0]`).
    """

    def __init__(
        self, eps_stack=None, beta_stack=None, t_stack=None, eps_env=1 + 0j, k_vac=None
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
        self._eps_env = eps_env

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
            Quasistatic  reflection coefficient of the sample.
        """
        beta_total = self.beta_stack[0] * np.ones_like(q)
        for i in range(self.t_stack.shape[0]):
            layer_decay = np.exp(-2 * q * self.t_stack[i])
            beta_total = (beta_total + self.beta_stack[i + 1] * layer_decay) / (
                1 + beta_total * self.beta_stack[i + 1] * layer_decay
            )
        return beta_total

    def transfer_matrix(self, theta_in=None, q=None, k_vac=None, polarization="p"):
        # Default to self.k_vac if k_vac is None.
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

        # Transmission matrix depends on polarisation
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
        M = self.transfer_matrix(
            theta_in=theta_in, q=q, k_vac=k_vac, polarization=polarization
        )
        return M[..., 1, 0] / M[..., 0, 0]

    def trans_coef(self, theta_in=None, q=None, k_vac=None, polarization="p"):
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


def bulk_sample(eps_sub=None, beta=None, eps_env=1 + 0j, **kwargs):
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
        Dielectric function of material i.

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
