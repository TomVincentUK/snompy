.. _demodulation:

Demodulation
============

Typically in a SNOM experiment, we can't measure :math:`\sigma_{scat}`
directly, because the total scattered electric field from the far-field
laser spot is much, much bigger than the electric field from the near
field (see the page :ref:`scattering` for the definition of
:math:`\sigma_{scat}`).

Instead, we oscillate the AFM tip height, :math:`z_{tip}`,  at a frequency
:math:`\omega_{tip}`, then use a
`lock-in amplifier <https://en.wikipedia.org/wiki/Lock-in_amplifier>`_ to
demodulate the total detected signal at higher harmonics of that frequency,
:math:`n \omega_{tip}` (where :math:`n = 2, 3, 4, \ldots`).
This oscillation modulates the near field interaction, but mostly leaves
the far field unchanged, so the lock-in can extract the near-field part of
the signal by looking for only parts of the signal that change with the
right frequency.

The lock-in-demodulated signals that we actually detect are therefore
given, not by :math:`\sigma_{scat}`, but by

.. math::
   :label: scatter_from_eff_pol_n

   \sigma_{scat, n} = (1 + c r)^2 \alpha_{eff, n},

with amplitude and phase

.. math::
   :label: amp_and_phase_n

   \begin{align*}
      s_n &= |\sigma_{scat, n}|, \ \text{and}\\
      \phi_n &= \arg(\sigma_{scat, n}),
   \end{align*}

where :math:`\alpha_{eff, n}` is the :math:`n^{th}` Fourier coefficient of
:math:`\alpha_{eff}` (for a particular tip height and tapping amplitude).

This has the great advantage that the non-linear :math:`z_{tip}`-dependence of
the near-field interaction means we can achieve a greater near-field
confinement by choosing a higher value of :math:`n`.

Demodulation is an integral part of a SNOM experiment, so we need to
account for it in our modelling if we want accurate results.
The rest of this page will take you through how demodulation is
implemented in ``pysnom``.
The process for demodulation is exactly the same for both the finite dipole
model (FDM) and point dipole model (PDM), so the examples here can be
switched between the two models using the tabs on this page.

.. hint::
   :class: dropdown

   It might seem like there are lots of steps, but don't worry!
   The final section of this page, `Putting it all together`_, will
   introduce the function :func:`pysnom.demodulate.demod`, which
   automatically takes care of all the tricky details for you.

   In fact, as it's such in integral part of simulating SNOM experiments,
   ``pysnom`` provides functions with built-in demodulation for directly
   calculating :math:`\alpha_{eff, n}`: :func:`pysnom.fdm.eff_pol_n`,
   :func:`pysnom.fdm.eff_pol_n`, and
   :func:`pysnom.pdm.eff_pol_n`.