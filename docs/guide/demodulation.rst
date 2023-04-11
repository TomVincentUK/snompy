.. _demodulation:

Demodulation
============

Typically in a SNOM experiment, we can't measure :math:`\sigma_{scat}`
directly, because the total scattered electric field from the far-field
laser spot is much, much bigger than the electric field from the near
field (see the page :ref:`scattering` for the definition of
:math:`\sigma_{scat}`).

Instead, we oscillate the AFM tip height, :math:`z`,  at a frequency
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

This has the great advantage that the non-linear :math:`z`-dependence of
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
   calculating :math:`\alpha_{eff, n}`: :func:`pysnom.fdm.eff_pol_n_bulk`,
   :func:`pysnom.fdm.eff_pol_n_multi`, and
   :func:`pysnom.pdm.eff_pol_n_bulk`.

The undemodulated effective polarisability
------------------------------------------

As an example, lets take a look at the :math:`z`-dependence of
:math:`\alpha_{eff}` for a sample of bulk silicon (Si).

The following script plots the amplitude of :math:`\alpha_{eff}` for a
range of :math:`z` values from 0 to 200 nm.

.. tab-set::

   .. tab-item:: FDM

      .. plot:: guide/demodulation/eff_pol_0_fdm.py
         :align: center

   .. tab-item:: PDM

      .. plot:: guide/demodulation/eff_pol_0_pdm.py
         :align: center

This is the parameter we want to model, but we can't measure this directly
using SNOM.
We'll need to simulate a lock-in measurement if we want to compare our
models to experimental results.
Note that the decay of the effective polarisability is non-linear, which
will become important later.

.. hint::
   :class: dropdown

   In this section we show the real and imaginary parts of the effective
   polarisability, :math:`\Re(\alpha_{eff})` and :math:`\Im(\alpha_{eff})`,
   which makes it easier to visualise complex demodulation.
   However, in practice it's more common to study the amplitude and phase,
   (:math:`|\alpha_{eff}|`) and (:math:`\arg(\alpha_{eff})`).

Modulating the height of the AFM tip
------------------------------------

The first step in simulating the modulation and demodulation of a SNOM
signal will be to modulate the height of the AFM probe according to

.. math::
   :label: z_mod

   z(t) = z_0 + A_{tip} \left(1 + \cos(\omega_{tip}t)\right),

where :math:`z_0` is the bottom of the height oscillation, :math:`A_{tip}`
is the oscillation amplitude, and :math:`t` is time.

The following script shows how the effective polarisability responds to a
sinusoidal modulation of the tip height as described above:

.. tab-set::

   .. tab-item:: FDM

      .. plot:: guide/demodulation/modulated_fdm.py
         :align: center

   .. tab-item:: PDM

      .. plot:: guide/demodulation/modulated_pdm.py
         :align: center

This shows a very important result: thanks to the non-linear :math:`z`
decay, a sinusoidal modulation of :math:`z` leads to a periodic *but
non-sinusoidal* modulation of :math:`\alpha_{eff}`.

Fourier analysis
----------------

To understand demodulation, and how :math:`\alpha_{eff}` relates to
:math:`\alpha_{eff, n}`, it's helpful to analyse this signal in the
frequency domain.

As it's periodic but non-sinusoidal, :math:`\alpha_{eff}(t)` can be
described by a
`Fourier series <https://en.wikipedia.org/wiki/Fourier_series>`_,

.. math::
   :label: Fourier_series

   \alpha_{eff}(t) =
   \sum_{n=-\infty}^{\infty} \alpha_{eff, n} e^{i n \omega_{tip} t}.

This is a series of complex sinusoids with frequencies at multiples,
:math:`n`, of :math:`\omega_{tip}`.

.. hint::
   :class: dropdown

   Equation :eq:`Fourier_series` includes negative values of :math:`n`,
   which means it accounts for
   `negative frequencies <https://en.wikipedia.org/wiki/Fourier_transform#Negative_frequency>`_.
   Don't worry if this is confusing!
   For SNOM demodulation, we usually only need to worry about positive
   :math:`n` values.

   The negative frequency terms are needed to fully reconstruct complex
   signals (like :math:`\alpha_{eff}`).
   But, as we're only interested in extracting particular
   :math:`\alpha_{eff, n}` values, we can essentially ignore them here.

The values of :math:`\alpha_{eff, n}` are what we probe with SNOM, and they
take the form of complex-valued coefficients that multiply each sinusoid.
They modify the oscillations such that the :math:`n^{th}` sinusoid has
amplitude :math:`|\alpha_{eff, n}|`, and phase
:math:`\arg\left(\alpha_{eff, n}\right)`.

The following figure shows the modulated :math:`\alpha_{eff}(t)` signal
that we calculated above, along with the first few terms of equation
:eq:`Fourier_series`.

.. tab-set::

   .. tab-item:: FDM

      .. plot:: guide/demodulation/Fourier_fdm.py
         :align: center
         :include-source: False

   .. tab-item:: PDM

      .. plot:: guide/demodulation/Fourier_pdm.py
         :align: center
         :include-source: False

We can see that the :math:`n=0` term accounts for the DC offset, and that
the amplitudes of the following terms drop off quickly with increasing
:math:`n`.

Note that if the :math:`z` decay of :math:`\alpha_{eff}` was linear the
sinusoidal :math:`z` modulation would create a purely sinusoidal
:math:`\alpha_{eff}` modulation, which would mean only the :math:`n=0` and
:math:`n=1` terms would remain in the signal.

Extracting Fourier coefficients
-------------------------------

Once we've modulated the effective polarisability by changing the
tip height, the next step is to demodulate the resulting signal to extract
the desired Fourier coeficients, :math:`\alpha_{eff, n}`.

In a lock-in amplifier, this is done by multiplying the input signal by a
complex oscillator
:math:`\left(e^{i n \omega t} = \cos(n \omega t) + i \sin(n \omega t)\right)`
synced to the desired harmonic of the tapping frequency, then low-pass
filtering the product to remove all but the DC offset.
We can simulate this with the integral

.. math::
   :label: Fourier_integral_inf

   \alpha_{eff, n} =
   \int_{-\infty}^{\infty}
   \alpha_{eff}(t)
   e^{i n \omega_{tip} t}
   dt
   = \int_{-\frac{1}{2 \omega_{tip}}}^{\frac{1}{2 \omega_{tip}}}
   \alpha_{eff}(t)
   e^{i n \omega_{tip} t}
   dt,

(which takes advantage of the fact that :math:`\alpha_{eff}(t)` is periodic
in :math:`\omega_{tip}`).

This can be simplified further by noting that the result is independent of
frequency, so we can set :math:`\omega_{tip}=1`.
The resulting integral then becomes

.. math::
   :label: Fourier_integral

   \alpha_{eff, n} =
   \int_{-\pi}^{\pi}
   \alpha_{eff}(\theta)
   e^{i n \theta}
   d\theta,

which can then be evaluated numerically using a method such as the
`trapezium rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_, as
shown in the example script below.

.. tab-set::

   .. tab-item:: FDM

      .. plot:: guide/demodulation/integral_fdm.py
         :align: center

   .. tab-item:: PDM

      .. plot:: guide/demodulation/integral_pdm.py
         :align: center

Putting it all together
-----------------------

In the sections above, we showed how to simulate a lock-in measurement, by
modulating a signal, then demodulating it to find the :math:`n^{th}`
Fourier coefficient.
If you're worried that it seems like a lot of work, that's because it is!

Thankfully ``pysnom`` has a built-in function
:func:`pysnom.demodulate.demod`, which takes care of all the tricky parts.
It's also vectorised, which means it can simulate demodulation on whole
arrays of data at once, with no need for looping.

Additionally, ``pysnom`` provides functions with built-in demodulation for
directly calculating :math:`\alpha_{eff, n}`:
:func:`pysnom.fdm.eff_pol_n_bulk`, :func:`pysnom.fdm.eff_pol_n_multi`, and
:func:`pysnom.pdm.eff_pol_n_bulk`.
These should be even simpler to use.

The script below shows the use of both to calculate approach curves for
several harmonics at once.

.. tab-set::

   .. tab-item:: FDM

      .. plot:: guide/demodulation/approach_fdm.py
         :align: center

   .. tab-item:: PDM

      .. plot:: guide/demodulation/approach_pdm.py
         :align: center

This shows that both methods produce exactly the same results, and also
that higher order demodulation leads to a faster decay of the SNOM signal
(*i.e.* stronger surface confinement).

.. hint::
   :class: dropdown

   In the script above, the `z` value is offset by `tapping_amplitude` for
   the approach curve calculated using :func:`pysnom.demodulate.demod`.
   That's because the definition for the AFM oscillation, as given in
   equation :eq:`z_mod`, is set so that the tip just barely contacts the
   sample at `z = 0`.
   For :func:`pysnom.demodulate.demod`, you need to specify the *centre* of
   the oscillation, not the bottom.

   This conversion is taken care of automatically by ``pysnom``'s functions
   with built-in demodulation, which is why the `z` value isn't offset for
   the approach curve calculated using :func:`pysnom.fdm.eff_pol_n_bulk` or
   :func:`pysnom.pdm.eff_pol_n_bulk`.
