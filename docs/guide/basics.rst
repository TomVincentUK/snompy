Basics of SNOM modelling
========================

To model contrast in scanning near-field optical microscopy (SNOM)
experiments, ``pysnom`` provides two models called the finite dipole model
(FDM) and the point dipole model (PDM).
We'll explain how each model works on the following pages, but first we'll
cover some basics of SNOM modelling: `Effective polarisability`_ and
`Demodulation`_.
These will be useful to understand both models.


Effective polarisability
------------------------

In both the FDM and the PDM, the SNOM contrast is modelled by calculating
the effective polarisability, :math:`\alpha_{eff}`, of an atomic force
microscope (AFM) tip and sample.
In this section we'll explain why that works.

The image below shows a typical scattering SNOM experiment, in which we
illuminate an AFM tip and sample with far-field light whose electric field
we can call :math:`E_{in}`.
This excites a near field at the apex of the AFM tip, which interacts with
the sample and scatters light with electric field :math:`E_{scat}` back
into the far-field.

.. image:: diagrams/tip_sample.svg
   :align: center

The near-field information in the sample is contained in the scattering
coefficient :math:`\sigma_{scat}`, which relates the near-field scattered
light to the incident light as

.. math::
   :label: scatter_def

   \sigma_{scat} = \frac{E_{scat}}{E_{in}}.

When the incident light falls on the tip and sample, the electric field
induces a polarisation of the charges inside them, and a consequent dipole
moment.
The electric fields induced in the tip and sample interact, so the tip and
sample couple together to produce a combined response to the external
field.
The strength of their dipole moment, relative to the incident field, is
given by the effective polarisability of the tip and sample,
:math:`\alpha_{eff}`.

It's this dipole which radiates near-field light back into the far field,
so the scattering coefficient can therefore be found from

.. math::
   :label: scatter_from_eff_pol

   \sigma_{scat} = (1 + c r)^2 \alpha_{eff},

where :math:`r` is the far-field reflection coefficient, and :math:`c` is
an empirical constant which can be used to compensate for differences
between particular experimental setups.
The :math:`(1 + c r)^2` term is included because the AFM tip is illuminated
both directly, and also by reflections from the sample surface, as shown in
the diagram above.

.. hint::

   It's common to assume that the :math:`(1 + c r)^2` term will be constant
   throughout a SNOM experiment, because the area of the far-field laser
   spot is so much bigger than the near-field-confined area probed by SNOM,
   so it's often neglected in analysis.
   However, there are many occasions where the far-field reflection
   coefficient *does* have a significant affect on results, particularly
   near large features or on cluttered substrates [1]_.
   *Don't neglect it without thinking!*


SNOM experiments are typically sensitive to not just the amplitude but also
the phase of the scattered light, relative to the incident light.
Because of this, :math:`\sigma_{scat}` takes the form of a complex number with
amplitude, :math:`s`, and phase, :math:`\phi`, given by

.. math::
   :label: amp_and_phase

   \begin{align*}
      s &= |\sigma_{scat}|, \ \text{and}\\
      \phi &= \arg(\sigma_{scat}).
   \end{align*}

In ``pysnom``, the effective polarisability is provided by the functions
:func:`pysnom.fdm.eff_pol_0_bulk`, :func:`pysnom.fdm.eff_pol_0_multi`, and
:func:`pysnom.pdm.eff_pol_0_bulk`.
However it's common to use high-harmonic-demodulated versions of these
functions, :math:`\alpha_{eff, n}`,  instead of the raw
:math:`\alpha_{eff}`.
The following section will explain why.

Demodulation
------------

Typically in a SNOM experiment, we can't measure :math:`\sigma_{scat}`
directly, because the total scattered electric field from the far-field
laser spot is much, much bigger than the electric field from the near
field.

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
given, not by equation :eq:`scatter_from_eff_pol`, but by

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
The rest of this section will take you through how demodulation is
implemented in ``pysnom``.

.. hint::

   It might seem like there are lots of steps, but don't worry!
   The final section of this page, `Putting it all together`_, will
   introduce the function :func:`pysnom.demodulate.demod`, which
   automatically takes care of all the tricky details for you.

   In fact, as it's such in integral part of simulating SNOM experiments,
   ``pysnom`` provides functions with built-in demodulation for directly
   calculating :math:`\alpha_{eff, n}`: :func:`pysnom.fdm.eff_pol_bulk`,
   :func:`pysnom.fdm.eff_pol_multi`, and :func:`pysnom.pdm.eff_pol_bulk`.

The undemodulated effective polarisability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, lets take a look at the :math:`z`-dependence of
:math:`\alpha_{eff}` for a sample of bulk silicon (Si), calculated using
the FDM.

The following script plots the amplitude of :math:`\alpha_{eff}` for a
range of :math:`z` values from 0 to 200 nm.

.. plot:: guide/plots/basics_eff_pol_0.py
   :align: center

This is the parameter we want to model, but we can't measure this directly
using SNOM.
We'll need to simulate a lock-in measurement if we want to compare our
models to experimental results.
Note that the decay of the effective polarisability is non-linear, which
will become important later.

.. hint::

   In this section we show only the real part of effective polarisability,
   :math:`\Re(\alpha_{eff})`, which makes it easier to visualise complex
   demodulation.
   However, in practice it's more common to study the amplitude
   (:math:`|\alpha_{eff}|`) or phase (:math:`\arg(\alpha_{eff})`).

Modulating the height of the AFM tip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step in simulating the modulation and demodulation of a SNOM
signal will be to modulate the height of the AFM probe according to

.. math::
   :label: z_mod

   z(t) = z_0 + A_{tip} \left(1 + \cos(\omega_{tip}t)\right),

where :math:`z_0` is the bottom of the height oscillation, :math:`A_{tip}`
is the oscillation amplitude, and :math:`t` is time.

The following script shows how the effective polarisability responds to a
sinusoidal modulation of the tip height as described above:

.. plot:: guide/plots/basics_modulated.py
   :align: center

This shows a very important result: thanks to the non-linear :math:`z`
decay, a sinusoidal modulation of :math:`z` leads to a periodic *but
non-sinusoidal* modulation of :math:`\alpha_{eff}`.

Fourier analysis
^^^^^^^^^^^^^^^^

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

The values of :math:`\alpha_{eff, n}` are what we probe with SNOM, and they
take the form of complex-valued coefficients that multiply each sinusoid.
They modify the oscillations such that the :math:`n^{th}` sinusoid has
amplitude :math:`|\alpha_{eff, n}|`, and phase
:math:`\arg\left(\alpha_{eff, n}\right)`.

The following figure shows the modulated :math:`\alpha_{eff}(t)` signal
that we calculated above, along with the first few terms of equation
:eq:`Fourier_series`.

.. plot:: guide/plots/basics_Fourier.py
   :align: center
   :include-source: False

We can see that the :math:`n=0` term accounts for the DC offset, and that
the amplitudes of the following terms drop off quickly with increasing
:math:`n`.

Note that if the :math:`z` decay of :math:`\alpha_{eff}` was linear the
sinusoidal :math:`z` modulation would create a purely sinusoidal
:math:`\alpha_{eff}` modulation, which would mean only the :math:`n=0` and
:math:`n=1` terms would remain in the signal.

.. hint::

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

Extracting Fourier coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we've modulated the effective polarisability by changing the
tip height, the next step is to demodulate the resulting signal to extract
the desired Fourier coeficients, :math:`\alpha_{eff, n}`.

In a lock-in amplifier, this is done by multiplying the input signal by a
complex oscillator (:math:`e^{i n \omega t}`), synced to the desired
harmonic of the tapping frequency, then low-pass filtering the product to
remove all but the DC offset.
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

.. plot:: guide/plots/basics_integral.py
   :align: center

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

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
:func:`pysnom.fdm.eff_pol_bulk`, :func:`pysnom.fdm.eff_pol_multi`, and
:func:`pysnom.pdm.eff_pol_bulk`.
These should be even simpler to use.

The script below shows the use of both to calculate approach curves for
several harmonics at once.

.. plot:: guide/plots/basics_approach.py
   :align: center

This shows that both methods produce exactly the same results, and also
that higher order demodulation leads to a faster decay of the SNOM signal
(*i.e.* stronger surface confinement).

.. hint::

   In the script above, the `z` value is offset by `tapping amplitude` for
   the approach curve calculated using :func:`pysnom.demodulate.demod`.
   That's because the definition for the AFM oscillation, as given in
   equation :eq:`z_mod`, is set so that the tip just barely contacts the
   sample at `z = 0`.
   For :func:`pysnom.demodulate.demod`, you need to specify the *centre* of
   the oscillation, not the bottom.

   This conversion is taken care of automatically by ``pysnom``'s functions
   with built-in demodulation, which is why the `z` value isn't offset for
   the approach curve calculated using :func:`pysnom.fdm.eff_pol_bulk`.

References
----------
.. [1] L. Mester, A. A. Govyadinov, and R. Hillenbrand, “High-fidelity
   nano-FTIR spectroscopy by on-pixel normalization of signal harmonics,”
   Nanophotonics, vol. 11, no. 2, p. 377, 2022, doi:
   10.1515/nanoph-2021-0565.