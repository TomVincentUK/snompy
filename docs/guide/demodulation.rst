.. _demodulation:

Demodulation
============

SNOM relies heavily on `lock-in amplifier <https://en.wikipedia.org/wiki/Lock-in_amplifier>`_ demodulation of various :math:`z_{tip}` -dependent signals at different harmonics, :math:`n`, of an AFM tip's tapping frequency.
For this reason ``snompy`` features code to simulate lock-in measurements of arbitrary functions.
In most cases, you can ignore this, and rely on versions of the ``snompy`` functions that have this functionality built-in.
For example :func:`snompy.fdm.eff_pol_n` is a modulated and demodulated version of :func:`snompy.fdm.eff_pol`, so you don't need to worry about simulating the demodulation in your own code.

This section describes how lock-in measurements are simulated in ``snompy``, and also how you can use the demodulation functionality on arbitrary functions that aren't included in the package.

Lock-in amplifier simulation
----------------------------

A lock-in amplifier extracts a periodic signal with a known frequency from a noisy background.
It works by multiplying the input signal by a sinusoidal reference signal of the desired frequency, then using a low-pass filter to remove unwanted high-frequency components.
This effectively suppresses all frequency components except for the component which is at the same frequency and in-phase-with the reference signal.
In modern lock-ins, two orthogonal reference signals are usually used in order to measure both the amplitude and phase of the desired signal.

To simulate this, we first need to convert a non-periodic function :math:`g(x)` into a periodic function :math:`f(\theta)`.
We apply a modulation as:

.. math::
    f(\theta) = g\left(x_0 + A_x \cos(\theta)\right)

We can then simulate demodulation of :math:`f(\theta)` at different harmonics, :math:`n`, by multiplying it by a complex sinusoid (:math:`e^{-i \theta n}`, whose real and imaginary components are orthogonal) then integrating over a single period of the oscillation as:

.. math::

   \frac{1}{2 \pi} \int_{-\pi}^{\pi} f(\theta) e^{-i \theta n} d{\theta}

In ``snompy`` this process is implemented by the function :func:`snompy.demodulate.demod`.

Adapting the process for AFM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For AFM demodulation we apply a slight variation of the general approach above.
To simulate a periodic modulation of :math:`z_{tip}` with tapping amplitude :math:`A_{tip}`, we apply a modulation like:

.. math::
    f(\theta) = g\left(z_{tip} + A_{tip} \left(1+\cos(\theta)\right)\right)

Note that the cosine term here has been offset so that the oscillation is centred on :math:`z_{tip}+A_{tip}`, rather than on :math:`z_{tip}` itself.
This is to ensure that when :math:`z_{tip}=0` the lowest part of the oscillation never becomes negative, which would mean intersecting with the sample surface.