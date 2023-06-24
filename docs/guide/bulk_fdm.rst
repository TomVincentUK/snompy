.. _bulk_fdm:

Bulk finite dipole model
========================

The finite dipole model (FDM) is one method for estimating the effective
polarisability of an atomic force microscope (AFM) tip and a sample.
This can be used to predict scattering in scanning near-field optical
microscopy (SNOM) measurements, as described on the page :ref:`scattering`.

On this page we introduce the FDM for bulk samples and give examples
showing how the bulk FDM is implemented in ``pysnom``.

Ellipsoid model of an AFM tip
-----------------------------

The first step in the FDM is to represent the AFM tip by a perfectly
conducting ellipsoid.
This is a good approximation for the elongated probe at the tip's end, and
there is an analytical solution for its response to an electrical field.

The electric field of the incident light will vary with time, :math:`t`,
and frequency, :math:`\omega`, as :math:`E \propto E_{in} e^{i \omega t}`.
But we can make the quasistatic approximation, which assumes electric
fields change much slower than the time needed for charges in the system to
reach equilibrium.
That means we can represent the incident light by a static, vertically
oriented electric field :math:`E_{in}`.

Let's first consider the response of an isolated ellipsoid (with no sample
surface nearby) to a vertical electric field.
The image below shows the vertical component of the electric field response
of an ellipsoid to a vertical :math:`E_{in}`.

.. image:: bulk_fdm/dipole_field.svg
   :align: center

We can see that the resulting field looks like that of a dipole formed by
two charges, :math:`Q_0` and :math:`-Q_0`, close to the ends of the
ellipsoid [1]_.

This dipole, which we call :math:`p_0`, is what gives the finite dipole
model its name.
The word finite here refers to the fact that the dipole has a finite
length, and is used to contrast with the *point* dipole model
(:ref:`PDM <bulk_pdm>`), an earlier model for the effective polarisability.
The position of the two charges are found at distances

.. math::
   :label: z_q0_pos

   d_{Q0} \approx \frac{1.31 L_{tip} r_{tip}}{L_{tip} + 2 r_{tip}}

from the ends of the ellipsoid, where :math:`r_{tip}` is the radius of curvature
at the pointy end, and :math:`L_{tip}` is the semi-major axis length (the
distance from the ellipsoid centre to the pointy end).

The strength of the electric dipole moment can be related to the charges
and their separation as

.. math::
   :label: p_0

   p_0 = 2 (L_{tip} - d_{Q0}) Q_0 \quad (\approx 2 L_{tip} Q_0, \ \mathrm{for} \ r_{tip} \ll L_{tip}).

Tip-sample interaction
----------------------

Now let's consider what happens when we move our model AFM tip close to a
sample's surface, with a tip-sample separation of :math:`z_{tip}`.

When they are close enough, charges induced in the ellipsoid will interact
with the sample.
But due to the elongated shape, we can make the approximation that this
only happens for the lower of the two charges.
The other charge is too far away.
This means that we can neglect the effect of the :math:`-Q_0` charge, and
model the sample response as the response to a single point charge
:math:`Q_0` [1]_.

The image below shows the various induced charges, counter charges, and
image charges which are used in the FDM to model the tip-sample
interaction.

.. image:: bulk_fdm/fdm.svg
   :align: center

We can model the electric field response of the sample to the charge
:math:`Q_0`, at a height of :math:`z_{tip} + d_{Q0}`, using
the
`method of image charges <https://en.wikipedia.org/wiki/Method_of_image_charges>`_.

This means we can add a fictitious image charge, :math:`Q'_0 = -\beta Q_0`,
at a depth of :math:`z_{tip} + d_{Q0}` below the surface.
Here, :math:`\beta` is the electrostatic reflection coefficient of the
surface, given by

.. math::
   :label: beta

   \beta =
   \frac{\varepsilon_{sub} - \varepsilon_{env}}
   {\varepsilon_{sub} + \varepsilon_{env}},

where :math:`\varepsilon_{env}` is the dielectric function of the
environment (:math:`\varepsilon_{env} = 1` for air or vacuum), and
:math:`\varepsilon_{sub}` is the dielectric function of the sample (the
subscript "sub" here is short for substrate).
In ``pysnom``, equation :eq:`beta` is provided by the function
:func:`pysnom.reflection.refl_coeff`.

The charge :math:`Q'_0` acts back on the tip and induces a further
polarisation, which we can model as another point charge :math:`Q_1`, at a
distance :math:`d_{Q1} \approx r_{tip} / 2` away from the end of the tip.

.. hint::
   :class: toggle

   Modelling the response of the tip to :math:`Q'_0` as a single point
   charge is just an approximation.
   In reality, the polarisation induced in the tip has a complicated charge
   distribution which is quite tricky to calculate [1]_.
   But replacing that distribution with a single, representative point
   charge allows us to solve the electrostatic equations, and gives a model
   that matches well to experimental results.

With the addition of :math:`Q_1`, we need to add some more charges to our
model:
the sample response to :math:`Q_1` can be represented by another image
charge, :math:`Q'_1 = \beta Q_1`, at a depth of :math:`z_{tip} + d_{Q1}` below
the surface;
and, for conservation of charge within the tip, :math:`Q_1` must have a
counter charge :math:`-Q_1`, which is situated in the centre of the
ellipsoid.

The value of :math:`Q_1` can be solved for by accounting for contributions
to the overall polarisation from :math:`Q_0` and also from :math:`Q_1`
itself [2]_, as

.. math::
   :label: q_1

   Q_1 = \beta (f_0 Q_0 + f_1 Q_1)

(neglecting the influence of the :math:`-Q_1` charge as it's far from the
sample).

Here, the parameters :math:`f_i` account for the geometrical features of
the tip, and the positions of the charges within them.
They are given by the formula

.. math::
   :label: f_i_bulk

   f_i = \left(g - \frac{r_{tip} + 2 z_{tip} + d_{Qi}}{2 L_{tip}} \right)
   \frac{\ln\left(\frac{4 L_{tip}}{r_{tip} + 4 z_{tip} + 2 d_{Qi}}\right)}
   {\ln\left(\frac{4 L_{tip}}{r_{tip}}\right)},

where :math:`g \approx 0.7` is an empirical factor that describes how much
of the induced charge is relevant for the near-field interaction (see
`Parameters`_ for more details on how this factor affects the results).
In ``pysnom``, equation :eq:`f_i_bulk` is provided by the function
:func:`pysnom.fdm.geom_func`.

The charges :math:`Q_1` and :math:`-Q_1` form another dipole

.. math::
   :label: p_1

   p_1 = (L_{tip} - d_{Q1}) Q_1 \quad (\approx L_{tip} Q_1, \ \mathrm{for} \ r_{tip} \ll L_{tip}).

The effective polarisability of the tip and sample can then be found from
the total induced dipole, as

.. math::
   :label: eff_pol_bulk_fdm

   \alpha_{eff}
   = \frac{p_0 + p_1}{E_{in}}
   \approx \frac{2 L_{tip} Q_0}{E_{in}}
   \left(1 + \frac{f_0 \beta}{2 (1 - f_1 \beta)}\right)
   \propto 1 + \frac{f_0 \beta}{2 (1 - f_1 \beta)}.

In ``pysnom``, equation :eq:`eff_pol_bulk_fdm` is provided by the function
:func:`pysnom.fdm.eff_pol`.

Demodulating the FDM
--------------------

Typically we're not interested in the raw effective polarisability, but in
the :math:`n_{th}`-harmonic-demodulated effective polarisability
:math:`\alpha_{eff, n}`.
That's because the signals measured in real SNOM experiments are determined
by the demodulated near-field scattering coefficient

.. math::
   :label: fdm_scattering

   \sigma_{scat, n} \propto \alpha_{eff, n},

with amplitude and phase

.. math::
   :label: fdm_amp_and_phase

   \begin{align*}
      s_n &= |\sigma_{scat, n}|, \ \text{and}\\
      \phi_n &= \arg(\sigma_{scat, n}).
   \end{align*}

This is explained in detail on the dedicated page :ref:`demodulation`.

In ``pysnom``, :math:`\alpha_{eff, n}` for bulk FDM is provided by the
function :func:`pysnom.fdm.eff_pol_n`.

Using pysnom for bulk FDM
-------------------------

In this section we'll show how the bulk FDM can be used in ``pysnom`` by
simulating an approach curve from bulk silicon (Si) in a few different
ways.

.. hint::
   :class: toggle

   An approach curve is a type of AFM measurement where values are recorded
   while the tip is moved towards the sample surface, typically until the
   two make contact.

   The same data can be acquired by a retraction curve, which moves the tip
   *away* from the sample, though the term approach curve is often used to
   refer to either type of measurement.

Initial setup
^^^^^^^^^^^^^

To begin with, let's import the libraries that we'll need, set the
:math:`z_{tip}` values for our approach curves, and set up some axes that we can
plot our results in.
For :math:`z_{tip}`, we'll set a range of points from 0 to 100 nm.

We'll do all the calculations in `SI base units <https://en.wikipedia.org/wiki/SI_base_unit>`_,
but we can also plot :math:`z_{tip}` in nm to make our figure tidier.

.. plot::
   :context:
   :caption: An empty set of axes.
   :alt: An empty set of axes.

   import matplotlib.pyplot as plt
   import numpy as np

   import pysnom

   # Define an approach curve on Si
   z_nm = np.linspace(0, 100, 512)  # Useful for plotting
   z_tip = z_nm * 1e-9  # Convert to nm to m (we'll work in SI base units)

   # Set up an axis for plotting
   fig, ax = plt.subplots()
   ax.set(
      xlabel=r"$z_{tip}$ / nm",
      xlim=(z_nm.min(), z_nm.max()),
      ylabel=r"$\frac{\alpha_{eff, \ n}}{(\alpha_{eff, \ n})|_{z_{tip} = 0}}$",
   )
   fig.tight_layout()

Using dielectric function
^^^^^^^^^^^^^^^^^^^^^^^^^

Now let's create an approach curve to display in these axes.
We'll use :func:`pysnom.fdm.eff_pol_n` to calculate the effective
polarisability.

We need to tell the function our tip height :math:`z_{tip}`, the tapping
amplitude :math:`A_{tip}` (see :ref:`demodulation` for details on this
parameter), the demodulation harmonic :math:`n`, and some way of specifying
the sample's response to light (in this first example we'll use
:math:`\varepsilon_{sub}`).
These arguments are called `z_tip`, `A_tip`, `n`, and
`eps_samp`.

Let's use :math:`A_{tip} = 25` nm, :math:`n = 2`, and
:math:`\varepsilon_{sub} = 11.7` (the mid-IR dielectric function of Si) [3]_ to
calculate our first approach curve.

.. plot::
   :context:
   :caption: An approach curve from Si, calculated from the dielectric function.
   :alt: An approach curve from Si, calculated from the dielectric function.

   # Set the parameters for our first approach curve
   A_tip = 25e-9
   single_harmonic = 2
   eps_samp = 11.7  # The mid-IR dielectric function of Si


   # Calculate an approach curve using the dielectric function
   alpha_eff_0 = pysnom.fdm.eff_pol_n(
      z_tip=z_tip,
      A_tip=A_tip,
      n=single_harmonic,
      eps_samp=eps_samp,
   )
   alpha_eff_0 /= alpha_eff_0[0]  # Normalise to z_tip = 0

   # Add the approach curve to the figure
   ax.plot(
      z_nm,
      np.abs(alpha_eff_0),
      label=r"Default parameters (via $\varepsilon$), $n = " f"{single_harmonic}" r"$",
   )
   ax.legend()

This shows the expected response, that the effective polarisability decays
with distance from the sample.

Using reflection coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it's easier to specify the sample's response as a reflection
coefficient :math:`\beta`, instead of a dielectric function
:math:`\varepsilon_{sub}`.
In :func:`pysnom.fdm.eff_pol_n`, we can do this by using the argument
`beta` instead of `eps_samp`.

To calculate the reflection coefficient of Si, we'll use the function
:func:`pysnom.reflection.refl_coeff`, and assume that our environment has a
dielectric function of 1 (for air or vacuum).

We should expect to see exactly the same approach curve here that we
calculated before, so we'll draw the new curve with a dashed line so we can
still see the original plot.

.. plot::
   :context:
   :caption: Add a second approach curve calculated from the reflection coefficient.
   :alt: Add a second approach curve calculated from the reflection coefficient.

   # Calculate reflection coefficient from the Si dielectric function
   beta = pysnom.reflection.refl_coeff(1, eps_samp)

   # Calculate an approach curve using the reflection coefficient
   alpha_eff_1 = pysnom.fdm.eff_pol_n(
      z_tip=z_tip,
      A_tip=A_tip,
      n=single_harmonic,
      beta=beta,
   )
   alpha_eff_1 /= alpha_eff_1[0]  # Normalise to z_tip = 0

   # Add the new approach curve to the figure
   ax.plot(
      z_nm,
      np.abs(alpha_eff_1),
      label=r"Default parameters (via $\beta$), $n = " f"{single_harmonic}" r"$",
      ls="--",
   )
   ax.legend()  # Update the legend

As we expected, both lines overlap nicely, which shows that specifying the
material response via :math:`\varepsilon` and :math:`\beta` are equivalent.

Changing the default parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above examples, we didn't specify parameters like the radius
:math:`r_{tip}` or semi-major axis length :math:`L_{tip}` of the ellipsoid, or the
empirical factor :math:`g`, so the function reverted to its default values
(see :func:`pysnom.fdm.eff_pol_n` for the values of these defaults).

Lets add a new approach curve with a different set of tip parameters.

.. plot::
   :context:
   :caption: Add an approach curve with changes to the default parameters.
   :alt: Add an approach curve with changes to the default parameters.

   # Updates to the default parameters
   r_tip = 100e-9
   L_tip = 400e-9
   g_factor = 0.7

   # Calculate an approach curve with the updated parameters
   alpha_eff_2 = pysnom.fdm.eff_pol_n(
      z_tip=z_tip,
      A_tip=A_tip,
      n=single_harmonic,
      eps_samp=eps_samp,
      r_tip=r_tip,
      L_tip=L_tip,
      g_factor=g_factor,
   )
   alpha_eff_2 /= alpha_eff_2[0]  # Normalise to z_tip = 0

   # Add the new approach curve to the figure
   ax.plot(
      z_nm,
      np.abs(alpha_eff_2),
      label=r"Custom parameters (via $\varepsilon$), $n = " f"{single_harmonic}" r"$",
      ls=":",
   )
   ax.legend()  # Update the legend

In this case, we see a new, distinct shape for the approach curve thanks to
the different tip parameters.

Taking advantage of array broadcasting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Where possible, ``pysnom`` uses ``numpy``-style
`array broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
This means multiple parameters can be varied at once, by providing arrays
with different shapes as inputs.

Lets take advantage of that to calculate several new approach curves at
once, for some more harmonics using our custom parameters.

.. plot::
   :context:

   # Create a range of harmonics
   multiple_harmonics = np.arange(3, 6)

   # Calculate several approach curves at once using array broadcasting
   alpha_eff_3 = pysnom.fdm.eff_pol_n(
      z_tip=z_tip[:, np.newaxis],  # newaxis added for array broadcasting
      A_tip=A_tip,
      n=multiple_harmonics,
      eps_samp=eps_samp,
      r_tip=r_tip,
      L_tip=L_tip,
      g_factor=g_factor,
   )
   alpha_eff_3 /= alpha_eff_3[0]  # Normalise to z_tip = 0

   ax.plot(
      z_nm,
      np.abs(alpha_eff_3),
      label=[
         r"Custom parameters (via $\varepsilon$), $n = " f"{n}" r"$"
         for n in multiple_harmonics
      ],  # list of labels (one per harmonic)
      ls=":",
   )
   ax.legend()  # Update the legend

This shows another key result for SNOM experiments: that higher harmonics
decay faster with distance than lower ones, which means they have a higher
surface sensitivity.

Parameters
----------

[Explanations of parameters (perhaps with graphs)?]

References
----------
.. [1] A. Cvitkovic, N. Ocelic, and R. Hillenbrand, “Analytical model for
   quantitative prediction of material contrasts in scattering-type
   near-field optical microscopy,” Opt. Express, vol. 15, no. 14, p. 8550,
   2007, doi: 10.1364/oe.15.008550.
.. [2] B. Hauer, A. P. Engelhardt, and T. Taubner, “Quasi-analytical model
   for scattering infrared near-field microscopy on layered systems,” Opt.
   Express, vol. 20, no. 12, p. 13173, Jun. 2012,
   doi: 10.1364/OE.20.013173.
.. [3] L. Mester, A. A. Govyadinov, S. Chen, M. Goikoetxea, and R.
   Hillenbrand, “Subsurface chemical nanoidentification by nano-FTIR
   spectroscopy,” Nat. Commun., vol. 11, no. 1, p. 3359, Dec. 2020, doi:
   10.1038/s41467-020-17034-6.