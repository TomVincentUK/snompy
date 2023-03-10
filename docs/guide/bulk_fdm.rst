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

.. image:: diagrams/dipole_field.svg
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

   z_{Q0} \approx \frac{1.31 L r}{L + 2 r}

from the ends of the ellipsoid, where :math:`r` is the radius of curvature
at the pointy end, and :math:`L` is the semi-major axis length (the
distance from the ellipsoid centre to the pointy end).

The strength of the electric dipole moment can be related to the charges
and their separation as

.. math::
   :label: p_0

   p_0 = 2 (L - z_{Q0}) Q_0 \quad (\approx 2 L Q_0, \ \mathrm{for} \ r \ll L).

Tip-sample interaction
----------------------

Now let's consider what happens when we move our model AFM tip close to a
sample's surface, with a tip-sample separation of :math:`z`.

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

.. image:: diagrams/fdm.svg
   :align: center

We can model the electric field response of the sample to the charge
:math:`Q_0`, at a height of :math:`z + z_{Q0}`, using
the
`method of image charges <https://en.wikipedia.org/wiki/Method_of_image_charges>`_.

This means we can add a fictitious image charge, :math:`Q'_0 = -\beta Q_0`,
at a depth of :math:`z + z_{Q0}` below the surface.
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
distance :math:`z_{Q1} \approx r / 2` away from the end of the tip.

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
charge, :math:`Q'_1 = \beta Q_1`, at a depth of :math:`z + z_{Q1}` below
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

   f_i = \left(g - \frac{r + 2 z + z_{Qi}}{2 L} \right)
   \frac{\ln\left(\frac{4 L}{r + 4 z + 2 z_{Qi}}\right)}
   {\ln\left(\frac{4 L}{r}\right)},

where :math:`g \approx 0.7` is an empirical factor that describes how much
of the induced charge is relevant for the near-field interaction (see
`Parameters`_ for more details on how this factor affects the results).
In ``pysnom``, equation :eq:`f_i_bulk` is provided by the function
:func:`pysnom.fdm.geom_func_bulk`.

The charges :math:`Q_1` and :math:`-Q_1` form another dipole

.. math::
   :label: p_1

   p_1 = (L - z_{Q1}) Q_1 \quad (\approx L Q_1, \ \mathrm{for} \ r \ll L).

The effective polarisability of the tip and sample can then be found from
the total induced dipole, as

.. math::
   :label: eff_pol_bulk_fdm

   \alpha_{eff}
   = \frac{p_0 + p_1}{E_{in}}
   \approx \frac{2 L Q_0}{E_{in}}
   \left(1 + \frac{f_0 \beta}{2 (1 - f_1 \beta)}\right)
   \propto 1 + \frac{f_0 \beta}{2 (1 - f_1 \beta)}.

In ``pysnom``, equation :eq:`eff_pol_bulk_fdm` is provided by the function
:func:`pysnom.fdm.eff_pol_0_bulk`.

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
function :func:`pysnom.fdm.eff_pol_bulk`.

Using pysnom for bulk FDM
-------------------------

[This section should explain :func:`pysnom.fdm.eff_pol_bulk` and relate it
to the section above, including different ways to specify the material
parameters.]

Parameters
----------

[A section talking about how each parameter of the FDM affects the results.
Possibly with graphs for some parameters (`z`, `radius`, `semi_maj_ax`)]

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