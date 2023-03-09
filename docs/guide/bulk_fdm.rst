.. _bulk_fdm:

Bulk finite dipole model
========================

The finite dipole model (FDM) is a method for estimating the effective
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

Tip-sample interaction
----------------------

Now let's consider what happens when we move our model AFM tip close to a
sample's surface.
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

* Charge in end of tip induces an image charge, which induces another charge in the tip
* That charge also has it's own image charge and counter-charge (ref Cvitkovic, all above)
* The system can be solved for the effective polarisability as: eqn (ref Hauer)
* geom function: eqn (ref [2]_])
* Properties of eff_pol_0:
  * Complex number -> amplitude and phase
  * Decays non-linearly from sample surface
  * Depends on dielectric functions of sample and environment

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