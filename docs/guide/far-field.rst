.. _far-field:

Far-field reflections
=====================

In :ref:`basics`, we showed the image below to explain why far-field reflections are important for modelling SNOM measurements.

.. image:: basics/tip_sample.svg
   :align: center
   :alt: An image showing an atomic force microscope (AFM) tip close to a flat sample, with a beam of light shining onto the area where they meet. At the apex of the AFM tip and the area of the sample just below it, there are glowing lights representing a charge polarization induced by the beam of light. Inside the beam of light there are two arrows labeled "E_in", one which shines onto the tip directly and one which bounces off the sample then onto the tip. There are also two arrows with dotted lines, labelled "E_scat", one which shines back along the light beam from the tip directly and one which bounces off the sample first. The point where the arrows bounce off the sample is labelled "far-field reflections".

We showed that the detected SNOM signal, :math:`\sigma_n`, depends on the effective polarizability, :math:`\alpha_{eff, n}`, and a far-field reflection term, :math:`(1 + c_r r)^2`, as

.. math::
   :label: demod_scatter_recap

   \sigma_{scat, n} = (1 + c_r r)^2 \alpha_{eff, n},

where :math:`r` is the the far-field, Fresnel reflection coefficient and :math:`c_r` is an empirical factor that describes the detected strength of the reflected light compared to the incident light.

DESCRIBE THIS PAGE.

Fresnel reflection coefficient
------------------------------

Bulk samples
^^^^^^^^^^^^

The Fresnel reflection coefficient relates the strength of a reflected light beam to the strength of the incident beam.
For a bulk sample it can be calculated simply.

The image below shows a diagram of a simple reflection from a bulk sample.
An incident light beam with electric field :math:`E_{in}` hits the sample surface.
Then, part of the beam is reflected with field :math:`E_{r}`, and part is transmitted with field :math:`E_{t}`.

.. image:: far-field/fresnel.svg
   :align: center

The angle of the reflected beam to the surface normal, :math:`\theta_r`, is the same as the angle of incidence, :math:`\theta_{in}`, and the angle of the transmitted beam can be found from `Snell's law <https://en.wikipedia.org/wiki/Snell%27s_law>`_ as

.. math::

   n_0 \sin(\theta_{in}) = n_1 \sin(\theta_t),

where :math:`n_0` and :math:`n_1` are the refractive indices of the environment and the sample.

.. hint::
   In general, the refractive index, :math:`n`, can be found from the permittivity as :math:`n = \sqrt{\varepsilon}` for non-magnetic materials.

The value of the reflection coefficient depends on the angle and polarization, :math:`P`, of the incident light as

.. math::

   r = \begin{cases}
      r_p = \frac{n_0 \cos(\theta_{in} - n_1 \cos(\theta_{r})}{n_0 \cos(\theta_{in} + n_1 \cos(\theta_{r})}, & \text{for} \ P = p\\
      r_s = \frac{n_1 \cos(\theta_{in} - n_0 \cos(\theta_{r})}{n_1 \cos(\theta_{in} + n_0 \cos(\theta_{r})}, & \text{for} \ P = s\\
   \end{cases}

.. hint::
   The term p polarization refers to electric fields that are parallel to the plane of incidence (as shown in the drawing above).
   The term s polarisation comes from the German word senkrecht, and refers to electric fields that are perpendicular to the plane of incidence.
   In SNOM measurements we almost always use p polarisation.

Multilayer samples
^^^^^^^^^^^^^^^^^^

For multilayer samples, calculating the reflection coefficient becomes more complicated, as we must account for reflections from multiple surfaces.
In ``pysnom`` we use the `transfer matrix method <https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)>`_ to calculate reflection and transmission coefficients from multilayer samples [1]_.

Accounting for far-field in SNOM simulations
--------------------------------------------

The most common use for the far-field reflection coefficient in ``pysnom`` is to calculate the far-field factor :math:`(1 + c_r r)^2` (as in equation :eq:`demod_scatter_recap`).
In this section we'll show a worked example of how that can be done, by simulating SNOM spectra from `poly(methyl methacrylate) <https://en.wikipedia.org/wiki/Poly(methyl_methacrylate)>`_ (PMMA) layers of different thickness on Si.

Let's start by creating a model of permittivity for our PMMA

References
----------
.. [1] T. Zhan, X. Shi, Y. Dai, X. Liu, and J. Zi, “Transfer matrix  method for optics in graphene layers,” J Phys. Condens. Matter, vol. 25, no. 21, p. 215301, May 2013, doi: 10.1088/0953-8984/25/21/215301.