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

The Fresnel reflection coefficient relates the strength of an incident light beam to the beam that gets reflected from it's surface.

