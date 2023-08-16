.. _basics:

Basics of SNOM modelling
========================

This page covers the basic information that you'll need to understand the models in ``pysnom``.
It starts with a description of the tip-sample interaction, and how we can use effective polarizability to model SNOM scattering.
Then it introduces signal demodulation, the process which allows the tiny near-field signals to be distinguished from the huge background signal in real SNOM experiments.
Then finally it describes signal normalization, which we need to make quantitative comparisons between modelled and experimental data.

Scattering and effective polarizability
---------------------------------------

The image below shows a typical scattering SNOM experiment.

.. image:: basics/tip_sample.svg
   :align: center
   :alt: An image showing an atomic force microscope (AFM) tip close to a flat sample, with a beam of light shining onto the area where they meet. At the apex of the AFM tip and the area of the sample just below it, there are glowing lights representing a charge polarization induced by the beam of light. Inside the beam of light there are two arrows labeled "E_in", one which shines onto the tip directly and one which bounces off the sample then onto the tip. There are also two arrows with dotted lines, labelled "E_scat", one which shines back along the light beam from the tip directly and one which bounces off the sample first. The point where the arrows bounce off the sample is labelled "far-field reflections".

A far-field light source with electric field :math:`E_{in}` shines onto an AFM tip close to a sample.
The electric field interacts with the charges in the tip and sample to create a near-field polarization.
Because they are close, the tip and sample couple together to produce a combined dipole in response to the electric field, which scatters light with electric field :math:`E_{scat}` back into the far field.

This process can be described with the fundamental equation for modelling SNOM scattering,

.. math::
   :label: scatter

   \sigma = \frac{E_{scat}}{E_{in}} = (1 + c_r r)^2 \alpha_{eff}.


This introduces the scattering coefficient, :math:`\sigma`, which relates the strength of the near-field scattered light to the strength of the incident light.

The right-hand side of the equation is made from two parts:

*  The **effective polarizability** of the tip and sample, :math:`\alpha_{eff}`.
   This determines the strength of the near-field polarization that is excited by :math:`E_{in}`, so it contains all the information about the near-field sample interaction.

   The value of :math:`\alpha_{eff}` is challenging to model, and depends on many factors, including the dielectric function of the sample, the shape of the tip, and the distance between the tip and sample.

   The main task of ``pysnom`` is to calculate :math:`\alpha_{eff}` using different methods (see :ref:`FDM <fdm>` and :ref:`PDM <pdm>` for details).

*  A **far-field factor**, :math:`(1 + c_r r)^2`.
   Here :math:`r` is the Fresnel reflection coefficient for far-field light, and :math:`c_r` is an additional empirical factor that describes the detected strength of the reflected light compared to the incident light (this factor can vary based on the microscope setup but will be constant throughout an experiment).

   As shown by the diagram above, the :math:`(1 + c_r r)` term appears because the tip is illuminated both directly and through reflection from the sample surface.
   The scattered light is also detected directly and by reflection, which means the total contribution from the far field appears twice, as :math:`(1 + c_r r) (1 + c_r r) = (1 + c_r r)^2`.

   See :ref:`far-field` for details of how to calculate this term using ``pysnom``.

   .. hint::
      .. _far_field_warning:

      It's common to assume that the :math:`(1 + c_r r)^2` term will be constant throughout a SNOM experiment, because the area of the far-field laser spot is so much bigger than the near-field-confined area probed by SNOM.
      So it's often neglected in analysis.

      However, there are many occasions where the far-field reflection coefficient *does* have a significant affect on results, particularly
      near large features or on cluttered substrates [1]_.
      *Don't neglect it without thinking!*

SNOM experiments are typically sensitive to not just the amplitude but also
the phase of the scattered light, relative to the incident light.
Because of this, :math:`\sigma` takes the form of a complex number
with amplitude, :math:`s`, and phase, :math:`\phi`, given by

.. math::
   :label: amp_and_phase

   \begin{align*}
      s &= |\sigma|, \ \text{and}\\
      \phi &= \arg(\sigma).
   \end{align*}

Demodulation
------------

In an ideal SNOM experiment, we would detect only the near-field scattered light (:math:`E_{scat}` from equation :eq:`scatter`).

However in real experiments there is a problem.
The detector that collects the light picks up the whole of the reflected light beam, including both near-field and far-field reflections.
That means the part of the detected light that is scattered from the tip is only a tiny, tiny portion of the total detected light, and the background light completely swamps the useful signal.

To get around that problem, we typically oscillate the AFM tip height, :math:`z_{tip}`,  at a frequency :math:`\omega_{tip}`, then use a `lock-in amplifier <https://en.wikipedia.org/wiki/Lock-in_amplifier>`_ to demodulate the total detected signal at higher harmonics of that frequency, :math:`n \omega_{tip}` (where :math:`n = 2, 3, 4, \ldots`).

This oscillation modulates the near-field interaction, but mostly leaves the far field unchanged, so the lock-in can extract the near-field part of the signal by looking for only parts of the signal that change with the right frequency.

The lock-in-demodulated signals that we actually detect are determined, not by equation :eq:`scatter`, but by

.. math::
   :label: demod_scatter

   \sigma_n = \frac{E_{scat, n}}{E_{in}} = (1 + c_r r)^2 \alpha_{eff, n},

with amplitude and phase

.. math::
   :label: demod_amp_and_phase

   \begin{align*}
      s_n &= |\sigma_n|, \ \text{and}\\
      \phi_n &= \arg(\sigma_n).
   \end{align*}

In these equations a subscript :math:`n` indicates that a signal is demodulated at the :math:`n^\text{th}` harmonic.

For modelling SNOM signals, the practical difference here is that we must calculate the demodulated effective polarizability, :math:`\alpha_{eff, n}`, instead of just :math:`\alpha_{eff}`.
``pysnom`` has the ability to calculate both of these quantities, as well as a function which can be used to simulate lock-in measurements of arbitrary functions (see the page :ref:`Demodulation <demodulation>` for more details).

.. _normalization:

Normalization
-------------

As discussed above, the signal that is detected in a standard SNOM experiment is :math:`E_{scat, n}`.
However the actual detected strength of the signal depends on a number of factors that may be unknown.

To make this clearer, we can rearrage equation :eq:`demod_scatter` as

.. math::
   :label: E_scat_n

   E_{scat, n} = E_{in} (1 + c_r r)^2 \alpha_{eff, n}.

This shows that the detected signal depends on :math:`E_{in}` (which can vary with the type of source, the alignment, and the light energy), and a far-field factor.
Additionally, the detected signal will also depend on the sensitivity and alignment of the detector.

For quantitative SNOM measurements, we therefore usually normalize our signal to a SNOM measurement from a known reference material (typically gold or silicon).
This gives us the near-field contrast, :math:`\eta_n`, which is described by

.. math::
   :label: eta_n

   \eta_n
   = \frac{\sigma_n}{\sigma_n^{\text{(ref)}}}
   = \frac{E_{scat, n}}{E_{scat, n}^{\text{(ref)}}}
   = \frac{(1 + c_r r)^2 \alpha_{eff, n}}
   {(1 + c_r r^{\text{(ref)}})^2 \alpha_{eff, n}^{\text{(ref)}}},

where a superscript :math:`\text{(ref)}` indicates a quantity taken from the reference material.

Here the unknown :math:`E_{in}` terms cancel, and (provided the experimental conditions remain the same) any detector-related effects should also cancel.
As any unknown conditions have been removed, :math:`\eta_n` can be used for quantitative comparisons between experimental and modelled data.

Additionally, if :math:`r \approx r^{\text{(ref)}}` the far-field terms should cancel too, meaning

.. math::
   :label: eta_n_no_far_field

   \eta_n
   \approx \frac{\alpha_{eff, n}}{\alpha_{eff, n}^{\text{(ref)}}}

(however see :ref:`the hint above <far_field_warning>` for advice on when this is safe to do).

References
----------
.. [1] L. Mester, A. A. Govyadinov, and R. Hillenbrand, “High-fidelity
   nano-FTIR spectroscopy by on-pixel normalization of signal harmonics,”
   Nanophotonics, vol. 11, no. 2, p. 377, 2022, doi:
   10.1515/nanoph-2021-0565.