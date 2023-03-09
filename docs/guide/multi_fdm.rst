.. _multi_fdm:

Multilayer finite dipole model
==============================

* This section follows on from the foundation of bulk FDM, read that first

Extending the finite dipole method to multiple layers
-----------------------------------------------------

* Description of problem: multiple interfaces complicates solution
* We can replace it with a single interface that matches the boundary conditions of E and phi
* To calculate the E and phi we perform integral over all k
* Multilayer reflection coefficient description given below
* Then we solve for the positions and charges of the images of q_0, q_1
* We substitute those into a modified version of the equation for alpha
* We can then demodulate, exactly as described for the bulk (and in more detail on next page)

The multilayer reflection coefficient
-------------------------------------

* 3 layers
* Extend by recursion
