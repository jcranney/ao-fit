# ao-fit
Tools and tests for validating the misregistration identification scheme of MAVIS.

## Concept
MAVIS will be capable of measuring an interaction matrix between its {post-focal DMs, LGS Jitter Mirrors} and its {NGS WFSs slopes, LGS WFSs slopes, NGS WFS flux, LGS WFS flux}. MAVIS will also be able to generate a synthetic interaction matrix based on system parameters (DM misregistrations, etc.). The difference between these two matrices corresponds to the system modelling error, and therefore can be used to optimise the system parameters.

In principal, for a set of system parameters $\rho$, and a model of the interaction matrix $D(\rho)$, we can define an optimisation function:
$$
\hat\rho = \argmin_\rho{\|D_{\mathrm{meas}} - D(\rho) \|}
$$
for some norm $\|\cdot\|$.

## Validation
Uses `pyrao`, we can generate interaction matrices based on system parameters. To validate the concept described above, we can define system parameters similar to MAVIS, perturb them slightly, and then test the ability to find the perturbed values given only the nominal ones and an interaction matrix generated using the perturbed ones.

This experiment is performed in `validation.py`.


### Todo
 - noisy measured imat,
 - ✅ DM x/y misregistration,
 - WFS x/y misregistration,
 - DM rotation,
 - WFS rotation,
 - DM zoom,
 - WFS zoom,

### Results so far
Using `scipy.optim.least_square`, we get:
 - full convergence (to order of machine epsilon)
 - over large perturbations (~0.2 metres)
 - in very few steps (~6 steps)
 - in very short time (~0.3 seconds)

Using the full interaction matrix would be expensive, so we downsample the elements to use heuristically:
```python
# build the full imat (about 20 seconds)
imat = build_imat()

# find the sensitive elements
sensitivity_mask = np.abs(imat) > 1e-3

# get their indices
indices = list(zip(*np.nonzero(sensitivity_mask)))

# further downsample the sensitive elements until we have only ~1000 (e.g.)
indices = [
    indices[i]
    for i in np.random.randint(
        low=0, high=len(indices) - 1,
        size=[1000]
    )
]

# sample the imat at a subset of the sensitive indices (about 2 ms)
imat = build_imat(indices=indices)
```

A typical optimisation run (using `validation.py`) will exit with result status:
```
took 0.29 seconds
     message: `gtol` termination condition is satisfied.
     success: True
      status: 1
         fun: [ 0.000e+00 -6.083e-12 ... -2.708e-12  2.746e-12]
           x: [ 1.745e-03  7.772e-02  2.741e-02 -2.433e-01  9.847e-02
               -7.260e-02]
        cost: 2.3770505140501584e-21
         jac: [[-1.731e-01  4.350e-03 ...  0.000e+00 -0.000e+00]
               [ 0.000e+00  0.000e+00 ...  0.000e+00 -0.000e+00]
               ...
               [ 0.000e+00  0.000e+00 ...  0.000e+00 -0.000e+00]
               [ 0.000e+00  0.000e+00 ...  0.000e+00 -0.000e+00]]
        grad: [-1.043e-13 -7.990e-14 -3.333e-09 -5.375e-09 -8.079e-14
               -6.161e-14]
  optimality: 5.37502944100362e-09
 active_mask: [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  0.000e+00
                0.000e+00]
        nfev: 6
        njev: 6
[ 0.00174467  0.07772004  0.02740716 -0.2432777   0.09846721 -0.07259582]
[ 0.00174467  0.07772004  0.02740716 -0.2432777   0.09846721 -0.07259582]
```
The bottom two lines showing that the true perturbation is found incredibly precisely, in this case with only 6 steps of the optimiser required to reach this precision.