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