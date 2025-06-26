import pyrao  # type: ignore
import numpy as np
import scipy.optimize as opt  # type: ignore
import time

# define MAVIS WFSs
lgs_wfss = [
    (17.5 * np.cos(theta), 17.5 * np.sin(theta))
    for theta in np.arange(8) * 2 * np.pi / 8
]


def build_imat(pert=[0, 0, 0, 0, 0, 0], indices=None):
    """
    take the perturbation vector and generate either a full of sparse imat
    """
    # building the entire system is very quick, not much computation required
    system_geoms = []
    for dm in dms(pert):
        system_geoms.append(
            pyrao.SystemGeom.new(
                teldiam=8.0,
                cobs=0.16,
                r0=0.12,
                coupling=0.3,
                nactux=41,
                dmalt=dm["alt"],
                pitch=dm["pitch"],
                nsubx=40,
                ntssamples=0,
                nphisamples=41,
                wfs_dirs=lgs_wfss,
                ts_dirs=[],
                dm_delta=dm["pert"],
                gsalt=90e3,
            )
        )
    system_geom = pyrao.SystemGeom.merge_com(system_geoms)

    # sampling the imat is typically the expensive part:
    if indices is None:
        # the first time through, we build the nominal imat
        # takes ~20 seconds on my laptop
        imat = np.array(system_geom.imat())
    else:
        # every other time, we only sample it sparsely
        # takes ~2ms per 1000 samples
        imat = np.array(system_geom.imat_sparse(indices))
    return imat


def dms(pert):
    """
    return the DM configuration (eventually probably WFSs too) including the
    specified perturbations
    """
    return [
        {"alt": 0.0, "pitch": 0.22, "pert": (pert[0], pert[1])},
        {"alt": 6000.0, "pitch": 0.25, "pert": (pert[2], pert[3])},
        {"alt": 13500.0, "pitch": 0.30, "pert": (pert[4], pert[5])},
    ]


def init():
    """
    - construct an interaction matrix at the "home" position
    - find the elements that are sensitive and save their indices
    - resample the imat at the perturbed position at those indices
    - return the sampled imat, perturbations, and indices.
    """
    # first build the full matrix without any perturbations (how would we
    # know them in the first place?)
    imat = build_imat()
    # find the sensitive elements
    sensitivity_mask = np.abs(imat) > 1e-3
    # get their indices
    indices = list(zip(*np.nonzero(sensitivity_mask)))
    # reset rng for consistent results
    np.random.default_rng()
    # downsample the sensitive elements until we have only ~1000 (e.g.)
    indices = [
        indices[i]
        for i in np.random.randint(low=0, high=len(indices) - 1, size=[1000])
    ]

    # define the perturbations
    pert = np.array(
        [
            0.00174467,  # dm1x
            0.07772004,  # dm1y
            0.02740716,  # dm2x
            -0.2432777,  # dm2y
            0.09846721,  # dm3x
            -0.07259582,  # dm3y
        ]
    ).flatten()

    # sample the imat at the sensitive indices with the perturbation present
    imat = build_imat(pert, indices=indices)

    # todo: add noise to imat
    # imat = imat + noise

    return imat, np.array(pert).flatten(), indices


# initialise system
imat_true, pert_true, indices = init()


def test_time():
    """
    time how long it takes to sample the sparse imat
    """
    t1 = time.time()
    for i in range(100):
        build_imat(indices=indices)
    t2 = time.time()
    return (t2 - t1)/(i+1)


# time it out of curiosity
print(test_time())


def cost_vector(pert):
    """
    We will use a least squares solver, so can build a cost vector of the
    signed residuals.
    """
    err = imat_true - build_imat(pert, indices)
    return err


if __name__ == "__main__":
    # run optimisation
    t1 = time.time()
    pert_initial = np.zeros(6)
    result = opt.least_squares(cost_vector, pert_initial)
    t2 = time.time()

    # print results
    print(f"took {t2 - t1:0.2f} seconds")
    pert_estimate = result["x"]
    print(result)
    print(pert_estimate, pert_true)
