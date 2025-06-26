import pyrao  # type: ignore
import numpy as np
import scipy.optimize as opt  # type: ignore
import time

lgs_wfss = [
    (17.5 * np.cos(theta), 17.5 * np.sin(theta))
    for theta in np.arange(8) * 2 * np.pi / 8
]


def build_imat(dms):
    imats = []
    for dm in dms:
        system_geom = pyrao.SystemGeom.new(
            teldiam=2.0,
            cobs=0.16,
            r0=0.12,
            coupling=0.3,
            nactux=11,
            dmalt=dm["alt"],
            pitch=dm["pitch"],
            nsubx=10,
            ntssamples=0,
            nphisamples=21,
            wfs_dirs=lgs_wfss,
            ts_dirs=[],
            dm_delta=dm["pert"],
            gsalt=90e3,
        )
        system_geom.filter_meas([x == 1 for x in system_geom.pmeas()])
        imats.append(np.array(system_geom.imat()))

    imat = np.concatenate(imats, axis=1)
    return imat


dms_true = [
    {"alt": 0.0, "pitch": 0.22, "pert": (0.00174467, 0.07772004)},
    {"alt": 6000.0, "pitch": 0.25, "pert": (0.02740716, -0.2432777)},
    {"alt": 13500.0, "pitch": 0.30, "pert": (0.09846721, -0.07259582)},
]
imat_true = build_imat(dms_true)


def test_time():
    t1 = time.time()
    build_imat(dms_true)
    t2 = time.time()
    return t2-t1


print(test_time())


def cost(pert):
    dms = [
        {"alt": 0.0, "pitch": 0.22, "pert": (pert[0], pert[1])},
        {"alt": 6000.0, "pitch": 0.25, "pert": (pert[2], pert[3])},
        {"alt": 13500.0, "pitch": 0.30, "pert": (pert[4], pert[5])},
    ]
    err = np.linalg.norm(imat_true - build_imat(dms), ord="fro")
    return err


pert_estimate = np.zeros(6)
result = opt.minimize(cost, pert_estimate)
print(result)
