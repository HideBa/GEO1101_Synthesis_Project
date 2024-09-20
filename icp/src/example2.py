import os
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
import numpy as np
import rerun as rr


def main(true_affine=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    bunny_target = np.loadtxt(os.path.join(script_dir, "data/bunny_target.txt"))
    X1 = np.zeros((bunny_target.shape[0], bunny_target.shape[1] + 1))
    X1[:, :-1] = bunny_target
    X2 = np.ones((bunny_target.shape[0], bunny_target.shape[1] + 1))
    X2[:, :-1] = bunny_target
    X = np.vstack((X1, X2))

    bunny_source = np.loadtxt(os.path.join(script_dir, "data/bunny_source.txt"))
    Y1 = np.zeros((bunny_source.shape[0], bunny_source.shape[1] + 1))
    Y1[:, :-1] = bunny_source
    Y2 = np.ones((bunny_source.shape[0], bunny_source.shape[1] + 1))
    Y2[:, :-1] = bunny_source
    Y = np.vstack((Y1, Y2))

    Ysubset = Y[1::2, :]
    rr.init("rerun_ICP_pointCloud")
    rr.spawn()
    rr.log("X", rr.Points3D(X, radii=0.05))
    rr.log("Y", rr.Points3D(Y, radii=0.05))
    rr.log("Y sub", rr.Points3D(Ysubset, radii=0.05))
    reg = DeformableRegistration(**{"X": X, "Y": Ysubset})
    reg.register()

    # YT = reg.transform_point_cloud(Y=Y)

    # rr.log("transformed", rr.Points3D(YT, radii=0.05))


if __name__ == "__main__":
    main(true_affine=True)
