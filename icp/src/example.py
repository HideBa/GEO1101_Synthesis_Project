import os
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
import numpy as np
import rerun as rr


def main(true_affine=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    bunny_target = np.loadtxt(os.path.join(script_dir, "data/bunny_target.txt"))
    bunny_source = np.loadtxt(os.path.join(script_dir, "data/bunny_source_head.txt"))
    scaled_bunny_source = bunny_source * 3

    rr.init("rerun_ICP_pointCloud")
    rr.spawn()
    rr.log("source points", rr.Points3D(bunny_source, radii=0.001))
    rr.log("target points", rr.Points3D(bunny_target, radii=0.001))
    rr.log("scaled source points", rr.Points3D(scaled_bunny_source, radii=0.001))

    # affine = AffineRegistration(**{"X": bunny_target, "Y": scaled_bunny_source})
    # TY, reg = affine.register()
    reg = RigidRegistration(X=bunny_target, Y=scaled_bunny_source)
    TY, reg = reg.register()
    rr.log("transformed", rr.Points3D(TY, radii=0.001))


if __name__ == "__main__":
    main(true_affine=True)
