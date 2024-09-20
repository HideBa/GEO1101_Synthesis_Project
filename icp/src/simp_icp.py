import os
import numpy as np
import rerun as rr
from simpleicp import PointCloud, SimpleICP


def main(true_affine=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    bunny_target = np.loadtxt(os.path.join(script_dir, "data/bunny_target.txt"))
    bunny_source = np.loadtxt(os.path.join(script_dir, "data/bunny_source_head.txt"))
    scaled_bunny_source = bunny_source * 1

    pc_fix = PointCloud(bunny_target, columns=["x", "y", "z"])
    pc_mov = PointCloud(scaled_bunny_source, columns=["x", "y", "z"])

    icp = SimpleICP()

    icp.add_point_clouds(pc_fix, pc_mov)

    rr.init("rerun_ICP_pointCloud")
    rr.spawn()
    rr.log("source points", rr.Points3D(bunny_source, radii=0.001))
    rr.log("target points", rr.Points3D(bunny_target, radii=0.001))
    # rr.log("scaled source points", rr.Points3D(scaled_bunny_source, radii=0.001))

    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = (
        icp.run(max_overlap_distance=10)
    )
    rr.log("transformed", rr.Points3D(X_mov_transformed, radii=0.001))


if __name__ == "__main__":
    main(true_affine=True)
