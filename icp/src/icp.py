import os
import numpy as np
import laspy
from functools import partial
from pycpd import RigidRegistration, AffineRegistration
import rerun as rr
import matplotlib.pyplot as plt
from simpleicp import PointCloud, SimpleICP


def read_pc(filepath):
    las = laspy.read(filepath)
    print(f"Loaded {filepath} with {len(las.points)} points.")
    points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
    return points


def translate_points(points, translate):
    return points + translate


def visualize(iteration, error, X, Y, ax, fig, save_fig=False):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="red", label="Target")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="blue", label="Source")
    ax.text2D(
        0.87,
        0.92,
        "Iteration: {:d}\nQ: {:06.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )
    ax.legend(loc="upper left", fontsize="x-large")
    ax.view_init(90, -90)
    if save_fig is True:
        ax.set_axis_off()

    plt.draw()
    if save_fig is True:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, "data/image.png")

        fig.savefig(out_path, dpi=600)  # Used for making gif.
    plt.pause(0.001)


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, "data/fme_thinned_clipped2.las")
    target_path = os.path.join(script_dir, "data/fme_thinned.las")
    # source_path = os.path.join(script_dir, "data/geolab_iphone_thinned.las")
    # target_path = os.path.join(script_dir, "data/geolab_iphone_thinned_clipped.las")
    source = read_pc(source_path)
    translation = [10, 10, 10]
    # source = translate_points(source, translation)
    target = read_pc(target_path)

    pc_fix = PointCloud(target, columns=["x", "y", "z"])
    pc_mov = PointCloud(source, columns=["x", "y", "z"])

    icp = SimpleICP()

    icp.add_point_clouds(pc_fix, pc_mov)

    rr.init("rerun_ICP_pointCloud")
    rr.spawn()
    rr.log("source points", rr.Points3D(source, radii=0.01))
    rr.log("target points", rr.Points3D(target, radii=0.01))

    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = (
        icp.run(max_overlap_distance=100)
    )
    rr.log("transformed", rr.Points3D(X_mov_transformed, radii=0.01))

    # plt.show()


if __name__ == "__main__":
    main()
