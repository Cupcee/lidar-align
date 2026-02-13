import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

# OLD_CALIB_STRING = """
# 1,0,0,0,0,1,0,0,0,0,1,0;
# 0.997292,0.0680656,-0.027833,0.0239639,-0.0669792,0.997021,0.0382632,-1.10121,0.0303544,-0.0362953,0.99888,0.0138038;
# -0.886873,-0.461957,0.007202,-0.208495,0.461488,-0.885015,0.0614494,0.606671,-0.0220131,0.0578215,0.998084,-0.0851192;
# 0.922887,-0.384039,-0.0281881,-2.26607,0.385063,0.920876,0.0609422,0.634961,0.00255362,-0.067097,0.997743,0.110059;
# -0.855559,0.517699,0.00258428,-0.200085,-0.517377,-0.855184,0.0313555,-1.71323,0.0184428,0.0254893,0.999505,-0.0186664;
# 0.913619,0.406555,0.00338869,-2.2398,-0.406505,0.913295,0.0254494,-1.69465,0.00725159,-0.0246285,0.99967,0.201087;
# -0.996481,0.0105359,-0.0831498,-3.12797,-0.0111834,-0.99991,0.00732328,-0.415618,-0.0830654,0.00822729,0.996509,0.449154
# """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch apply lidar calibration corrections."
    )
    parser.add_argument("input_dir", help="Directory containing input .pcd files")
    parser.add_argument("calib_string", help="Calibration string")
    parser.add_argument("output_dir", help="Directory to save corrected .pcd files")
    parser.add_argument(
        "--calibration_json",
        required=True,
        help="Path to the JSON file from the visualizer",
    )
    return parser.parse_args()


def parse_old_calibration(raw_string):
    """Parses the original hardcoded calibration string."""
    transforms = {}
    clean_str = raw_string.replace("\n", "").strip()
    parts = clean_str.split(";")

    for idx, part in enumerate(parts):
        if not part.strip():
            continue
        vals = [float(x) for x in part.split(",")]
        mat = np.eye(4)
        mat[0, :] = vals[0:4]
        mat[1, :] = vals[4:8]
        mat[2, :] = vals[8:12]
        transforms[idx] = mat
    return transforms


def main():
    args = parse_args()

    # 1. Setup Directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 2. Load Calibration Data
    # Old (Original) Transforms
    old_transforms = parse_old_calibration(args.calib_string)

    # New (Corrected) Transforms from JSON
    with open(args.calibration_json, "r") as f:
        new_transforms_raw = json.load(f)

    # Convert JSON lists to numpy arrays
    new_transforms = {}
    for k, v in new_transforms_raw.items():
        new_transforms[int(k)] = np.array(v)

    # 3. Pre-calculate Correction Matrices
    # T_correction = T_new * inv(T_old)
    correction_matrices = {}
    print("\n--- Calibration Plan ---")
    for sid, t_old in old_transforms.items():
        if sid in new_transforms:
            t_new = new_transforms[sid]
            # Calculate relative transform
            t_corr = np.dot(t_new, np.linalg.inv(t_old))
            correction_matrices[sid] = t_corr

            # Check if it's identity (no change)
            if not np.allclose(t_corr, np.eye(4)):
                print(f"Sensor {sid}: Applying correction.")
            else:
                print(f"Sensor {sid}: No significant change detected.")
        else:
            print(f"Sensor {sid}: No new calibration found. Keeping original.")
            correction_matrices[sid] = np.eye(4)

    # 4. Process Files
    pcd_files = glob.glob(os.path.join(args.input_dir, "*.pcd"))
    print(f"\nProcessing {len(pcd_files)} files...")

    for pcd_path in tqdm(pcd_files):
        # A. Read File
        t_pcd = o3d.t.io.read_point_cloud(pcd_path)

        # Check required fields
        if "sensor_id" not in t_pcd.point:
            print(f"Skipping {pcd_path}: Missing 'sensor_id'")
            continue

        # Get data as numpy (Pass by reference where possible, but we need to write back)
        np_xyz = t_pcd.point.positions.numpy()
        np_ids = t_pcd.point["sensor_id"].numpy().flatten()

        # Create a copy for the corrected positions
        corrected_xyz = np_xyz.copy()

        # B. Apply Corrections per Sensor
        unique_sensors = np.unique(np_ids)

        for sid in unique_sensors:
            if sid not in correction_matrices:
                continue

            mat = correction_matrices[sid]
            if np.allclose(mat, np.eye(4)):
                continue

            # Get points for this sensor
            mask = np_ids == sid
            points_subset = corrected_xyz[mask]

            # Apply Transform Manually: P_new = (R * P_old.T + T).T
            # Open3D/Numpy logic: P_new = P_old @ R.T + T
            R = mat[:3, :3]
            T = mat[:3, 3]

            # This is the fastest numpy way to transform
            points_transformed = np.dot(points_subset, R.T) + T

            # Write back to the main array
            corrected_xyz[mask] = points_transformed

        # C. Update Tensor and Save
        t_pcd.point.positions = o3d.core.Tensor(corrected_xyz)

        filename = os.path.basename(pcd_path)
        out_path = os.path.join(args.output_dir, filename)

        # Write (compressed=True is usually smaller and faster)
        o3d.t.io.write_point_cloud(out_path, t_pcd, write_ascii=False)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
