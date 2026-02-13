import argparse
import json
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# DEFAULT_CALIB_STRING = """
# 1,0,0,0,0,1,0,0,0,0,1,0;
# 0.997292,0.0680656,-0.027833,0.0239639,-0.0669792,0.997021,0.0382632,-1.10121,0.0303544,-0.0362953,0.99888,0.0138038;
# -0.886873,-0.461957,0.007202,-0.208495,0.461488,-0.885015,0.0614494,0.606671,-0.0220131,0.0578215,0.998084,-0.0851192;
# 0.922887,-0.384039,-0.0281881,-2.26607,0.385063,0.920876,0.0609422,0.634961,0.00255362,-0.067097,0.997743,0.110059;
# -0.855559,0.517699,0.00258428,-0.200085,-0.517377,-0.855184,0.0313555,-1.71323,0.0184428,0.0254893,0.999505,-0.0186664;
# 0.913619,0.406555,0.00338869,-2.2398,-0.406505,0.913295,0.0254494,-1.69465,0.00725159,-0.0246285,0.99967,0.201087;
# -0.996481,0.0105359,-0.0831498,-3.12797,-0.0111834,-0.99991,0.00732328,-0.415618,-0.0830654,0.00822729,0.996509,0.449154
# """

COLOR_MAP = {
    0: [120, 40, 120],
    1: [200, 160, 29],
    2: [185, 131, 44],
    3: [231, 205, 232],
    4: [151, 243, 194],
    5: [244, 199, 13],
    6: [255, 91, 219],
    7: [0, 150, 255],
    8: [208, 138, 11],
    9: [182, 37, 231],
    10: [88, 237, 6],
    11: [55, 151, 131],
    12: [243, 136, 95],
    13: [130, 28, 79],
    14: [206, 191, 98],
    15: [39, 150, 54],
    16: [225, 111, 6],
    17: [237, 126, 132],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pcd", help="Path to input PCD file")
    parser.add_argument("calib_string", help="Calibration string")
    return parser.parse_args()


class LidarApp:
    def __init__(self, pcd_path, calib_string):
        self.pcd_path = pcd_path
        self.calib_string = calib_string

        # Data storage
        self.sensor_ids = []
        self.pcd_dict = {}  # Original PointClouds
        self.initial_transforms = {}
        self.adjustment_transforms = {}
        self.active_sensor_idx = 0

        # Steps
        self.trans_step = 0.05
        self.rot_step = 0.1 * (np.pi / 180)

        # Initialize GUI
        self.window = gui.Application.instance.create_window(
            "Lidar Calibrator", 1600, 900
        )
        w = self.window

        # 1. 3D Scene Widget (Left)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(w.renderer)
        self.widget3d.set_on_key(self.on_key)

        # 2. Control Panel (Right)
        em = w.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Info Labels
        self.lbl_active = gui.Label("Active Sensor: -")
        self.lbl_active.text_color = gui.Color(1.0, 0.5, 0.0)
        self.panel.add_child(self.lbl_active)
        self.panel.add_child(gui.Label(""))  # Spacer

        # Instructions
        self.add_instruction("TAB", "Switch Sensor")
        self.add_instruction("W / S", "Move Y (Forward/Back)")
        self.add_instruction("A / D", "Move X (Left/Right)")
        self.add_instruction("Q / E", "Move Z (Up/Down)")
        self.panel.add_child(gui.Label(""))
        self.add_instruction("J / L", "Yaw (Rotate Z)")
        self.add_instruction("I / K", "Pitch (Rotate Y)")
        self.add_instruction("U / O", "Roll (Rotate X)")
        self.panel.add_child(gui.Label(""))
        self.add_instruction("ENTER", "Save Calibration")

        # Layout
        w.set_on_layout(self._on_layout)
        w.add_child(self.widget3d)
        w.add_child(self.panel)

        # Load Data
        self.parse_calibration()
        self.load_pcd()
        self.update_scene()
        self.update_labels()

    def add_instruction(self, key_text, desc_text):
        h = gui.Horiz(0)
        l_key = gui.Label(f"[{key_text}]")
        l_key.text_color = gui.Color(0.6, 0.8, 1.0)  # Light Blue
        l_desc = gui.Label(f" : {desc_text}")
        h.add_child(l_key)
        h.add_child(l_desc)
        self.panel.add_child(h)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300
        self.widget3d.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(
            r.x + r.width - panel_width, r.y, panel_width, r.height
        )

    def parse_calibration(self):
        clean_str = self.calib_string.replace("\n", "").strip()
        parts = clean_str.split(";")
        for idx, part in enumerate(parts):
            if not part.strip():
                continue
            vals = [float(x) for x in part.split(",")]
            mat = np.eye(4)
            mat[0, :] = vals[0:4]
            mat[1, :] = vals[4:8]
            mat[2, :] = vals[8:12]
            self.initial_transforms[idx] = mat

    def load_pcd(self):
        t_pcd = o3d.t.io.read_point_cloud(self.pcd_path)
        xyz = t_pcd.point.positions.numpy()
        ids = t_pcd.point["sensor_id"].numpy().flatten()
        lbls = t_pcd.point["label"].numpy().flatten()

        self.sensor_ids = sorted(np.unique(ids))

        for sid in self.sensor_ids:
            mask = ids == sid
            # Create standard PCD
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[mask])

            # Colors
            colors = np.zeros((np.sum(mask), 3))
            masked_lbls = lbls[mask]
            for i, l in enumerate(masked_lbls):
                rgb = COLOR_MAP.get(l, [128, 128, 128])
                colors[i] = [c / 255.0 for c in rgb]
            pcd.colors = o3d.utility.Vector3dVector(colors)

            self.pcd_dict[sid] = pcd
            self.adjustment_transforms[sid] = np.eye(4)

            # Add to Scene immediately (we will update transform later)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.point_size = 3.0
            self.widget3d.scene.add_geometry(str(sid), pcd, mat)

    def update_scene(self):
        active_id = self.sensor_ids[self.active_sensor_idx]

        for sid in self.sensor_ids:
            # 1. Calculate Transform
            # Display = T_adj * T_identity (Geometry is already fused)
            # We actually just modify the scene node transform
            t_adj = self.adjustment_transforms[sid]

            self.widget3d.scene.set_geometry_transform(str(sid), t_adj)

            # 2. Update material (Dim inactive)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.point_size = 3.0
            if sid != active_id:
                # Dim via base color multiplier (works for Lit shader)
                mat.base_color = [0.2, 0.2, 0.2, 1.0]
            else:
                mat.base_color = [1.0, 1.0, 1.0, 1.0]

            self.widget3d.scene.modify_geometry_material(str(sid), mat)

        self.widget3d.force_redraw()

    def update_labels(self):
        sid = self.sensor_ids[self.active_sensor_idx]
        self.lbl_active.text = f"Active Sensor ID: {sid}"

    def on_key(self, event):
        if event.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.HANDLED

        k = event.key
        print(f"[DEBUG] Pressed key: {k}")

        # Switch Sensor (TAB)
        if k == 9:
            self.active_sensor_idx = (self.active_sensor_idx + 1) % len(self.sensor_ids)
            self.update_scene()
            self.update_labels()
            return gui.Widget.EventCallbackResult.HANDLED

        # Save (ENTER)
        if k == 10:
            self.save()
            return gui.Widget.EventCallbackResult.HANDLED

        # Movement mappings
        sid = self.sensor_ids[self.active_sensor_idx]

        # Translation
        if k == 119:
            self.apply_trans(sid, 1, 1)  # W
        elif k == 115:
            self.apply_trans(sid, 1, -1)  # S
        elif k == 100:
            self.apply_trans(sid, 0, 1)  # D
        elif k == 97:
            self.apply_trans(sid, 0, -1)  # A
        elif k == 113:
            self.apply_trans(sid, 2, 1)  # Q
        elif k == 101:
            self.apply_trans(sid, 2, -1)  # E

        # Rotation
        elif k == 108:
            self.apply_rot(sid, 2, -1)  # L
        elif k == 106:
            self.apply_rot(sid, 2, 1)  # J
        elif k == 105:
            self.apply_rot(sid, 1, 1)  # I
        elif k == 107:
            self.apply_rot(sid, 1, -1)  # K
        elif k == 117:
            self.apply_rot(sid, 0, 1)  # U
        elif k == 111:
            self.apply_rot(sid, 0, -1)  # O

        return gui.Widget.EventCallbackResult.HANDLED

    def apply_trans(self, sid, axis, sign):
        t = [0, 0, 0]
        t[axis] = sign * self.trans_step
        mat = np.eye(4)
        mat[:3, 3] = t
        self.adjustment_transforms[sid] = mat @ self.adjustment_transforms[sid]
        self.update_scene()

    def apply_rot(self, sid, axis, sign):
        angle = sign * self.rot_step
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(3)
        if axis == 0:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 2:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        mat = np.eye(4)
        mat[:3, :3] = R
        self.adjustment_transforms[sid] = mat @ self.adjustment_transforms[sid]
        self.update_scene()

    def save(self):
        out = {}
        for sid in self.sensor_ids:
            if sid in self.initial_transforms:
                # Final = Adjustment * Initial
                final = self.adjustment_transforms[sid] @ self.initial_transforms[sid]
                out[int(sid)] = final.tolist()

        with open("calibration_fixed.json", "w") as f:
            json.dump(out, f, indent=4)

        # Visual feedback
        dlg = gui.Dialog("Saved")
        dlg_layout = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        dlg_layout.add_child(gui.Label("Calibration saved to calibration_fixed.json"))
        ok = gui.Button("OK")
        ok.set_on_clicked(self.window.close_dialog)
        dlg_layout.add_child(ok)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input_pcd):
        print("File not found")
    else:
        gui.Application.instance.initialize()
        app = LidarApp(args.input_pcd, args.calib_string)
        gui.Application.instance.run()
