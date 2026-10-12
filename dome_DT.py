import pyvista as pv
import numpy as np
import time
import matplotlib.pyplot as plt


# ==========================================
# PART 1: PHYSICS MODEL
# ==========================================
class MechanicalModel:
    def __init__(self):
        self.K_motor = 1.8
        self.I_no_load = 0.5
        self.d_shaft = 0.030
        self.r_shaft = self.d_shaft / 2
        self.shaft_len = 0.350
        self.w_key = 0.008
        self.h_key = 0.007
        self.l_key = 0.040
        self.hub_len = 0.080
        self.motor_radius = 0.120
        self.motor_length = 0.220
        self.base_width = 0.280
        self.base_length = 0.300
        self.base_thickness = 0.030
        self.yield_strength = 16.5e6

    def get_data(self, current):
        torque = max(0.0, self.K_motor * (current - self.I_no_load))
        if torque <= 0:
            shear, crushing = 0.0, 0.0
        else:
            shear = (2 * torque) / (self.d_shaft * self.w_key * self.l_key)
            crushing = (4 * torque) / (self.d_shaft * self.h_key * self.l_key)

        max_stress = max(shear, crushing)
        ratio = min(max_stress / self.yield_strength, 1.0)

        if ratio <= 0.25:
            color = (0.0, ratio / 0.25, 1.0)
        elif ratio <= 0.50:
            color = (0.0, 1.0, 1.0 - ((ratio - 0.25) / 0.25))
        elif ratio <= 0.75:
            color = (((ratio - 0.50) / 0.25), 1.0, 0.0)
        else:
            color = (1.0, 1.0 - ((ratio - 0.75) / 0.25), 0.0)

        return torque, shear, crushing, color, max_stress


# ==========================================
# PART 2: 3D VISUALIZATION SETUP
# ==========================================
model = MechanicalModel()

plotter = pv.Plotter(lighting='none')
plotter.set_background("white")
plotter.enable_anti_aliasing("ssaa")

shaft_geo = pv.Cylinder(center=(0, 0, model.shaft_len / 2 - 0.05), direction=(0, 0, 1), radius=model.r_shaft,
                        height=model.shaft_len, resolution=60)
key_z_center = (model.shaft_len - 0.05) - (model.l_key / 2) - 0.01
key_geo = pv.Box(
    bounds=(-model.w_key / 2, model.w_key / 2, model.r_shaft - model.h_key / 2, model.r_shaft + model.h_key / 2,
            key_z_center - model.l_key / 2, key_z_center + model.l_key / 2))
shaft_with_keyway = shaft_geo.triangulate().boolean_difference(key_geo.triangulate())
hub_geo = pv.Cylinder(center=(0, 0, key_z_center), direction=(0, 0, 1), radius=model.r_shaft * 3.5,
                      height=model.hub_len, resolution=60)

motor_z_center = 0.04
motor_geo = pv.Cylinder(center=(0, 0, motor_z_center), direction=(0, 0, 1), radius=model.motor_radius,
                        height=model.motor_length, resolution=80)
encoder_geo = pv.Cylinder(center=(0, 0, motor_z_center + (model.motor_length / 2) + 0.02), direction=(0, 0, 1),
                          radius=model.r_shaft + 0.015, height=0.03, resolution=40)

base_y_max = -model.motor_radius
base_z_center = motor_z_center
base_geo = pv.Box(bounds=(-model.base_width / 2, model.base_width / 2, base_y_max - model.base_thickness, base_y_max,
                          base_z_center - model.base_length / 2, base_z_center + model.base_length / 2))
sensor_box_geo = pv.Box(
    bounds=(model.motor_radius - 0.01, model.motor_radius + 0.04, -0.03, 0.05, motor_z_center - 0.05,
            motor_z_center + 0.05))

bolts = []
for bx in [-(model.base_width / 2 - 0.03), (model.base_width / 2 - 0.03)]:
    for bz in [-(model.base_length / 2 - 0.03), (model.base_length / 2 - 0.03)]:
        bolts.append(pv.Cylinder(center=(bx, base_y_max + 0.005, base_z_center + bz), direction=(0, 1, 0), radius=0.01,
                                 height=0.04))
bolts_geo = bolts[0].merge([bolts[1], bolts[2], bolts[3]])

shaft_actor = plotter.add_mesh(shaft_with_keyway, color="lightgrey", pbr=True, metallic=0.8, roughness=0.4)
hub_actor = plotter.add_mesh(hub_geo, color="cyan", opacity=0.3, pbr=True, metallic=0.2, roughness=0.1)
key_actor = plotter.add_mesh(key_geo, color="blue", smooth_shading=True, pbr=True, metallic=0.3, roughness=0.2)
encoder_actor = plotter.add_mesh(encoder_geo, color="darkgrey", pbr=True, metallic=0.7, roughness=0.3)
motor_actor = plotter.add_mesh(motor_geo, color="#34495e", pbr=True, metallic=0.5, roughness=0.6)
base_actor = plotter.add_mesh(base_geo, color="#7f8c8d", pbr=True, metallic=0.3, roughness=0.7)
sensor_box_actor = plotter.add_mesh(sensor_box_geo, color="#bdc3c7")
bolts_actor = plotter.add_mesh(bolts_geo, color="silver", pbr=True, metallic=0.9, roughness=0.2)

plotter.add_light(pv.Light(position=(5, 5, 10), color='white', intensity=1.5))
plotter.add_light(pv.Light(position=(-5, -5, 5), color='#d6e4f0', intensity=0.6))
plotter.add_light(pv.Light(position=(0, -5, -10), color='#fff9e6', intensity=0.8))

# ==========================================
# PART 3: MATPLOTLIB INTERACTIVE DASHBOARD
# ==========================================
plt.ion()
fig, ax = plt.subplots(figsize=(6, 5))
fig.canvas.manager.set_window_title('Digital Twin Analytics')

theoretical_torques = np.linspace(0, 40, 50)
theoretical_stresses = (2 * theoretical_torques) / (model.d_shaft * model.w_key * model.l_key) / 1e6
ax.plot(theoretical_torques, theoretical_stresses, color="gray", linewidth=2, linestyle="--",
        label="Theoretical Relationship")

# Initialize the live red marker
# Initialize the live marker (Start as green!)
live_marker, = ax.plot([], [], marker='o', color='green', markersize=10, markeredgecolor='black', label="Live Data")

ax.set_title("Shear Stress vs. Input Torque", fontsize=14, fontweight='bold')
ax.set_xlabel("Input Torque (Nm)", fontsize=12)
ax.set_ylabel("Shear Stress (MPa)", fontsize=12)
ax.set_xlim(0, 40)
ax.set_ylim(0, max(theoretical_stresses) + 5)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()
fig.tight_layout()
fig.show()

# --- STATE MANAGEMENT ---
state = {
    "current": 0.0,
    "rpm": 60.0,
    "angle": 0.0,
    "last_time": time.time(),
    "running": True,
    "needs_ui_update": True  # Flag to control when redraws happen safely
}


# ==========================================
# PART 4: SAFE UI CALLBACKS
# ==========================================
# The slider NO LONGER renders graphics. It only updates the state variable.
def slider_callback(value):
    state["current"] = value
    state["needs_ui_update"] = True


plotter.add_slider_widget(slider_callback, [0, 20], value=0, title="Motor Current (A)", pointa=(0.6, 0.9),
                          pointb=(0.95, 0.9), style='modern', color='black')


def toggle_animation(flag):
    state["running"] = flag
    state["last_time"] = time.time()


plotter.add_checkbox_button_widget(toggle_animation, value=True, position=(10, 10), size=35, border_size=2,
                                   color_on='green', color_off='red', background_color='grey')
plotter.add_text("Play / Pause", position=(55, 15), font='arial', font_size=16, color="black")

# ==========================================
# PART 5: THE GAME LOOP (THREAD-SAFE)
# ==========================================
print("Digital Twin Running. Close the PyVista window to exit.")
plotter.show(interactive_update=True)

while True:
    try:
        # 1. Update Physics & UI (Only if slider moved)
        if state["needs_ui_update"]:
            torque, shear, crush, rgb_color, stress_val = model.get_data(state["current"])
            
            # Update 3D Colors
            key_actor.prop.color = rgb_color
            
            # Update 2D Chart Data
            live_marker.set_data([torque], [shear / 1e6])
            
            # --- NEW LOGIC: Dynamic Graph Dot Color & Visibility ---
            if state["current"] < 3.0:
                # Hide the dot completely if under 3 Amps
                live_marker.set_visible(False)
            else:
                # Show the dot and determine its color
                live_marker.set_visible(True)
                if state["current"] > 18.6:
                    live_marker.set_color('red')
                else:
                    live_marker.set_color('green')
            
            fig.canvas.draw_idle()
            
            # Update HUD Text
            status = "CRITICAL" if stress_val > model.yield_strength else "NORMAL"
            text = f"SHAFT-KEY ASSEMBLY\n------------------\nCurrent:  {state['current']:>5.1f} A\nTorque:   {torque:>5.1f} Nm\nStress:   {stress_val / 1e6:>5.1f} MPa\nLimit:    {model.yield_strength / 1e6:>5.0f} MPa\nStatus:   {status}"
            plotter.add_text(text, position='upper_left', font='courier', font_size=14, color='black', name='hud')
            
            state["needs_ui_update"] = False

        # 2. Update Animation Angles (Requires at least 3 Amps to spin)
        if state["running"] and state["current"] >= 3.0:
            dt = time.time() - state["last_time"]
            omega = (2 * np.pi * state["rpm"]) / 60.0
            state["angle"] += omega * dt
            transform = pv.transformations.axis_angle_rotation(axis=(0, 0, 1), angle=np.degrees(state["angle"]), point=(0, 0, 0))
            
            shaft_actor.user_matrix = transform
            hub_actor.user_matrix = transform
            key_actor.user_matrix = transform
            encoder_actor.user_matrix = transform  

        state["last_time"] = time.time()

        # 3. Master Render Step (Happens safely once per frame)
        plotter.update()
        fig.canvas.flush_events()
        time.sleep(0.015)

    except Exception as e:
        print(f"Simulation Closed. Reason: {e}")
        break