# Unitree MuJoCo G1 Primer Workshop

## Three-Hour Instructor-Guided Workshop

This workshop introduces the Unitree G1 humanoid robot model in MuJoCo using Windows 11, Ubuntu in WSL 2, Visual Studio Code, and a project stored under `C:\StudentWork`.

It covers installation, model inspection, joint-state reading, bounded torque commands, PD position control, CSV logging, creation of a course-owned fixed-base G1 model, whole-body stabilization, gravity/bias compensation, and deterministic repeated testing.

This workshop deliberately stops before the Gymnasium and DQN implementation. The validated controller developed here will become the foundation for that next stage.

## Learning Outcomes

By the end of this workshop, students should be able to:

1. Explain the purpose of MuJoCo in humanoid-robot simulation.
2. Describe bodies, joints, actuators, sensors, degrees of freedom, `qpos`, and `qvel`.
3. Load and inspect the Unitree G1 29-DOF model.
4. Read G1 joint positions and velocities.
5. Send bounded torque commands to a selected actuator.
6. Implement a PD position controller.
7. Record simulation observations to CSV.
8. Explain why a fixed-base model is useful for introductory RL.
9. Stabilize all non-controlled joints.
10. Use MuJoCo bias forces to improve tracking accuracy.
11. Verify deterministic simulation behaviour.

## Environment and College Constraints

- Windows 11
- Visual Studio Code
- WSL 2 with Ubuntu 24.04
- WSLg graphical support
- Project path in Windows:

```text
C:\StudentWork\Code\CSCN8020\Unitree_MuJoCo_G1_Primer_Workshop
```

- Project path in WSL:

```text
/mnt/c/StudentWork/Code/CSCN8020/Unitree_MuJoCo_G1_Primer_Workshop
```

College-specific conditions:

- Work remains under `C:\StudentWork`.
- WSL runs through the `root` account.
- Unitree's repository is cloned under `external/`.
- Official Unitree files are not modified directly.
- Training-oriented simulation runs headlessly.
- The viewer is mainly for inspection and demonstrations.

## Three-Hour Schedule

| Time | Activity |
|---:|---|
| 0:00–0:20 | Humanoid robotics and MuJoCo primer |
| 0:20–0:45 | Environment verification and installation |
| 0:45–1:15 | Load and inspect the Unitree G1 |
| 1:15–1:45 | Read joint state and implement one-joint control |
| 1:45–2:10 | Record observations and improve viewer lifecycle |
| 2:10–2:35 | Create and validate the fixed-base G1 |
| 2:35–2:50 | Stabilize non-controlled joints |
| 2:50–3:00 | Add bias compensation and assess the milestone |

# Part 1 — Robotics and MuJoCo Primer

This section explains the terms that appear later in the workshop output and code.

### Scene

A **scene** is the complete MuJoCo simulation description loaded from an XML file. It may include:

- The robot model
- The ground or other environment objects
- Lighting
- Cameras
- Initial poses
- Keyframes
- Physics settings

For this workshop, the scene file is typically:

```text
scene_29dof.xml
```

or the course-owned fixed-base version:

```text
scene_29dof_fixed_base.xml
```

### Body

A **body** is a rigid component of the robot, such as the pelvis, torso, upper arm, forearm, or hand.

### Joint

A **joint** connects two bodies and defines how they may move relative to one another.

For example:

```text
Joint: left_elbow_joint
```

means the MuJoCo joint named `left_elbow_joint` is being controlled and observed. It is a hinge joint representing the rotational motion of the G1's left elbow.

### Actuator

An **actuator** is the simulated motor or force-generating mechanism that applies control to a joint.

For example:

```text
Actuator: left_elbow
```

means the actuator named `left_elbow` applies torque to `left_elbow_joint`.

The joint describes the allowed motion. The actuator applies the control effort.

### Joint ID

The **Joint ID** is MuJoCo's internal numeric identifier for a joint after the XML model has been loaded.

For example:

```text
Joint ID: 18
```

means MuJoCo stored `left_elbow_joint` as joint number 18 in the loaded model.

IDs should not be assumed in advance. The code looks them up by name.

### Actuator ID

The **Actuator ID** is MuJoCo's internal numeric identifier for an actuator.

For example:

```text
Actuator ID: 18
```

means the `left_elbow` actuator is actuator number 18 in the loaded model.

The joint ID and actuator ID may happen to be the same number, but they represent different types of objects.

### Degree of Freedom

A **degree of freedom**, or DOF, is an independent direction of motion.

The original G1 scene contains 29 actuated robot joints plus six floating-base velocity DOFs. The fixed-base teaching model contains only the 29 actuated joint DOFs.

### `qpos`

`qpos` is MuJoCo's generalized position vector.

It stores position information such as:

- Joint angles
- Sliding-joint positions
- Floating-base position
- Floating-base orientation

### `qpos index`

The **`qpos index`** is the position in the `data.qpos` array where the selected joint's position is stored.

For example:

```text
qpos index: 18
```

means the left-elbow angle is read using:

```python
data.qpos[18]
```

### `qvel`

`qvel` is MuJoCo's generalized velocity vector.

It stores joint angular velocities, joint linear velocities, and floating-base velocities.

### `qvel index`

The **`qvel index`** is the position in the `data.qvel` array where the selected joint's velocity is stored.

For example:

```text
qvel index: 18
```

means the left-elbow angular velocity is read using:

```python
data.qvel[18]
```

### Joint range

The **joint range** is the permitted angular interval for a hinge joint.

For example:

```text
Joint range: [-1.047, 2.094] rad
```

means the elbow angle is limited to approximately:

- `-1.047 rad`, or `-60°`
- `2.094 rad`, or `120°`

The controller clips requested targets so they remain inside this range.

### Radians (`rad`)

A **radian** is the standard angular unit used by MuJoCo.

Useful conversions include:

$$
\pi \text{ rad} = 180^\circ
$$

$$
1 \text{ rad} \approx 57.3^\circ
$$

For example:

$$
-0.8 \text{ rad} \approx -45.8^\circ
$$

### Torque

**Torque** is the rotational effect produced by an actuator.

For a rotational joint such as the elbow, torque is the control effort that causes angular acceleration or resists gravity and other forces.

### Torque range

The **torque range** is the minimum and maximum torque that the actuator is allowed to command.

For example:

```text
Torque range: [-25.000, 25.000] Nm
```

means the actuator command is limited to between `-25 Nm` and `25 Nm`.

This is not the same as the joint-angle range.

### Newton-metre (`Nm`)

A **newton-metre**, written `Nm`, is the standard unit of torque.

A positive or negative value indicates torque in opposite rotational directions around the joint axis.

### Requested target

The **requested target** is the angle supplied by the user through the command line.

For example:

```text
Requested target: -0.800 rad
```

means the user asked the controller to move the elbow to `-0.8 rad`.

### Applied target

The **applied target** is the target angle actually used after safety clipping.

For example:

```text
Applied target: -0.800 rad
```

means the requested value was already inside the joint range, so it was used without modification.

If the user requested an unsafe value outside the joint range, the applied target would be clipped to the closest valid limit.

### Sensor

A **sensor** is a simulated measurement source. The G1 model includes:

- Joint-position sensors
- Joint-velocity sensors
- Joint-torque sensors
- IMU orientation
- IMU angular velocity
- IMU linear acceleration
- Frame position and velocity

### PD controller

A proportional-derivative controller converts a target angle into torque:

$$
\tau = K_p(q_{target}-q)-K_d\dot q
$$

where:

- $\tau$ is the commanded torque.
- $K_p$ is the proportional gain.
- $K_d$ is the derivative gain.
- $q_{target}$ is the desired joint angle.
- $q$ is the current joint angle.
- $\dot q$ is the current joint angular velocity.

The proportional term reacts to angle error. The derivative term reduces oscillation by opposing excessive velocity.


# Part 2 — Verify the WSL Environment

Run in the VS Code WSL terminal:

```bash
cat /etc/os-release
nvidia-smi
python3 --version
code --version
echo $DISPLAY
echo $WAYLAND_DISPLAY
df -h /
```

Expected indicators:

- Ubuntu 24.04
- WSLg display values such as `:0` and `wayland-0`
- Sufficient disk space
- NVIDIA visibility when a compatible GPU is available


# Part 3 — Install Ubuntu Dependencies

Before installing MuJoCo or Unitree software, install the Linux packages required for Python virtual environments, C/C++ compilation, CMake/Ninja builds, OpenGL rendering, GLFW window management, and WSLg graphics.

Run:

```bash
apt update

apt install -y \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    cmake \
    ninja-build \
    pkg-config \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev
```

### Why this step matters

MuJoCo is installed with `pip`, but its viewer and some Unitree tools depend on system-level graphics and build libraries.

### Expected result

The installation should finish without unresolved dependency errors.


# Part 4 — Create the Python Environment

Create a project-local virtual environment so this workshop does not interfere with other Python installations.

```bash
cd /mnt/c/StudentWork/Code/CSCN8020/Unitree_MuJoCo_G1_Primer_Workshop

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### What these commands do

- `python3 -m venv .venv` creates an isolated environment.
- `source .venv/bin/activate` activates it.
- Upgrading `pip`, `setuptools`, and `wheel` reduces package-installation problems.

Verify:

```bash
which python
python --version
pip --version
```

The Python path should point inside `.venv`. Activate this environment whenever you return to the project.


# Part 5 — Install and Validate MuJoCo

Install MuJoCo in the active environment:

```bash
pip install mujoco
```

Verify:

```bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

### Why validate MuJoCo first

This separates general MuJoCo or graphics problems from Unitree-specific problems. If the import and viewer work, Python, MuJoCo, OpenGL, GLFW, and WSLg are functioning.

The validated environment used MuJoCo 3.10.0 and Python 3.12.3.

## Basic Viewer Test

Create `test_mujoco_viewer.py` in the project root.

```python
import time

import mujoco
import mujoco.viewer


MODEL_XML = """
<mujoco model="viewer_test">
    <worldbody>
        <light pos="0 0 3"/>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
```

Run:

```bash
python test_mujoco_viewer.py
```

Checkpoint:

- The viewer opens.
- The box falls onto the plane.
- Closing the viewer ends the program.


# Part 6 — Clone Unitree MuJoCo

Clone the official Unitree repository into an `external/` directory:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
```

### Why use `external/`

This keeps third-party Unitree files separate from course-owned files, discourages accidental edits, and allows the course repository to ignore the large dependency.

Inspect the clone:

```bash
cd external/unitree_mujoco

git status
git log -1 --oneline
git rev-parse HEAD
find unitree_robots/g1 -maxdepth 2 -type f | sort
```

Locate G1 XML files:

```bash
find unitree_robots/g1 \
  -type f \
  \( -name "*.xml" -o -name "*.mjcf" \) \
  -printf "%P\n" | sort
```

Expected files include `g1_29dof.xml` and `scene_29dof.xml`. The validated upstream revision is `ae6a8403e272733e9996ef59990880330496177f` (`ae6a840`).

### Windows filesystem warnings

A collision between `terrain.STL` and `terrain.stl`, or files ending in `:Zone.Identifier`, may appear under `/mnt/c`. These do not affect the G1 workshop.


# Part 7 — Inspect the G1 Model

Create the source folder and inspection script:

```bash
mkdir -p src
touch src/inspect_g1_model.py
```

### Why create this script

Before writing control code, students should see how MuJoCo represents the G1. The script reports bodies, joints, DOFs, actuators, sensors, names, IDs, indices, and torque limits.

Copy the following code into `src/inspect_g1_model.py`.

```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def get_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"<unnamed-{object_id}>"


def print_model_summary(model):
    print("\n=== G1 MODEL SUMMARY ===")
    print(f"Bodies:              {model.nbody}")
    print(f"Joints:              {model.njnt}")
    print(f"Degrees of freedom:  {model.nv}")
    print(f"Position variables:  {model.nq}")
    print(f"Actuators:           {model.nu}")
    print(f"Sensors:             {model.nsensor}")
    print(f"Simulation timestep: {model.opt.timestep:.6f} seconds")


def print_joint_information(model):
    joint_type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }

    print("\n=== JOINTS ===")
    for joint_id in range(model.njnt):
        name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        print(
            f"{joint_id:2d} | {name:35s} | "
            f"type={joint_type_names.get(joint_type, str(joint_type)):6s} | "
            f"qpos_address={model.jnt_qposadr[joint_id]:2d} | "
            f"dof_address={model.jnt_dofadr[joint_id]:2d}"
        )


def print_actuator_information(model):
    print("\n=== ACTUATORS ===")
    for actuator_id in range(model.nu):
        name = get_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        low, high = model.actuator_ctrlrange[actuator_id]
        print(
            f"{actuator_id:2d} | {name:35s} | "
            f"control range=[{low:9.3f}, {high:9.3f}]"
        )


def print_sensor_information(model):
    print("\n=== SENSORS ===")
    for sensor_id in range(model.nsensor):
        name = get_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        print(
            f"{sensor_id:2d} | {name:35s} | "
            f"dimension={model.sensor_dim[sensor_id]:2d} | "
            f"data_address={model.sensor_adr[sensor_id]:3d}"
        )


def run_viewer(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            delay = model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if delay > 0:
                time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    scene_path = args.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    print_model_summary(model)
    print_joint_information(model)
    print_actuator_information(model)
    print_sensor_information(model)

    if not args.no_viewer:
        run_viewer(model)


if __name__ == "__main__":
    main()
```

Run:

```bash
python src/inspect_g1_model.py \
  external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --no-viewer
```

Then:

```bash
python src/inspect_g1_model.py \
  external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

Validated original-scene summary:

```text
Bodies:              31
Joints:              30
Degrees of freedom:  35
Position variables:  36
Actuators:           29
Sensors:             95
Simulation timestep: 0.002000 seconds
```


# Part 8 — Single-Joint Controller

Create:

```bash
touch src/control_single_joint.py
```

Copy the complete controller code supplied in this workshop into that file.

### Why this controller is needed

The G1 model accepts torque, not a desired angle directly. The controller therefore:

1. Finds the joint and actuator by name.
2. Reads angle and angular velocity.
3. Computes PD torque.
4. Adds MuJoCo bias-force compensation.
5. Clips torque to actuator limits.
6. Holds all non-controlled joints.
7. Logs observations to CSV.
8. Supports headless and viewer execution.

Headless mode is the required mode for repeated tests and future RL training.

## Run the Controller on the Original Scene

```bash
python src/control_single_joint.py \
  --scene external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --target -0.8 \
  --duration 2 \
  --no-viewer
```

The robot may fall because the original scene has a floating base and no balance controller.

## Viewer Shutdown Note

Under WSLg, close rendered controller and environment windows manually when prompted. This avoids the native viewer crash caused by immediate programmatic shutdown.

In the validated environment:

- Physics completed correctly.
- Results were printed.
- CSV output was written.
- Headless execution was stable.

Use `--no-viewer` for training and repeated experiments.


# Part 9 — Preserve the Working Environment

Since the project must remain reproducible under college restrictions, save both the exact working environment and a simple student installation file.

## 9.1 Create the exact lock file

```bash
pip freeze > requirements-lock.txt
```

### Why `requirements-lock.txt`

This records every installed package and exact version, including indirect dependencies installed automatically by `pip`.

Use it for:

- Reproducing the instructor's exact working environment
- Diagnosing version differences
- Restoring a known-good installation

It is deliberately not named `requirements.txt` because it contains many low-level dependencies students do not need to manage directly.

## 9.2 Create the standard student-facing file

```bash
cat > requirements.txt <<'EOF'
mujoco==3.10.0
numpy==2.5.1
EOF
```

Students can install the direct dependencies with:

```bash
pip install -r requirements.txt
```

| File | Purpose |
|---|---|
| `requirements.txt` | Simple student installation |
| `requirements-lock.txt` | Exact reproduction and troubleshooting |

## 9.3 Create `.gitignore`

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.py[cod]
.vscode/
external/
logs/
results/
*.csv
EOF
```

### Why these items are ignored

- `.venv/`: local Python environment
- `__pycache__/`: generated bytecode
- `.vscode/`: local editor settings
- `external/`: cloned third-party repository
- `logs/`, `results/`, `*.csv`: generated experiment output

This keeps the course repository clean and small.


# Part 10 — Create a Course-Owned Fixed-Base G1

The original Unitree G1 scene uses a floating base. This allows the humanoid to translate, rotate, and fall.

That behaviour is appropriate for locomotion and balance research, but it introduces unnecessary complexity for an introductory elbow-control and reinforcement-learning task.

In this step, you will create a separate course-owned fixed-base version of the G1 model.

The original Unitree files will remain unchanged.

## 10.1 Create the required files and directories

From the project root, run:

```bash
mkdir -p assets
touch src/create_fixed_base_g1.py
```

This creates:

- The `assets/` directory, where course-owned robot assets will be stored
- The Python script that will generate the fixed-base model

Open:

```text
src/create_fixed_base_g1.py
```

Replace its contents with the complete code below.

```python
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_MODEL = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/g1_29dof.xml"
)

SOURCE_SCENE = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/scene_29dof.xml"
)

OUTPUT_DIRECTORY = Path("assets/g1_fixed_base")

OUTPUT_MODEL = OUTPUT_DIRECTORY / "g1_29dof_fixed_base.xml"
OUTPUT_SCENE = OUTPUT_DIRECTORY / "scene_29dof_fixed_base.xml"

SOURCE_MESH_DIRECTORY = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/meshes"
)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indentation = "\n" + level * "  "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "

        for child in element:
            indent_xml(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = indentation

    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def create_fixed_base_model() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Source G1 model was not found: {SOURCE_MODEL}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(SOURCE_MODEL)
    root = tree.getroot()

    compiler = root.find("compiler")

    if compiler is None:
        raise RuntimeError(
            "The source model does not contain a compiler element."
        )

    compiler.set("meshdir", "meshes")

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "The source model does not contain a worldbody element."
        )

    pelvis = worldbody.find(".//body[@name='pelvis']")

    if pelvis is None:
        raise RuntimeError(
            "The pelvis body was not found in the G1 model."
        )

    # MuJoCo supports either:
    #   <freejoint name="..."/>
    # or:
    #   <joint name="..." type="free"/>
    free_joint_candidates = list(pelvis.findall("freejoint"))

    free_joint_candidates.extend(
        joint
        for joint in pelvis.findall("joint")
        if joint.get("type") == "free"
    )

    if len(free_joint_candidates) != 1:
        child_descriptions = [
            {
                "tag": child.tag,
                "name": child.get("name"),
                "type": child.get("type"),
            }
            for child in pelvis
            if child.tag in {"joint", "freejoint"}
        ]

        raise RuntimeError(
            "Expected exactly one free joint inside the pelvis, "
            f"but found {len(free_joint_candidates)}. "
            f"Pelvis joint elements: {child_descriptions}"
        )

    free_joint = free_joint_candidates[0]

    removed_joint_name = free_joint.get(
        "name",
        "<unnamed free joint>",
    )

    pelvis.remove(free_joint)

    # Place the fixed pelvis at a practical height above the floor.
    pelvis.set("pos", "0 0 0.80")
    pelvis.set("quat", "1 0 0 0")

    indent_xml(root)

    tree.write(
        OUTPUT_MODEL,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Removed free joint: {removed_joint_name}")
    print(f"Created fixed-base model: {OUTPUT_MODEL}")


def copy_meshes() -> None:
    destination = OUTPUT_DIRECTORY / "meshes"

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(
        SOURCE_MESH_DIRECTORY,
        destination,
    )

    print(f"Copied meshes to: {destination}")


def create_fixed_base_scene() -> None:
    if not SOURCE_SCENE.is_file():
        raise FileNotFoundError(
            f"Source G1 scene was not found: {SOURCE_SCENE}"
        )

    tree = ET.parse(SOURCE_SCENE)
    root = tree.getroot()

    include = root.find("include")

    if include is None:
        raise RuntimeError(
            "The source scene does not contain an include element."
        )

    include.set(
        "file",
        OUTPUT_MODEL.name,
    )

    keyframe = root.find("keyframe")

    if keyframe is not None:
        root.remove(keyframe)

    indent_xml(root)

    tree.write(
        OUTPUT_SCENE,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Created fixed-base scene: {OUTPUT_SCENE}")


def main() -> None:
    create_fixed_base_model()
    copy_meshes()
    create_fixed_base_scene()

    print("\nFixed-base G1 assets created successfully.")


if __name__ == "__main__":
    main()
```

## 10.2 Why generate a new model instead of editing Unitree files?

The official Unitree files are stored under:

```text
external/unitree_mujoco/
```

These files should be treated as third-party source code.

Creating a separate course-owned model:

- Preserves the original Unitree model
- Makes the workshop modifications explicit
- Allows easy comparison with the upstream model
- Prevents future Unitree updates from overwriting course changes
- Keeps all teaching-specific assets under `assets/`
- Makes the workshop easier to reproduce and troubleshoot

Do not manually edit the original XML files under:

```text
external/unitree_mujoco/
```

## 10.3 What the generator changes

The script performs the following operations:

1. Loads the official `g1_29dof.xml` model.
2. Finds the body named `pelvis`.
3. Searches for either:
   - `<freejoint>`
   - `<joint type="free">`
4. Removes the floating-base joint.
5. Places the pelvis at:

```text
0 0 0.80
```

6. Sets the pelvis orientation to the identity quaternion:

```text
1 0 0 0
```

7. Copies the G1 mesh directory.
8. Creates a new fixed-base model XML file.
9. Creates a new scene XML file.
10. Updates the scene to load the fixed-base model.
11. Removes the original keyframe because it may contain floating-base position values that no longer match the new model.

The generated files will be stored under:

```text
assets/g1_fixed_base/
```

## 10.4 Run the generator

From the project root, run:

```bash
python src/create_fixed_base_g1.py
```

Expected output:

```text
Removed free joint: floating_base_joint
Created fixed-base model: assets/g1_fixed_base/g1_29dof_fixed_base.xml
Copied meshes to: assets/g1_fixed_base/meshes
Created fixed-base scene: assets/g1_fixed_base/scene_29dof_fixed_base.xml

Fixed-base G1 assets created successfully.
```

The exact free-joint name may differ if the upstream XML uses an unnamed free joint.

## 10.5 Confirm that the floating-base joint was removed

Run:

```bash
grep -nE 'freejoint|type="free"' \
  assets/g1_fixed_base/g1_29dof_fixed_base.xml
```

Expected result:

```text
No output
```

No output means the generated model no longer contains a floating-base joint.

## 10.6 Inspect the generated model

Run:

```bash
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --no-viewer
```

Expected model summary:

```text
Bodies:              31
Joints:              29
Degrees of freedom:  29
Position variables:  29
Actuators:           29
Sensors:             95
Simulation timestep: 0.002000 seconds
```

The original floating-base scene had:

```text
30 joints
35 velocity DOFs
36 position variables
```

The fixed-base model has:

```text
29 joints
29 velocity DOFs
29 position variables
```

This confirms that the floating base was removed while preserving the 29 actuated robot joints.

## 10.7 Open the fixed-base model in the viewer

After the headless inspection succeeds, run:

```bash
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml
```

Expected behaviour:

- The G1 appears in the MuJoCo viewer.
- The pelvis remains fixed in space.
- The robot does not fall as a complete floating body.
- Individual joints can still move.

Close the viewer window when the inspection is complete.

Under WSLg, a segmentation fault may occasionally appear after the viewer closes. If the model loaded, displayed, and completed its output correctly, treat this as the known viewer-shutdown issue.

```bash
python src/create_fixed_base_g1.py
```

Expected output:

```text
Removed free joint: floating_base_joint
Created fixed-base model: assets/g1_fixed_base/g1_29dof_fixed_base.xml
Copied meshes to: assets/g1_fixed_base/meshes
Created fixed-base scene: assets/g1_fixed_base/scene_29dof_fixed_base.xml

Fixed-base G1 assets created successfully.
```

Verify:

```bash
grep -nE 'freejoint|type="free"' \
  assets/g1_fixed_base/g1_29dof_fixed_base.xml
```

No output should appear.


# Part 11 — Validate the Fixed-Base Model

Inspect the generated model headlessly first:

```bash
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --no-viewer
```

### Why headless first

This confirms that the XML and mesh paths are valid and that the model loads before graphics are introduced.

Validated summary:

```text
Bodies:              31
Joints:              29
Degrees of freedom:  29
Position variables:  29
Actuators:           29
Sensors:             95
Simulation timestep: 0.002000 seconds
```

Then open the viewer:

```bash
python src/inspect_g1_model.py \
  assets/g1_fixed_base/scene_29dof_fixed_base.xml
```

The pelvis should remain fixed. The left elbow maps to joint, actuator, `qpos`, and `qvel` index 18.


# Part 12 — Deterministic Elbow Test

Run once:

```bash
python src/control_single_joint.py \
  --scene assets/g1_fixed_base/scene_29dof_fixed_base.xml \
  --target -0.8 \
  --duration 2 \
  --no-viewer
```

### Why use `--no-viewer`

Headless mode runs faster, avoids WSLg shutdown problems, and matches future RL training.

Repeat five times:

```bash
for run in {1..5}; do
    python src/control_single_joint.py \
      --scene assets/g1_fixed_base/scene_29dof_fixed_base.xml \
      --target -0.8 \
      --duration 2 \
      --no-viewer
done
```

Check that final angle and error are identical or extremely close.

Validated result:

```text
Final angle: -0.7968 rad
Final error: -0.0032 rad
```

This confirms that the controlled plant is deterministic enough for Gymnasium and DQN.


# Part 13 — Why Bias Compensation Matters

A pure PD controller must maintain some error to produce the torque needed to oppose gravity.

MuJoCo provides generalized bias forces through:

```python
data.qfrc_bias
```

The controller adds:

```python
commanded_torque = pd_torque + bias_torque
```

This reduced the validated error from about `0.0981 rad` to `0.0032 rad`.

$$
0.0032 \text{ rad} \approx 0.18^\circ
$$

### Why this matters for RL

The future DQN agent should learn high-level discrete decisions, not spend its capacity learning predictable gravity compensation.


# Part 14 — Inspect the Logged Data

The controller writes:

```text
results/single_joint_control.csv
```

Inspect it:

```bash
head results/single_joint_control.csv
tail results/single_joint_control.csv
```

### Why record the data

The CSV provides evidence that the controller works and supports later analysis of rise time, overshoot, steady-state error, velocity, torque demand, and controller tuning.

Columns:

- `simulation_time`
- `target_angle_rad`
- `joint_angle_rad`
- `joint_velocity_rad_s`
- `angle_error_rad`
- `commanded_torque_nm`

Generated CSV files are ignored by Git because they are experiment outputs, not source files.


# Current Milestone Status

| Milestone | Status | Verified Result |
|---|---|---|
| WSL 2 and Ubuntu | Complete | Ubuntu 24.04 |
| WSLg graphics | Complete | Viewer opened |
| NVIDIA visibility | Complete | RTX A500 visible |
| System dependencies | Complete | Installed successfully |
| Python environment | Complete | Project-local `.venv` |
| MuJoCo installation | Complete | Version 3.10.0 |
| Basic viewer test | Complete | Falling-box test worked |
| Unitree repository clone | Complete | Stored under `external/` |
| G1 29-DOF loading | Complete | Model loaded |
| Model inspection | Complete | Joints, actuators, sensors listed |
| Joint-state reading | Complete | Position and velocity available |
| Torque commands | Complete | Left elbow responded |
| PD position control | Complete | Target tracking worked |
| CSV recording | Complete | Data logged |
| Viewer shutdown issue | Documented | WSLg-only; headless stable |
| Course-owned fixed-base scene | Complete | Original files untouched |
| Fixed-base validation | Complete | 29 joints and 29 DOFs |
| Non-controlled joint stabilization | Complete | Whole-body hold controller |
| Bias compensation | Complete | `qfrc_bias` added |
| Deterministic repeated runs | Complete | Identical results |
| Accurate elbow tracking | Complete | About 0.0032 rad error |
| Gymnasium environment | Next | Not implemented |
| Student-written DQN | Later | Not implemented |
| Pretrained policy playback | Later | Not implemented |
| SDK2 and Cyclone DDS | Later | Not installed |
| Physical G1 deployment | Later | Instructor-controlled stage |
| Assignment 3 revision | Later | To be completed |


# Reflection Questions

1. Why is the actuator control range different from the joint-angle range?
2. Why does the original scene have more velocity DOFs than actuators?
3. Why is a floating-base humanoid harder to control?
4. Why should official Unitree XML files remain unchanged?
5. Why does holding the other joints improve repeatability?
6. Why does bias compensation reduce steady-state error?
7. Why should DQN training run headlessly?
8. Which controller values should become part of the future Gymnasium observation?


# Troubleshooting

### MuJoCo import fails

```bash
source .venv/bin/activate
pip install mujoco
```

### Viewer does not open

```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
```

### Scene file not found

Return to the project root.

### Git collision or `Zone.Identifier`

Expected under `/mnt/c`; does not affect this G1 workshop.

### Fixed-base generator finds zero free joints

Support both `<freejoint>` and `<joint type="free">`.

### Viewer exits with `Segmentation fault`

Use `--no-viewer`. Headless simulation is the required training mode.

### Repeated results differ

Confirm the fixed-base scene, deterministic reset, non-controlled-joint holding, and absence of random perturbations.


# Workshop Completion

The installation and basic-interface milestone is complete when:

- The G1 model loads.
- The fixed-base model contains 29 joints and 29 DOFs.
- The elbow reaches `-0.8 rad`.
- Five headless runs produce the same result.
- Final absolute error is close to `0.0032 rad`.

The next approved development stage is a small Gymnasium environment around this controller.


# Part 15 — Build the Gymnasium Environment

The validated MuJoCo controller will now be wrapped in a small Gymnasium environment.

This environment becomes the foundation for the later student-written DQN assignment.

## Environment design

### Observation

Each observation contains four values:

```text
[
    current elbow angle,
    current elbow velocity,
    goal angle,
    angle error
]
```

The angle error is:

$$
e=q_{goal}-q
$$

### Actions

| Action | Meaning |
|---:|---|
| `0` | Decrease the internal controller target |
| `1` | Hold the internal controller target |
| `2` | Increase the internal controller target |

The action does not directly apply torque. It changes the internal desired elbow position used by the validated low-level PD controller.

### Success condition

The environment succeeds when the actual elbow remains inside the allowed position tolerance for eight consecutive environment steps.

### Episode ending

- `terminated=True` means the task-defined success condition was reached.
- `truncated=True` means the fixed episode-length limit was reached first.


## 15.1 Install Gymnasium

Activate the project environment:

```bash
cd /mnt/c/StudentWork/Code/CSCN8020/Unitree_MuJoCo_G1_Primer_Workshop
source .venv/bin/activate
```

Install Gymnasium:

```bash
pip install "gymnasium==1.3.0"
```

Update the direct dependency file:

```bash
cat > requirements.txt <<'EOF'
gymnasium==1.3.0
mujoco==3.10.0
numpy==2.5.1
EOF
```

Refresh the exact lock file:

```bash
pip freeze > requirements-lock.txt
```

Verify:

```bash
python -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)"
```

### Why both files are updated

- `requirements.txt` remains the simple student installation file.
- `requirements-lock.txt` records the exact working environment, including indirect dependencies.


## 15.2 Create the Gymnasium package

Run:

```bash
mkdir -p src/g1_rl
touch src/g1_rl/__init__.py
touch src/g1_rl/g1_elbow_env.py
```

The `__init__.py` file makes `g1_rl` an importable Python package.


## 15.3 Create `src/g1_rl/g1_elbow_env.py`

Copy the complete working environment implementation below into:

```text
src/g1_rl/g1_elbow_env.py
```

```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


RenderMode = Literal["human"]


class G1ElbowTargetEnv(gym.Env[np.ndarray, int]):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    ACTION_DECREASE = 0
    ACTION_HOLD = 1
    ACTION_INCREASE = 2

    CONTROLLED_JOINT = "left_elbow_joint"
    CONTROLLED_ACTUATOR = "left_elbow"

    def __init__(
        self,
        scene_path: str | Path = (
            "assets/g1_fixed_base/"
            "scene_29dof_fixed_base.xml"
        ),
        render_mode: RenderMode | None = None,
        goal_angle: float | None = None,
        goal_range: tuple[float, float] = (-0.9, 1.5),
        action_increment: float = 0.08,
        frame_skip: int = 10,
        maximum_episode_steps: int = 150,
        success_tolerance: float = 0.04,
        required_success_steps: int = 8,
        controlled_kp: float = 20.0,
        controlled_kd: float = 2.0,
        hold_kp: float = 30.0,
        hold_kd: float = 3.0,
    ) -> None:
        super().__init__()

        self.scene_path = Path(scene_path).expanduser().resolve()

        if not self.scene_path.is_file():
            raise FileNotFoundError(
                f"G1 scene file was not found: {self.scene_path}"
            )

        if render_mode not in {None, "human"}:
            raise ValueError(
                "render_mode must be None or 'human'."
            )

        if not np.isfinite(action_increment) or action_increment <= 0:
            raise ValueError(
                "action_increment must be greater than zero."
            )

        if not np.isfinite(frame_skip) or frame_skip <= 0:
            raise ValueError(
                "frame_skip must be greater than zero."
            )

        if (
            not np.isfinite(maximum_episode_steps)
            or maximum_episode_steps <= 0
        ):
            raise ValueError(
                "maximum_episode_steps must be greater than zero."
            )

        if not np.isfinite(success_tolerance) or success_tolerance <= 0:
            raise ValueError(
                "success_tolerance must be greater than zero."
            )

        if (
            not np.isfinite(required_success_steps)
            or required_success_steps <= 0
        ):
            raise ValueError(
                "required_success_steps must be greater than zero."
            )

        gain_values = {
            "controlled_kp": controlled_kp,
            "controlled_kd": controlled_kd,
            "hold_kp": hold_kp,
            "hold_kd": hold_kd,
        }

        for gain_name, gain_value in gain_values.items():
            if not np.isfinite(gain_value) or gain_value < 0:
                raise ValueError(
                    f"{gain_name} must be greater than or equal to zero."
                )

        self.render_mode = render_mode
        self.fixed_goal_angle = goal_angle
        self.goal_range = goal_range
        self.action_increment = action_increment
        self.frame_skip = frame_skip
        self.maximum_episode_steps = maximum_episode_steps
        self.success_tolerance = success_tolerance
        self.required_success_steps = required_success_steps

        self.controlled_kp = controlled_kp
        self.controlled_kd = controlled_kd
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd

        self.model = mujoco.MjModel.from_xml_path(
            str(self.scene_path)
        )
        self.data = mujoco.MjData(self.model)

        self.joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            self.CONTROLLED_JOINT,
        )

        self.actuator_id = self._require_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            self.CONTROLLED_ACTUATOR,
        )

        self.qpos_index = int(
            self.model.jnt_qposadr[self.joint_id]
        )

        self.qvel_index = int(
            self.model.jnt_dofadr[self.joint_id]
        )

        joint_low, joint_high = self.model.jnt_range[
            self.joint_id
        ]

        self.joint_low = float(joint_low)
        self.joint_high = float(joint_high)

        self._validate_goal_range()

        self.joint_actuator_mapping = (
            self._build_joint_actuator_mapping()
        )

        self.action_space = spaces.Discrete(3)

        float_limit = np.finfo(np.float32).max

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.joint_low,
                    -float_limit,
                    self.joint_low,
                    self.joint_low - self.joint_high,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.joint_high,
                    float_limit,
                    self.joint_high,
                    self.joint_high - self.joint_low,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.goal_angle = 0.0
        self.controller_target = 0.0
        self.hold_targets = np.zeros(
            self.model.nq,
            dtype=np.float64,
        )

        self.episode_step = 0
        self.success_streak = 0
        self.viewer = None

    def _require_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        object_id = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise ValueError(
                f"MuJoCo object was not found: {name}"
            )

        return object_id

    def _validate_goal_range(self) -> None:
        if len(self.goal_range) != 2:
            raise ValueError(
                "goal_range must contain exactly two values."
            )

        goal_low, goal_high = self.goal_range

        if not np.isfinite([goal_low, goal_high]).all():
            raise ValueError(
                "goal_range bounds must be finite."
            )

        if goal_low >= goal_high:
            raise ValueError(
                "goal_range lower bound must be less than "
                "the upper bound."
            )

        if goal_low < self.joint_low:
            raise ValueError(
                "goal_range lower bound is outside the "
                "elbow joint range."
            )

        if goal_high > self.joint_high:
            raise ValueError(
                "goal_range upper bound is outside the "
                "elbow joint range."
            )

        if self.fixed_goal_angle is not None:
            if not np.isfinite(self.fixed_goal_angle):
                raise ValueError(
                    "goal_angle must be finite."
                )

            if not (
                self.joint_low
                <= self.fixed_goal_angle
                <= self.joint_high
            ):
                raise ValueError(
                    "goal_angle is outside the elbow joint range."
                )

    def _build_joint_actuator_mapping(
        self,
    ) -> list[tuple[int, int, int]]:
        mapping: list[tuple[int, int, int]] = []

        for actuator_id in range(self.model.nu):
            joint_id = int(
                self.model.actuator_trnid[actuator_id, 0]
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Actuator {actuator_id} is not attached "
                    "to a joint."
                )

            qpos_index = int(
                self.model.jnt_qposadr[joint_id]
            )

            qvel_index = int(
                self.model.jnt_dofadr[joint_id]
            )

            mapping.append(
                (
                    actuator_id,
                    qpos_index,
                    qvel_index,
                )
            )

        return mapping

    @staticmethod
    def _calculate_pd_torque(
        target_angle: float,
        current_angle: float,
        current_velocity: float,
        kp: float,
        kd: float,
    ) -> float:
        return (
            kp * (target_angle - current_angle)
            - kd * current_velocity
        )

    def _get_observation(self) -> np.ndarray:
        elbow_angle = float(
            self.data.qpos[self.qpos_index]
        )

        elbow_velocity = float(
            self.data.qvel[self.qvel_index]
        )

        angle_error = self.goal_angle - elbow_angle

        return np.array(
            [
                elbow_angle,
                elbow_velocity,
                self.goal_angle,
                angle_error,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        elbow_angle = float(
            self.data.qpos[self.qpos_index]
        )

        elbow_velocity = float(
            self.data.qvel[self.qvel_index]
        )

        angle_error = self.goal_angle - elbow_angle

        return {
            "elbow_angle": elbow_angle,
            "elbow_velocity": elbow_velocity,
            "goal_angle": self.goal_angle,
            "controller_target": self.controller_target,
            "angle_error": angle_error,
            "absolute_error": abs(angle_error),
            "success_streak": self.success_streak,
            "episode_step": self.episode_step,
        }

    def _select_goal(self) -> float:
        if self.fixed_goal_angle is not None:
            return float(self.fixed_goal_angle)

        low, high = self.goal_range

        return float(
            self.np_random.uniform(low, high)
        )

    def _apply_action(self, action: int) -> None:
        if action == self.ACTION_DECREASE:
            self.controller_target -= self.action_increment
        elif action == self.ACTION_HOLD:
            pass
        elif action == self.ACTION_INCREASE:
            self.controller_target += self.action_increment
        else:
            raise ValueError(
                f"Invalid action: {action}"
            )

        self.controller_target = float(
            np.clip(
                self.controller_target,
                self.joint_low,
                self.joint_high,
            )
        )

    def _apply_controller(self) -> None:
        for (
            actuator_id,
            qpos_index,
            qvel_index,
        ) in self.joint_actuator_mapping:
            current_position = float(
                self.data.qpos[qpos_index]
            )

            current_velocity = float(
                self.data.qvel[qvel_index]
            )

            if actuator_id == self.actuator_id:
                desired_position = self.controller_target
                kp = self.controlled_kp
                kd = self.controlled_kd
            else:
                desired_position = float(
                    self.hold_targets[qpos_index]
                )
                kp = self.hold_kp
                kd = self.hold_kd

            pd_torque = self._calculate_pd_torque(
                target_angle=desired_position,
                current_angle=current_position,
                current_velocity=current_velocity,
                kp=kp,
                kd=kd,
            )

            bias_torque = float(
                self.data.qfrc_bias[qvel_index]
            )

            commanded_torque = (
                pd_torque + bias_torque
            )

            control_low, control_high = (
                self.model.actuator_ctrlrange[
                    actuator_id
                ]
            )

            self.data.ctrl[actuator_id] = np.clip(
                commanded_torque,
                control_low,
                control_high,
            )

    def _calculate_reward(
        self,
        absolute_error: float,
        action: int,
    ) -> float:
        reward = -absolute_error

        if absolute_error <= self.success_tolerance:
            reward += 1.0

        if (
            absolute_error <= self.success_tolerance
            and action != self.ACTION_HOLD
        ):
            reward -= 0.05

        return float(reward)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.hold_targets = self.data.qpos.copy()

        if options is not None and "goal_angle" in options:
            selected_goal = float(
                options["goal_angle"]
            )

            if not np.isfinite(selected_goal) or not (
                self.joint_low
                <= selected_goal
                <= self.joint_high
            ):
                raise ValueError(
                    "The reset goal_angle is outside "
                    "the elbow joint range."
                )

            self.goal_angle = selected_goal
        else:
            self.goal_angle = self._select_goal()

        self.controller_target = float(
            self.data.qpos[self.qpos_index]
        )

        self.episode_step = 0
        self.success_streak = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is not in the action space."
            )

        self._apply_action(int(action))

        for _ in range(self.frame_skip):
            self._apply_controller()
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.episode_step += 1

        observation = self._get_observation()

        absolute_error = abs(
            float(observation[3])
        )

        if absolute_error <= self.success_tolerance:
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = (
            self.success_streak
            >= self.required_success_steps
        )

        truncated = (
            self.episode_step
            >= self.maximum_episode_steps
        )

        reward = self._calculate_reward(
            absolute_error=absolute_error,
            action=int(action),
        )

        if terminated:
            reward += 10.0

        info = self._get_info()
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

        if not self.viewer.is_running():
            return

        self.viewer.sync()

        time.sleep(
            1.0 / self.metadata["render_fps"]
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
```

## 15.4 Export the environment class

Place this in:

```text
src/g1_rl/__init__.py
```

```python
from .g1_elbow_env import G1ElbowTargetEnv

__all__ = ["G1ElbowTargetEnv"]
```


# Part 16 — Create the Working Validation Script

Create:

```bash
touch src/test_g1_elbow_env.py
```

Copy the complete working version below into the file.

This workshop intentionally includes only the successful rule-based policy. It does not include earlier experimental policies that failed the success test.

```python
from __future__ import annotations

import argparse
import time

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from g1_rl import G1ElbowTargetEnv


def choose_rule_based_action(
    observation: np.ndarray,
    controller_target: float,
    action_increment: float,
) -> int:
    """
    Move the internal controller target toward the final goal.

    Once the controller target reaches the goal, select HOLD and
    allow the low-level PD controller to settle the physical elbow.

    Observation:
        0: current elbow angle
        1: current elbow velocity
        2: goal angle
        3: angle error
    """

    goal_angle = float(observation[2])

    target_error = goal_angle - controller_target

    target_tolerance = action_increment / 2.0

    if target_error < -target_tolerance:
        return G1ElbowTargetEnv.ACTION_DECREASE

    if target_error > target_tolerance:
        return G1ElbowTargetEnv.ACTION_INCREASE

    return G1ElbowTargetEnv.ACTION_HOLD


def run_episode(
    env: gym.Env,
    seed: int,
) -> None:
    observation, info = env.reset(seed=seed)

    cumulative_reward = 0.0

    print("\n=== EPISODE START ===")
    print(f"Initial observation: {observation}")
    print(f"Goal angle: {info['goal_angle']:.4f} rad")

    while True:
        action = choose_rule_based_action(
            observation=observation,
            controller_target=float(
                info["controller_target"]
            ),
            action_increment=0.08,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += reward

        print(
            f"step={info['episode_step']:3d} | "
            f"action={action} | "
            f"angle={info['elbow_angle']:+.4f} | "
            f"velocity={info['elbow_velocity']:+.4f} | "
            f"controller_target="
            f"{info['controller_target']:+.4f} | "
            f"goal={info['goal_angle']:+.4f} | "
            f"error={info['angle_error']:+.4f} | "
            f"streak={info['success_streak']:2d} | "
            f"reward={reward:+.4f}"
        )

        if terminated or truncated:
            break

    print("\n=== EPISODE END ===")
    print(f"Terminated:        {terminated}")
    print(f"Truncated:         {truncated}")
    print(f"Success:           {info['is_success']}")
    print(f"Episode steps:     {info['episode_step']}")
    print(f"Final angle:       {info['elbow_angle']:.4f}")
    print(f"Goal angle:        {info['goal_angle']:.4f}")
    print(f"Final error:       {info['angle_error']:.4f}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the G1 elbow Gymnasium environment."
        )
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the MuJoCo viewer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    env = G1ElbowTargetEnv(
        render_mode="human" if args.render else None,
        goal_angle=-0.8,
    )

    try:
        print("Running Gymnasium environment checker.")

        check_env(
            env,
            skip_render_check=True,
        )

        print("Environment checker passed.")

        run_episode(
            env=env,
            seed=42,
        )

        if args.render:
            print("Close the MuJoCo viewer window to finish.")

            while (
                env.viewer is not None
                and env.viewer.is_running()
            ):
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

## 16.1 Where the rule-based policy is located

The rule-based policy is defined in:

```text
src/test_g1_elbow_env.py
```

inside:

```python
choose_rule_based_action(...)
```

It is not part of the Gymnasium environment itself.

The environment defines:

- Observation space
- Action space
- Physics
- Reward
- Success logic
- Termination and truncation

The validation script supplies a temporary hand-written policy used only to prove that the environment can be solved before DQN is introduced.

## 16.2 How the policy selects the next action

The policy compares the final goal with the internal controller target.

Example:

```text
goal angle        = -0.8000
controller target = -0.7200
action increment  =  0.0800
```

The target error is:

$$
-0.8000-(-0.7200)=-0.0800
$$

The decision tolerance is:

$$
0.0800/2=0.0400
$$

Because:

$$
-0.0800<-0.0400
$$

the policy selects action `0`, decreasing the controller target to `-0.8000`.

On the next step:

$$
-0.8000-(-0.8000)=0
$$

The policy therefore selects action `1` and holds the target steady while the low-level PD controller settles the physical elbow.

## 16.3 Run the first headless validation

Run from the project root:

```bash
PYTHONPATH=src python src/test_g1_elbow_env.py
```

`PYTHONPATH=src` temporarily adds the `src` directory to Python's import path so this import works:

```python
from g1_rl import G1ElbowTargetEnv
```

The required ending is:

```text
Terminated:        True
Truncated:         False
Success:           True
```

Validated reference result:

```text
Episode steps:     27
Final angle:       -0.7877
Goal angle:        -0.8000
Final error:       -0.0123
Cumulative reward: 10.5215
```


# Part 17 — Repeat the Headless Validation Five Times

Run:

```bash
for run in {1..5}; do
    echo
    echo "========================================"
    echo "Validation run $run"
    echo "========================================"

    PYTHONPATH=src python src/test_g1_elbow_env.py
done
```

## What to compare

For every run, confirm:

- `Environment checker passed.`
- `Terminated: True`
- `Truncated: False`
- `Success: True`
- The episode ends at step 27.
- The final angle is `-0.7877 rad`.
- The final error is `-0.0123 rad`.
- The cumulative reward is `10.5215`.

## Verified repeatability result

All five validated runs produced the same values:

```text
Episode steps:     27
Final angle:       -0.7877
Goal angle:        -0.8000
Final error:       -0.0123
Cumulative reward: 10.5215
```

This confirms deterministic behaviour for the fixed goal and fixed seed.


# Part 18 — Run the Optional Rendered Demonstration

After the five headless runs pass, run:

```bash
PYTHONPATH=src python src/test_g1_elbow_env.py --render
```

## Expected visual sequence

1. The fixed-base G1 appears.
2. Actions `0` move the internal controller target from `0.0` toward `-0.8 rad`.
3. At step 10, the controller target reaches `-0.8 rad`.
4. The policy switches to action `1`.
5. The elbow continues settling toward the goal.
6. The success streak increases from 1 through 8.
7. The episode terminates successfully at step 27.

## Verified rendered result

```text
Terminated:        True
Truncated:         False
Success:           True
Episode steps:     27
Final angle:       -0.7877
Goal angle:        -0.8000
Final error:       -0.0123
Cumulative reward: 10.5215
```

## WSLg viewer shutdown note

The console may print:

```text
Segmentation fault
```

after the successful episode results have already printed and the viewer closes.

In the validated setup, this is a viewer-cleanup issue under WSLg. It does not invalidate:

- MuJoCo physics
- Gymnasium API compliance
- Reward calculation
- Success logic
- Deterministic headless execution

All future training must remain headless. Rendering is optional and intended only for demonstrations.


## Part 18.1 — Interactive Gymnasium Viewer Demonstration

The earlier rendered validation starts the simulation immediately. In this demonstration, the viewer opens first and the simulation waits while the student prepares the camera.

The student can:

- Rotate, pan, and zoom the camera
- Position the G1 so the left elbow is visible
- Open the MuJoCo **Control** panel
- Return to the terminal and start the episode deliberately

The viewer's **Control** values are actuator commands in torque (`Nm`). They are not the Gymnasium action, the current elbow angle, or the controller target.

```text
Gymnasium action
        ↓
Internal controller target
        ↓
PD control plus bias compensation
        ↓
Actuator torque
        ↓
Physical elbow movement
```

## Create the interactive demonstration script

Create:

```bash
touch src/demo_g1_elbow_env.py
```

Open `src/demo_g1_elbow_env.py` and replace its contents with the complete code below.

```python
from __future__ import annotations

import argparse
import time

from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


def wait_for_student(
    env: G1ElbowTargetEnv,
) -> bool:
    """Keep the viewer open while the student prepares the camera."""

    print()
    print("=== CAMERA SETUP ===")
    print("The MuJoCo viewer is open.")
    print("Use the mouse to rotate, pan, and zoom.")
    print("Open the Control panel on the right if needed.")
    print("Return to the terminal and press Enter to begin.")
    print()

    input("Press Enter to start the Gymnasium demonstration...")

    return (
        env.viewer is not None
        and env.viewer.is_running()
    )


def run_demo(
    env: G1ElbowTargetEnv,
    seed: int,
    startup_countdown: int,
) -> None:
    observation, info = env.reset(seed=seed)

    env.render()

    if not wait_for_student(env):
        print("The viewer was closed before the demo started.")
        return

    print()
    print("Starting in:")

    for seconds_remaining in range(
        startup_countdown,
        0,
        -1,
    ):
        print(seconds_remaining)
        time.sleep(1.0)

    print("GO")
    print()

    cumulative_reward = 0.0

    while True:
        action = choose_rule_based_action(
            observation=observation,
            controller_target=float(
                info["controller_target"]
            ),
            action_increment=env.action_increment,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        cumulative_reward += reward

        print(
            f"step={info['episode_step']:3d} | "
            f"action={action} | "
            f"angle={info['elbow_angle']:+.4f} rad | "
            f"velocity={info['elbow_velocity']:+.4f} rad/s | "
            f"controller_target="
            f"{info['controller_target']:+.4f} rad | "
            f"goal={info['goal_angle']:+.4f} rad | "
            f"error={info['angle_error']:+.4f} rad | "
            f"streak={info['success_streak']:2d} | "
            f"reward={reward:+.4f}"
        )

        if (
            env.viewer is not None
            and not env.viewer.is_running()
        ):
            print("The viewer was closed before the episode finished.")
            break

        if terminated or truncated:
            print()
            print("=== DEMONSTRATION COMPLETE ===")
            print(f"Terminated:        {terminated}")
            print(f"Truncated:         {truncated}")
            print(f"Success:           {info['is_success']}")
            print(f"Episode steps:     {info['episode_step']}")
            print(f"Final angle:       {info['elbow_angle']:.4f}")
            print(f"Goal angle:        {info['goal_angle']:.4f}")
            print(f"Final error:       {info['angle_error']:.4f}")
            print(f"Cumulative reward: {cumulative_reward:.4f}")
            break

    print()
    input("Press Enter to close the viewer and end the demo...")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive visual demonstration of the "
            "Unitree G1 elbow Gymnasium environment."
        )
    )

    parser.add_argument(
        "--goal",
        type=float,
        default=-0.8,
        help="Elbow goal angle in radians.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Gymnasium reset seed.",
    )

    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to count down before the episode starts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.countdown < 0:
        raise ValueError(
            "--countdown must be zero or greater."
        )

    env = G1ElbowTargetEnv(
        render_mode="human",
        goal_angle=args.goal,
    )

    try:
        run_demo(
            env=env,
            seed=args.seed,
            startup_countdown=args.countdown,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

## Run the interactive viewer demonstration

From the project root, run:

```bash
PYTHONPATH=src python src/demo_g1_elbow_env.py
```

The viewer opens first and the terminal displays:

```text
=== CAMERA SETUP ===
The MuJoCo viewer is open.
Use the mouse to rotate, pan, and zoom.
Open the Control panel on the right if needed.
Return to the terminal and press Enter to begin.
```

Before pressing Enter:

1. Rotate the camera until the G1's left arm is visible.
2. Pan so the robot is centred.
3. Zoom so the elbow movement is easy to see.
4. Expand the viewer's **Control** panel.
5. Locate the `left_elbow` actuator entry.

Return to the terminal and press Enter. The script counts down and starts the approved Gymnasium episode.

Compare the live terminal values with the viewer:

- `action` is the discrete Gymnasium decision.
- `controller_target` is the desired elbow position.
- `angle` is the actual simulated elbow position.
- `left_elbow` in the **Control** panel is the actuator torque.

Expected final result:

```text
Terminated:        True
Truncated:         False
Success:           True
Episode steps:     27
Final angle:       -0.7877
Goal angle:        -0.8000
Final error:       -0.0123
Cumulative reward: 10.5215
```

After reviewing the final pose, press Enter in the terminal to close the viewer.

Optional examples:

```bash
PYTHONPATH=src python src/demo_g1_elbow_env.py   --goal 1.0
```

```bash
PYTHONPATH=src python src/demo_g1_elbow_env.py   --countdown 5
```

Under WSLg, the console may print `Segmentation fault` after the successful results have printed and the viewer closes. Treat this as the known viewer-shutdown issue when the episode completed normally.


## Part 19 — Gymnasium Milestone Status

| Milestone | Status | Verified Result |
|---|---|---|
| Gymnasium installation | Complete | Version 1.3.0 |
| Custom environment package | Complete | `src/g1_rl` |
| Four-value observation | Complete | Angle, velocity, goal, error |
| Three discrete actions | Complete | Decrease, hold, increase |
| Low-level controller integration | Complete | PD plus bias compensation |
| Reward calculation | Complete | Error shaping and bonuses |
| Success streak | Complete | Eight consecutive steps |
| Fixed episode length | Complete | Maximum 150 steps |
| Headless execution | Complete | Default mode |
| Optional rendering | Complete | MuJoCo passive viewer |
| Environment checker | Complete | Passed |
| Rule-based validation | Complete | Successful at step 27 |
| Five-run determinism | Complete | Identical results |
| Rendered demonstration | Complete | Successful; known shutdown fault |
| DQN implementation | Not started | Next phase only |


## Stop Point Before DQN

Do not begin the student-written DQN implementation until you have fully understood this Gymnasium milestone.

At this point, the environment has demonstrated:

- Correct Gymnasium API behaviour
- Correct observations and actions
- Correct low-level control integration
- Correct reward logic
- Correct success and time-limit handling
- Deterministic headless execution
- Successful optional rendering

This notebook intentionally stops here. No DQN code is included.
