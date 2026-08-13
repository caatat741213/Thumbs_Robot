"""
G1 Fixed-Base Model Generator
Stage: Phase 0 (Model Setup)
Primary Function:
- Converts the official 3-finger hand Unitree G1 robot model (g1_29dof_with_hand.xml) into a fixed-base robot model.
- Removes the free joint from the pelvis body.
- Sets a practical base height and orientation, and updates meshing/include paths for scene XML.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths to the source MuJoCo model and scene files from unitree_mujoco submodule
SOURCE_MODEL = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/g1_29dof_with_hand.xml"
)

SOURCE_SCENE = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/scene_29dof.xml"
)

# Output directory and target files for the fixed-base G1 model
OUTPUT_DIRECTORY = Path("assets/g1_fixed_base")

OUTPUT_MODEL = OUTPUT_DIRECTORY / "g1_29dof_fixed_base.xml"
OUTPUT_SCENE = OUTPUT_DIRECTORY / "scene_29dof_fixed_base.xml"

SOURCE_MESH_DIRECTORY = Path(
    "external/unitree_mujoco/"
    "unitree_robots/g1/meshes"
)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """
    Recursively indents XML elements for a clean, human-readable format.
    """
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
    """
    Loads source G1 model, updates mesh directories, removes pelvis free joint, 
    and saves the modified XML as the fixed-base model.
    """
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Source G1 model was not found: {SOURCE_MODEL}"
        )

    # Ensure output directory exists
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Parse XML model
    tree = ET.parse(SOURCE_MODEL)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError(
            "The source model does not contain a compiler element."
        )

    # Redirect mesh search path locally
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

    # Search for free joint element inside pelvis (floating base joint)
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

    # Remove free joint to lock the pelvis in space
    pelvis.remove(free_joint)

    # Place the fixed pelvis at a practical height (80cm) above the floor.
    pelvis.set("pos", "0 0 0.80")
    pelvis.set("quat", "1 0 0 0")

    # Re-indent and write XML
    indent_xml(root)
    tree.write(
        OUTPUT_MODEL,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Removed free joint: {removed_joint_name}")
    print(f"Created fixed-base model: {OUTPUT_MODEL}")


def copy_meshes() -> None:
    """
    Copies 3D mesh files from the unitree submodule to local assets folder.
    """
    destination = OUTPUT_DIRECTORY / "meshes"

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(
        SOURCE_MESH_DIRECTORY,
        destination,
    )

    print(f"Copied meshes to: {destination}")


def create_fixed_base_scene() -> None:
    """
    Generates a scene XML that includes the newly created fixed-base G1 model.
    """
    if not SOURCE_SCENE.is_file():
        raise FileNotFoundError(
            f"Source G1 scene was not found: {SOURCE_SCENE}"
        )

    # Parse XML scene
    tree = ET.parse(SOURCE_SCENE)
    root = tree.getroot()

    include = root.find("include")
    if include is None:
        raise RuntimeError(
            "The source scene does not contain an include element."
        )

    # Include the newly generated fixed base XML model file
    include.set(
        "file",
        OUTPUT_MODEL.name,
    )

    # Remove baseline keyframe tags to prevent conflicts during reset
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    # Re-indent and write XML scene
    indent_xml(root)
    tree.write(
        OUTPUT_SCENE,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Created fixed-base scene: {OUTPUT_SCENE}")


def main() -> None:
    # Run the asset generation pipeline
    create_fixed_base_model()
    copy_meshes()
    create_fixed_base_scene()

    print("\nFixed-base G1 assets created successfully.")


if __name__ == "__main__":
    main()