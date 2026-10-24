"""把 LiteArm 与 LiteGrip 的 URDF 转成可仿真的 MJCF 场景。

真源：
- 机械臂 ``litearm-description/SE/litearm_description/urdf/litearm_description.urdf``
- 夹爪 ``litegrip-description/.../litegrip_description.urdf.xacro``

MuJoCo 不解析 ``package://``，也不解析 xacro，所以这里做最小转换：
- 保留所有 link 的惯性、视觉网格；
- body 名改成 ``link1..link7``，joint 名改成 ``j1..j7``，避免 link/joint 重名；
- 固定关节 ``ee_link`` 保留成无 joint 的 body，并挂 ``flange`` site；
- 夹爪两指保留 prismatic，关节名 ``finger_right/left``；
- 碰撞用网格包围盒生成的 box，视觉仍用原 STL。
"""

from __future__ import annotations

import hashlib
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rotations import matrix_to_quat_wxyz, rpy_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "sim_scene.xml"
DEFAULT_MESH_OUT = REPO_ROOT / "assets" / "meshes"

GRIPPER_STROKE_M = 0.042726
DEFAULT_TCP_OFFSET_M = 0.110
FINGER_COLLISION_LENGTH_M = 0.134


@dataclass
class Inertial:
    xyz: tuple[float, float, float] = (0, 0, 0)
    rpy: tuple[float, float, float] = (0, 0, 0)
    mass: float = 0.0
    inertia: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)


@dataclass
class Visual:
    mesh: str
    xyz: tuple[float, float, float] = (0, 0, 0)
    rpy: tuple[float, float, float] = (0, 0, 0)


@dataclass
class Link:
    name: str
    inertial: Inertial | None = None
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None = None
    upper: float | None = None


def _f(text: str | None, default: float = 0.0) -> float:
    return default if text is None else float(text)


def _vec(text: str | None, n: int) -> tuple[float, ...]:
    if text is None:
        return tuple([0.0] * n)
    values = [float(v) for v in text.split()]
    if len(values) != n:
        raise ValueError(f"expected {n} values, got {text!r}")
    return tuple(values)


def _parse_inertial(el: ET.Element | None) -> Inertial | None:
    if el is None:
        return None
    origin = el.find("origin")
    mass = el.find("mass")
    inertia = el.find("inertia")
    if mass is None or inertia is None:
        return None
    return Inertial(
        xyz=_vec(origin.get("xyz") if origin is not None else None, 3),
        rpy=_vec(origin.get("rpy") if origin is not None else None, 3),
        mass=float(mass.get("value", "0")),
        inertia=(
            _f(inertia.get("ixx")),
            _f(inertia.get("iyy")),
            _f(inertia.get("izz")),
            _f(inertia.get("ixy")),
            _f(inertia.get("ixz")),
            _f(inertia.get("iyz")),
        ),
    )


def _parse_visuals(link_el: ET.Element) -> list[Visual]:
    out: list[Visual] = []
    for vis in link_el.findall("visual"):
        origin = vis.find("origin")
        geom = vis.find("geometry")
        mesh = geom.find("mesh") if geom is not None else None
        if mesh is None or mesh.get("filename") is None:
            continue
        out.append(
            Visual(
                mesh=mesh.get("filename"),  # type: ignore[arg-type]
                xyz=_vec(origin.get("xyz") if origin is not None else None, 3),
                rpy=_vec(origin.get("rpy") if origin is not None else None, 3),
            )
        )
    return out


def parse_urdf(path: str | Path) -> tuple[dict[str, Link], list[Joint]]:
    root = ET.parse(path).getroot()
    links: dict[str, Link] = {}
    for el in root.findall("link"):
        name = el.get("name")
        if name is None:
            continue
        links[name] = Link(
            name=name,
            inertial=_parse_inertial(el.find("inertial")),
            visuals=_parse_visuals(el),
        )
    joints: list[Joint] = []
    for el in root.findall("joint"):
        parent = el.find("parent")
        child = el.find("child")
        if parent is None or child is None:
            continue
        origin = el.find("origin")
        axis = el.find("axis")
        limit = el.find("limit")
        joints.append(
            Joint(
                name=el.get("name", ""),
                type=el.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                xyz=_vec(origin.get("xyz") if origin is not None else None, 3),
                rpy=_vec(origin.get("rpy") if origin is not None else None, 3),
                axis=_vec(axis.get("xyz") if axis is not None else None, 3),
                lower=float(limit.get("lower"))
                if limit is not None and limit.get("lower")
                else None,
                upper=float(limit.get("upper"))
                if limit is not None and limit.get("upper")
                else None,
            )
        )
    return links, joints


def expand_gripper_xacro(path: str | Path, stroke_m: float = GRIPPER_STROKE_M) -> ET.Element:
    text = Path(path).read_text()
    text = text.replace('xmlns:xacro="http://www.ros.org/xacro"', "")
    text = text.replace("$(arg stroke)", str(stroke_m))
    text = text.replace("$(arg effort)", "10.0")
    text = text.replace("$(arg velocity)", "0.2")
    text = text.replace("$(arg joint_damping)", "0.05")
    text = text.replace("$(arg joint_friction)", "0.02")
    text = text.replace("$(arg hardware_plugin)", "mock_components/GenericSystem")
    # 删除 xacro:arg 与任何剩余 xacro 标签（ros2_control 段不是 URDF 树的一部分）
    import re

    text = re.sub(r"<xacro:arg[^>]*/>", "", text)
    text = re.sub(r"<xacro:[^>]*>.*?</xacro:[^>]*>", "", text, flags=re.S)
    text = re.sub(r"<xacro:[^>]*/>", "", text)
    return ET.fromstring(text)


def parse_gripper(path: str | Path, stroke_m: float = GRIPPER_STROKE_M):
    root = expand_gripper_xacro(path, stroke_m)
    links: dict[str, Link] = {}
    for el in root.findall("link"):
        name = el.get("name")
        if name is None:
            continue
        links[name] = Link(
            name=name,
            inertial=_parse_inertial(el.find("inertial")),
            visuals=_parse_visuals(el),
        )
    joints: list[Joint] = []
    for el in root.findall("joint"):
        parent = el.find("parent")
        child = el.find("child")
        if parent is None or child is None:
            continue
        origin = el.find("origin")
        axis = el.find("axis")
        limit = el.find("limit")
        joints.append(
            Joint(
                name=el.get("name", ""),
                type=el.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                xyz=_vec(origin.get("xyz") if origin is not None else None, 3),
                rpy=_vec(origin.get("rpy") if origin is not None else None, 3),
                axis=_vec(axis.get("xyz") if axis is not None else None, 3),
                lower=float(limit.get("lower"))
                if limit is not None and limit.get("lower")
                else None,
                upper=float(limit.get("upper"))
                if limit is not None and limit.get("upper")
                else None,
            )
        )
    return links, joints


def stl_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        # ASCII fallback
        text = data.decode("ascii", errors="ignore")
        vertices = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                vertices.append([float(v) for v in parts[1:]])
        arr = np.asarray(vertices, dtype=float)
        if arr.size == 0:
            raise ValueError(f"cannot parse STL: {path}")
        return arr.min(axis=0), arr.max(axis=0)
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for i in range(count):
        base = 84 + 50 * i + 12
        for j in range(3):
            v = struct.unpack_from("<3f", data, base + 12 * j)
            mins = np.minimum(mins, v)
            maxs = np.maximum(maxs, v)
    return mins, maxs


def _fmt(values, digits: int = 9) -> str:
    if isinstance(values, (int, float)):
        return f"{float(values):.{digits}g}"
    return " ".join(f"{float(v):.{digits}g}" for v in values)


def _quat_from_rpy(rpy) -> tuple[float, float, float, float]:
    return matrix_to_quat_wxyz(rpy_to_matrix(rpy))


def _lookat_quat(position, target) -> tuple[float, float, float, float]:
    """MuJoCo 相机沿 -Z 看；相机 +Z 指向目标的反方向。"""
    pos = np.asarray(position, dtype=float)
    tgt = np.asarray(target, dtype=float)
    z = pos - tgt
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return matrix_to_quat_wxyz(R)


def _stl_face_count(path: Path) -> int:
    with path.open("rb") as fh:
        fh.seek(80)
        raw = fh.read(4)
    return struct.unpack("<I", raw)[0] if len(raw) == 4 else 0


def _maybe_decimate(path: Path, *, max_faces: int = 150_000, threshold: int = 190_000) -> None:
    """MuJoCo 的 STL 解码器上限是 20 万面；超过就简化。"""
    if _stl_face_count(path) <= threshold:
        return
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确错误
        raise RuntimeError(
            f"{path.name} has more than {threshold} faces; install "
            "trimesh and fast-simplification to decimate it"
        ) from exc
    mesh = trimesh.load(path, force="mesh")
    simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
    simplified.export(path, file_type="stl")


def _copy_meshes(arm_mesh_dir: Path, gripper_mesh_dir: Path, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for src in sorted(arm_mesh_dir.glob("*.STL")):
        name = f"arm_{src.stem}"
        dst = out_dir / f"{name}.STL"
        shutil.copy2(src, dst)
        _maybe_decimate(dst)
        mapping[f"package://litearm_description/meshes/{src.name}"] = name
    for src in sorted(gripper_mesh_dir.glob("*.STL")):
        name = f"grip_{src.stem}"
        dst = out_dir / f"{name}.STL"
        shutil.copy2(src, dst)
        _maybe_decimate(dst)
        mapping[f"package://litegrip_description/meshes/{src.name}"] = name
    return mapping


def _inertial_xml(inertial: Inertial | None) -> str:
    if inertial is None or inertial.mass <= 0:
        return ""
    ixx, iyy, izz, ixy, ixz, iyz = inertial.inertia
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
    # 把惯性从 URDF 的惯性坐标系旋转到 link 坐标系；MuJoCo 不允许
    # fullinertia 与 inertial quat 同时出现。
    R = rpy_to_matrix(inertial.rpy)
    inertia = R @ inertia @ R.T
    vals = np.linalg.eigvalsh(inertia)
    if np.min(vals) <= 0:
        return ""
    return (
        f'<inertial pos="{_fmt(inertial.xyz)}" '
        f'mass="{_fmt(inertial.mass)}" '
        f'fullinertia="{_fmt(inertia[0, 0])} {_fmt(inertia[1, 1])} {_fmt(inertia[2, 2])} '
        f'{_fmt(inertia[0, 1])} {_fmt(inertia[0, 2])} {_fmt(inertia[1, 2])}"/>'
    )


def _mesh_geoms(link: Link, mesh_map: dict[str, str], material: str, group: int = 0) -> str:
    parts = []
    for vis in link.visuals:
        mesh = mesh_map.get(vis.mesh)
        if mesh is None:
            continue
        quat = _quat_from_rpy(vis.rpy)
        parts.append(
            f'<geom type="mesh" mesh="{mesh}" pos="{_fmt(vis.xyz)}" quat="{_fmt(quat)}" '
            f'material="{material}" contype="0" conaffinity="0" group="{group}"/>'
        )
    return "".join(parts)


def _collision_box(link: Link, mesh_dir: Path, mesh_map: dict[str, str]) -> str:
    """用网格包围盒生成不可见碰撞盒。"""
    if not link.visuals:
        return ""
    vis = link.visuals[0]
    mesh_name = mesh_map.get(vis.mesh)
    if mesh_name is None:
        return ""
    path = mesh_dir / f"{mesh_name}.STL"
    if not path.is_file():
        return ""
    lo, hi = stl_bbox(path)
    size = np.maximum((hi - lo) / 2.0, 0.004)
    center = (hi + lo) / 2.0
    quat = _quat_from_rpy(vis.rpy)
    return (
        f'<geom name="{link.name}_col" type="box" pos="{_fmt(center)}" quat="{_fmt(quat)}" '
        f'size="{_fmt(size)}" rgba="0 0 0 0" group="3" '
        f'contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>'
    )


def _emit_body(
    link_name: str,
    links: dict[str, Link],
    children: dict[str, list[Joint]],
    mesh_map: dict[str, str],
    mesh_dir: Path,
    body_names: dict[str, str],
    joint_names: dict[str, str],
    indent: int,
    *,
    joint: Joint | None = None,
    extra_children: dict[str, str] | None = None,
) -> str:
    """输出 link_name 的 body；joint 是该 body 相对父 body 的关节。"""
    pad = "  " * indent
    link = links[link_name]
    body = body_names.get(link_name, link_name)
    if joint is None:
        out = [f'{pad}<body name="{body}">']
    else:
        quat = _quat_from_rpy(joint.rpy)
        out = [f'{pad}<body name="{body}" pos="{_fmt(joint.xyz)}" quat="{_fmt(quat)}">']
        if joint.type in ("revolute", "continuous"):
            lo = joint.lower if joint.lower is not None else -math.pi
            hi = joint.upper if joint.upper is not None else math.pi
            out.append(
                f'{pad}  <joint name="{joint_names[joint.name]}" type="hinge" '
                f'axis="{_fmt(joint.axis)}" range="{_fmt(lo)} {_fmt(hi)}" damping="0.1"/>'
            )
        elif joint.type == "prismatic":
            lo = joint.lower if joint.lower is not None else 0.0
            hi = joint.upper if joint.upper is not None else 0.0
            out.append(
                f'{pad}  <joint name="{joint_names[joint.name]}" type="slide" '
                f'axis="{_fmt(joint.axis)}" range="{_fmt(lo)} {_fmt(hi)}" damping="0.5"/>'
            )
        elif joint.type != "fixed":
            raise ValueError(f"unsupported joint type {joint.type}")
    out.append(_inertial_xml(link.inertial))
    out.append(_mesh_geoms(link, mesh_map, "mat_link"))
    out.append(_collision_box(link, mesh_dir, mesh_map))
    if extra_children and link_name in extra_children:
        out.append(extra_children[link_name])
    for child_joint in children.get(link_name, []):
        out.append(
            _emit_body(
                child_joint.child,
                links,
                children,
                mesh_map,
                mesh_dir,
                body_names,
                joint_names,
                indent + 1,
                joint=child_joint,
                extra_children=extra_children,
            )
        )
    out.append(f"{pad}</body>")
    return "\n".join(part for part in out if part)


def _children_map(joints: list[Joint]) -> dict[str, list[Joint]]:
    out: dict[str, list[Joint]] = {}
    for joint in joints:
        out.setdefault(joint.parent, []).append(joint)
    return out


def _joint_limits(joints: list[Joint], name: str) -> tuple[float, float]:
    for j in joints:
        if j.name == name:
            return (j.lower or 0.0, j.upper or 0.0)
    raise KeyError(name)


def build_model(
    output: str | Path | None = None,
    *,
    arm_urdf: str | Path,
    gripper_xacro: str | Path,
    arm_meshes: str | Path | None = None,
    gripper_meshes: str | Path | None = None,
    mesh_out: str | Path | None = None,
    tcp_offset_m: float = DEFAULT_TCP_OFFSET_M,
    stroke_m: float = GRIPPER_STROKE_M,
    block_size_m: float = 0.03,
    start_x: tuple[float, float] = (0.09, 0.15),
    start_y: tuple[float, float] = (-0.20, -0.15),
    target_center: tuple[float, float] = (0.20, -0.10),
    target_radius_m: float = 0.04,
    timestep: float = 0.002,
) -> Path:
    arm_urdf = Path(arm_urdf)
    gripper_xacro = Path(gripper_xacro)
    arm_meshes = Path(arm_meshes) if arm_meshes else arm_urdf.parent.parent / "meshes"
    gripper_meshes = (
        Path(gripper_meshes) if gripper_meshes else gripper_xacro.parent.parent / "meshes"
    )
    out = Path(output or DEFAULT_OUTPUT)
    mesh_dir = Path(mesh_out or DEFAULT_MESH_OUT)

    mesh_map = _copy_meshes(arm_meshes, gripper_meshes, mesh_dir)
    arm_links, arm_joints = parse_urdf(arm_urdf)
    grip_links, grip_joints = parse_gripper(gripper_xacro, stroke_m)

    # body/joint 重命名，避免 link 与 joint 同名。
    body_names = {"base_link": "base_link"}
    joint_names = {}
    for i in range(1, 8):
        body_names[f"joint{i}"] = f"link{i}"
        joint_names[f"joint{i}"] = f"j{i}"
    body_names["ee_link"] = "ee_link"

    arm_children = _children_map(arm_joints)

    # 夹爪：base_footprint/base_link 都映射到 gripper_base，挂在 ee_link 下。
    grip_base = grip_links["base_link"]
    finger1 = grip_links["gripper_slider_link1"]
    finger2 = grip_links["gripper_slider_link2"]
    j_right = next(j for j in grip_joints if j.name == "gripper_slide_joint_right")
    j_left = next(j for j in grip_joints if j.name == "gripper_slide_joint_left")

    def gripper_geom(link: Link) -> str:
        return _mesh_geoms(link, mesh_map, "mat_gripper")

    finger_col_len = max(FINGER_COLLISION_LENGTH_M - j_right.xyz[2], 0.02)
    gripper_xml = f"""
    <body name="gripper_base">
      {_inertial_xml(grip_base.inertial)}
      {gripper_geom(grip_base)}
      <site name="tcp" pos="0 0 {_fmt(tcp_offset_m)}" size="0.006" rgba="1 0.2 0.2 1"/>
      <body name="finger_right" pos="{_fmt(j_right.xyz)}" quat="{_fmt(_quat_from_rpy(j_right.rpy))}">
        <joint name="finger_right" type="slide" axis="{_fmt(j_right.axis)}" range="0 {_fmt(stroke_m)}" damping="0.5"/>
        {gripper_geom(finger1)}
        <geom name="finger_right_col" type="box" pos="0 0 {_fmt(finger_col_len / 2.0)}"
              size="0.0235 0.030 {_fmt(finger_col_len / 2.0)}" rgba="0 0 0 0" group="3"
              contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
      </body>
      <body name="finger_left" pos="{_fmt(j_left.xyz)}" quat="{_fmt(_quat_from_rpy(j_left.rpy))}">
        <joint name="finger_left" type="slide" axis="{_fmt(j_left.axis)}" range="0 {_fmt(stroke_m)}" damping="0.5"/>
        {gripper_geom(finger2)}
        <geom name="finger_left_col" type="box" pos="0 0 {_fmt(finger_col_len / 2.0)}"
              size="0.0235 0.030 {_fmt(finger_col_len / 2.0)}" rgba="0 0 0 0" group="3"
              contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
      </body>
      <camera name="cam_wrist" pos="0.045 0 0.05" xyaxes="1 0 0 0 -1 0" fovy="70"/>
    </body>
"""

    arm_xml = _emit_body(
        "base_link",
        arm_links,
        arm_children,
        mesh_map,
        arm_meshes,
        body_names,
        joint_names,
        2,
        extra_children={
            "ee_link": '<site name="flange" pos="0 0 0" size="0.004" rgba="0.2 0.4 1 1"/>\n'
            + gripper_xml
        },
    )

    # 机械臂执行器：kp/kd 取固件默认量级；ctrlrange 用软限位。
    actuator_lines = []
    for i in range(1, 8):
        lo, hi = _joint_limits(arm_joints, f"joint{i}")
        kp = 400 if i <= 2 else (300 if i <= 4 else 80)
        kv = 5 if i <= 4 else 2.5
        force = 78 if i <= 2 else (21 if i <= 4 else 10)
        actuator_lines.append(
            f'<position name="j{i}" joint="j{i}" kp="{kp}" kv="{kv}" '
            f'ctrlrange="{_fmt(lo)} {_fmt(hi)}" forcerange="{-force} {force}"/>'
        )
    actuator_lines.append(
        f'<position name="finger_right_act" joint="finger_right" kp="800" kv="15" '
        f'ctrlrange="0 {_fmt(stroke_m)}" forcerange="-40 40"/>'
    )
    actuator_lines.append(
        f'<position name="finger_left_act" joint="finger_left" kp="800" kv="15" '
        f'ctrlrange="0 {_fmt(stroke_m)}" forcerange="-40 40"/>'
    )

    excludes = [
        ("base_link", "link1"),
        ("link1", "link2"),
        ("link2", "link3"),
        ("link3", "link4"),
        ("link4", "link5"),
        ("link5", "link6"),
        ("link6", "link7"),
        ("link7", "ee_link"),
        ("ee_link", "gripper_base"),
        ("gripper_base", "finger_right"),
        ("gripper_base", "finger_left"),
        ("finger_right", "finger_left"),
    ]
    exclude_xml = "".join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in excludes)

    arm_hash = hashlib.sha256(arm_urdf.read_bytes()).hexdigest()[:16]
    grip_hash = hashlib.sha256(gripper_xacro.read_bytes()).hexdigest()[:16]

    xml = f"""<mujoco model="litearm_gpt6astra">
  <!-- 由 scripts/build_sim_model.py 生成，不要手改。
       arm_urdf_sha256[:16]={arm_hash}
       gripper_xacro_sha256[:16]={grip_hash}
       tcp_offset_m={tcp_offset_m} stroke_m={stroke_m} -->
  <compiler angle="radian" eulerseq="XYZ" autolimits="true" meshdir="meshes"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint limited="true" damping="0.1" armature="0.005"/>
    <geom contype="0" conaffinity="0" condim="3" solimp="0.99 0.99 0.01" solref="0.02 1"/>
  </default>
  <asset>
    <material name="mat_link" rgba="0.55 0.58 0.62 1"/>
    <material name="mat_gripper" rgba="0.92 0.92 0.92 1"/>
    <material name="mat_table" rgba="0.42 0.42 0.42 1"/>
    <material name="mat_block" rgba="0.9 0.05 0.05 1"/>
    <material name="mat_target" rgba="0.1 0.8 0.1 0.5"/>
{"".join(f'    <mesh name="{v}" file="{v}.STL"/>' for v in sorted(set(mesh_map.values())))}
  </asset>
  <worldbody>
    <light name="light1" pos="1 1 2" directional="true" intensity="0.45"/>
    <light name="light2" pos="-1 -1 2" directional="true" intensity="0.35"/>
    <geom name="floor" type="plane" size="2 2 0.01" pos="0 0 -0.02" material="mat_table"/>
{arm_xml}
    <body name="table" pos="0 0 -0.02">
      <geom name="table_top" type="box" size="0.55 0.40 0.02" material="mat_table"
            contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
    </body>
    <body name="block" pos="{_fmt((start_x[0] + start_x[1]) / 2)} {_fmt((start_y[0] + start_y[1]) / 2)} {_fmt(block_size_m / 2)}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{_fmt(block_size_m / 2)} {_fmt(block_size_m / 2)} {_fmt(block_size_m / 2)}"
            mass="0.05" material="mat_block" contype="1" conaffinity="1"
            friction="1.0 0.005 0.0001"/>
    </body>
    <geom name="target_zone" type="cylinder" pos="{_fmt(target_center[0])} {_fmt(target_center[1])} 0.001"
          size="{_fmt(target_radius_m)} 0.001" material="mat_target" contype="0" conaffinity="0"/>
    <camera name="cam_head" pos="0 -0.85 0.75" quat="{_fmt(_lookat_quat((0.0, -0.85, 0.75), (0.15, -0.10, 0.02)))}" fovy="50"/>
  </worldbody>
  <contact>
    {exclude_xml}
  </contact>
  <equality>
    <weld name="grasp_weld" body1="gripper_base" body2="block" active="false" solref="0.02 1"/>
  </equality>
  <actuator>
    {chr(10).join("    " + line for line in actuator_lines)}
  </actuator>
</mujoco>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out
