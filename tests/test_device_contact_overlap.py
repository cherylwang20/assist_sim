"""No device geom may start inside a human collider: every MSK x device, ``stand`` pose.

MuJoCo only auto-excludes parent/child bodies.  On the freejoint lineages the torso
sits five spine joints above the pelvis and the feet and shanks carry extra
colliders, so a device part that overlaps the body by design (a belt on the pelvis,
a boot shank around the tibia) collides unless its config declares a
``contact.excludes`` pair.  An unexcluded overlap compiles and steps without error,
but the solver pushes the two apart at tens of kN on the first step -- the smoke
tests cannot see it.

The tolerance admits sub-millimetre-to-millimetre seating contact (Tutorial_L1's
actuator rests ~1 mm into the heel capsule) and nothing like the 16-196 mm overlaps
the excludes were added for.
"""

from __future__ import annotations

import mujoco as mj
import pytest

from assist_sim import load_combined

from .conftest import needs_myo_sim
from .test_smoke_combinations import EXPECTED

MAX_PENETRATION_M = 0.002


def _device_human_overlaps(model: mj.MjModel, data: mj.MjData, prefix: str) -> dict[tuple[str, str], float]:
    """Deepest penetration (m) per (device body, human body) pair in contact."""
    device_geoms = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith(prefix)}
    overlaps: dict[tuple[str, str], float] = {}
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom[0]), int(contact.geom[1])
        if (g1 in device_geoms) == (g2 in device_geoms):
            continue
        dev, hum = (g1, g2) if g1 in device_geoms else (g2, g1)
        key = (model.body(model.geom_bodyid[dev]).name, model.body(model.geom_bodyid[hum]).name)
        overlaps[key] = max(overlaps.get(key, 0.0), -contact.dist)
    return overlaps


@needs_myo_sim
@pytest.mark.parametrize("keys", list(EXPECTED), ids=lambda x: str(x))
def test_no_device_human_overlap_at_stand(keys):
    msk_key, device_key = keys
    model, data = load_combined(msk_key, device_key)
    mj.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mj.mj_forward(model, data)

    overlaps = _device_human_overlaps(model, data, device_key + "_")
    too_deep = {pair: round(depth * 1000, 1) for pair, depth in overlaps.items() if depth > MAX_PENETRATION_M}
    assert not too_deep, (
        f"{msk_key} x {device_key}: device geoms start inside human colliders (mm): {too_deep}. "
        "Add the pairs to the device config's contact.excludes for this MSK."
    )


# The two halves of the Dephy boot: shank side on tibia, foot side on calcn.  tb_toe
# (on toes) is left out because it never reaches the shank side.
DEPHY_SHANK_PARTS = ("exo_1", "tb_shank")
DEPHY_FOOT_PARTS = ("exo_2", "tb_mid")


@needs_myo_sim
@pytest.mark.parametrize("msk_key", sorted({msk for msk, device in EXPECTED if device == "DephyExoBoot_L1"}))
def test_dephy_boot_halves_do_not_collide(msk_key):
    """The boot's shank and foot halves overlap by design and must never generate contacts.

    On the subtalar lineages calcn is not welded to talus, so MuJoCo's parent filter
    does not cover tibia-side vs calcn-side bodies.  Sweeps the ankle through the
    boot's range [rad] because the overlap depth changes with ankle angle.
    """
    device_key = "DephyExoBoot_L1"
    model, data = load_combined(msk_key, device_key)
    halves = {}
    for side in "rl":
        for name in DEPHY_SHANK_PARTS:
            halves[model.body(f"{device_key}_{name}_{side}").id] = ("shank", side)
        for name in DEPHY_FOOT_PARTS:
            halves[model.body(f"{device_key}_{name}_{side}").id] = ("foot", side)

    colliding = set()
    for side in "rl":
        joint = model.joint(f"ankle_angle_{side}")
        low, high = model.jnt_range[joint.id]
        for angle in (low, 0.0, high):
            mj.mj_resetDataKeyframe(model, data, model.key("stand").id)
            data.qpos[model.jnt_qposadr[joint.id]] = angle
            mj.mj_forward(model, data)
            for i in range(data.ncon):
                b1, b2 = (int(model.geom_bodyid[g]) for g in data.contact[i].geom)
                h1, h2 = halves.get(b1), halves.get(b2)
                # Same leg, opposite halves.  Left-right contact is a real collision.
                if h1 and h2 and h1[1] == h2[1] and h1[0] != h2[0]:
                    colliding.add(tuple(sorted((model.body(b1).name, model.body(b2).name))))
    assert not colliding, f"{msk_key}: Dephy boot halves collide: {sorted(colliding)}"
