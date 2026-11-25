"""Personalised CartPole variant, derived deterministically from a student ID.

Every student gets different physics, a different faulty actuator and a different
target position for the reward-design question. This means results cannot be copied
between students, and every number in a report can be checked by re-running the
submitted model on the student's own variant.

    python -m rl_assignment.variant --student-id 123456M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass

SALT = "CCE5502-2026"


def normalise_id(student_id: str) -> str:
    """'123456 (M)' -> '123456M'; the VLE form '123456M-abcd0001' -> '123456M' (the username after
    the dash is dropped). Students without a student number use their username, e.g. 'abcde01'.
    Raises if nothing usable is left."""
    student_id = re.sub(r"\s*-\s*[A-Za-z]+\d+\s*$", "", student_id.strip())
    sid = re.sub(r"[^0-9A-Za-z]", "", student_id).upper()
    if len(sid) < 4:
        raise ValueError(f"student id {student_id!r} looks wrong — use your UM student number")
    return sid


def _uniform(digest: bytes, k: int, lo: float, hi: float) -> float:
    """k-th pseudo-random number in [lo, hi] taken from the hash digest."""
    x = int.from_bytes(digest[2 * k: 2 * k + 2], "big") / 65535.0
    return lo + (hi - lo) * x


@dataclass(frozen=True)
class Variant:
    student_id: str
    fingerprint: str        # short hash written into every run, used by the verification script
    pole_length: float      # half-length of the pole [m]      (default CartPole: 0.5)
    pole_mass: float        # [kg]                              (default: 0.1)
    cart_mass: float        # [kg]                              (default: 1.0)
    force_mag: float        # actuator force [N]                (default: 10.0)
    obs_noise_std: float    # std of Gaussian noise added to every observation
    fault_delay: int        # faulty robot: the motor executes each command this many control steps late (20 ms each)
    fault_side: str         # faulty robot: "left" or "right" push direction is also weakened ...
    fault_strength: float   # ... to this fraction of the nominal force
    target_x: float         # Part D: where the cart should be kept [m]

    def physics(self) -> dict:
        return dict(length=self.pole_length, masspole=self.pole_mass, masscart=self.cart_mass, force_mag=self.force_mag)

    def card(self) -> str:
        return "\n".join([
            f"CCE5502 RL assignment 2026 — variant for {self.student_id}  (fingerprint {self.fingerprint})",
            f"  pole half-length   {self.pole_length:.3f} m    (standard 0.500)",
            f"  pole mass          {self.pole_mass:.3f} kg   (standard 0.100)",
            f"  cart mass          {self.cart_mass:.3f} kg   (standard 1.000)",
            f"  actuator force     {self.force_mag:.2f} N    (standard 10.00)",
            f"  observation noise  σ = {self.obs_noise_std:.4f}",
            f"  faulty actuator    commands execute {self.fault_delay} steps ({20 * self.fault_delay} ms) late, and",
            f"                     pushing {self.fault_side.upper()} delivers only {100 * self.fault_strength:.0f}% of the force",
            f"  Part D target      x* = {self.target_x:+.2f} m",
        ])


def make_variant(student_id: str) -> Variant:
    sid = normalise_id(student_id)
    d = hashlib.sha256(f"{SALT}:{sid}".encode()).digest()
    return Variant(
        student_id=sid,
        fingerprint=hashlib.sha256(f"{SALT}:{sid}:fp".encode()).hexdigest()[:10],
        pole_length=round(_uniform(d, 0, 0.35, 0.80), 3),
        pole_mass=round(_uniform(d, 1, 0.05, 0.25), 3),
        cart_mass=round(_uniform(d, 2, 0.70, 1.60), 3),
        force_mag=round(_uniform(d, 3, 8.0, 13.0), 2),
        obs_noise_std=round(_uniform(d, 4, 0.0, 0.01), 4),
        fault_delay=3 + d[20] % 2,
        fault_side="left" if d[10] % 2 == 0 else "right",
        fault_strength=round(_uniform(d, 6, 0.60, 0.85), 2),
        target_x=round((1 if d[14] % 2 else -1) * _uniform(d, 8, 0.6, 1.4), 2),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--json", action="store_true", help="print as JSON instead of a readable card")
    a = ap.parse_args()
    v = make_variant(a.student_id)
    print(json.dumps(asdict(v), indent=2) if a.json else v.card())
