"""Typed, immutable per-robot profiles for verified algorithm stages."""

from core_retarget.robots.profiles.adam import ADAM_DMR_PROFILE
from core_retarget.robots.profiles.apollo import APOLLO_DMR_PROFILE
from core_retarget.robots.profiles.ara import (
    ARA_PROFILES,
    ROBOT_NEUTRAL_ARA_PROFILE,
    AraProfile,
    get_ara_profile,
)
from core_retarget.robots.profiles.fpa import FPA_PROFILES, FpaProfile, get_fpa_profile
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE
from core_retarget.robots.profiles.h1 import H1_DMR_PROFILE
from core_retarget.robots.profiles.h2 import H2_DMR_PROFILE
from core_retarget.robots.profiles.initial_collision import (
    ADAM_INITIAL_COLLISION_PROFILE,
    APOLLO_INITIAL_COLLISION_PROFILE,
    G1_INITIAL_COLLISION_PROFILE,
    H1_INITIAL_COLLISION_PROFILE,
    H2_INITIAL_COLLISION_PROFILE,
    INITIAL_COLLISION_PROFILES,
    K1_INITIAL_COLLISION_PROFILE,
    N1_INITIAL_COLLISION_PROFILE,
    OLI_INITIAL_COLLISION_PROFILE,
    PM01_INITIAL_COLLISION_PROFILE,
    R1_INITIAL_COLLISION_PROFILE,
    T1_INITIAL_COLLISION_PROFILE,
    get_initial_collision_profile,
)
from core_retarget.robots.profiles.k1 import K1_DMR_PROFILE
from core_retarget.robots.profiles.n1 import N1_DMR_PROFILE
from core_retarget.robots.profiles.oli import OLI_DMR_PROFILE
from core_retarget.robots.profiles.pm01 import PM01_DMR_PROFILE
from core_retarget.robots.profiles.r1 import R1_DMR_PROFILE
from core_retarget.robots.profiles.registry import DMR_PROFILES, get_dmr_profile
from core_retarget.robots.profiles.schema import (
    DmrProfile,
    IkSolverProfile,
    InitialCollisionProfile,
)
from core_retarget.robots.profiles.t1 import T1_DMR_PROFILE

__all__ = [
    "ADAM_DMR_PROFILE",
    "ADAM_INITIAL_COLLISION_PROFILE",
    "APOLLO_DMR_PROFILE",
    "APOLLO_INITIAL_COLLISION_PROFILE",
    "ARA_PROFILES",
    "DMR_PROFILES",
    "AraProfile",
    "DmrProfile",
    "FPA_PROFILES",
    "FpaProfile",
    "G1_DMR_PROFILE",
    "G1_INITIAL_COLLISION_PROFILE",
    "H1_DMR_PROFILE",
    "H1_INITIAL_COLLISION_PROFILE",
    "H2_DMR_PROFILE",
    "H2_INITIAL_COLLISION_PROFILE",
    "INITIAL_COLLISION_PROFILES",
    "IkSolverProfile",
    "InitialCollisionProfile",
    "K1_DMR_PROFILE",
    "K1_INITIAL_COLLISION_PROFILE",
    "N1_DMR_PROFILE",
    "N1_INITIAL_COLLISION_PROFILE",
    "OLI_DMR_PROFILE",
    "OLI_INITIAL_COLLISION_PROFILE",
    "PM01_DMR_PROFILE",
    "PM01_INITIAL_COLLISION_PROFILE",
    "R1_DMR_PROFILE",
    "R1_INITIAL_COLLISION_PROFILE",
    "ROBOT_NEUTRAL_ARA_PROFILE",
    "T1_DMR_PROFILE",
    "T1_INITIAL_COLLISION_PROFILE",
    "get_ara_profile",
    "get_dmr_profile",
    "get_fpa_profile",
    "get_initial_collision_profile",
]
