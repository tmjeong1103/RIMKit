# ROBOTIS K1 modifications

This K1 description is derived from ROBOTIS-GIT/ai_sapiens revision
c2880e89fb3451a07b6d2600e274224ffcf912e4 and is substantially modified.

The local XML changes include model and root-joint normalization, different
mesh paths and defaults, collision and actuator changes, differences in some
inertial representations, retargeting bodies and sites, and omission of the
upstream head body. The local torso_link.stl also differs from upstream. Its
exact historical modeling operation is unavailable; this hash-pinned local
research variant is intentionally retained to preserve the verified behavior.

The differing torso mesh is used only as visual geometry in the packaged XML;
it does not define collision, inertial, or retargeting landmarks. Replacing it
would still require a new motion baseline and visual review.
