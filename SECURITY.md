# Security policy

CoRe treats motion files and MJCF assets as untrusted input boundaries.
SOMA NPZ files are loaded with pickle disabled, object arrays are rejected,
and web uploads enforce file-size, frame-count, and path limits.

Please report security issues privately to the repository owner rather than
opening a public issue with exploit details.
