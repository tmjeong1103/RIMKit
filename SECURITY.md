# Security policy

RIMKit treats motion files and MJCF assets as untrusted input boundaries.
SOMA `.npz` files are loaded with pickle disabled, object arrays are rejected,
and web uploads enforce file-size, frame-count, and path limits.

Please report security issues privately to
[taemoon-jeong@korea.ac.kr](mailto:taemoon-jeong@korea.ac.kr) rather than
opening a public issue with exploit details. Include the affected version,
reproduction steps, and potential impact when possible.
