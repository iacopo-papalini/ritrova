"""Domain-aggregate HTTP routers (see ADR-012 §M1).

``create_app`` wires them up in a fixed order:

    1. All ``/api/…`` routers.
    2. ``pages`` last — its catch-all ``/{kind}/…`` would otherwise
       shadow more specific ``/api/…`` endpoints.

The ordering invariant matches ``CLAUDE.md:16``. Break it and the API
routes silently start returning HTML.
"""
