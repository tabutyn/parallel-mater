# Contact regression fixtures

`generic_tray.glb` preserves the authored Generic scene captured on
2026-10-06: a sphere dropping into concave GenericBlockB, attached to
GenericBlockA by a generic joint. Its geometry must not follow later gallery
art edits: it reproduces the body-order-dependent inward contact normal and
subsequent artificial uphill recovery.

`generic_contact_tests.cpp` checks both body orders at 4, 8, and 16 substeps,
using independent actual-mesh penetration samples and mechanical energy.
