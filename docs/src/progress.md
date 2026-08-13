```@meta
CurrentModule = InterfacialWaves
```

# [Progress](@id progress)

## Implemented and validated

- pure-gravity Longuet–Higgins travelling waves;
- pure-gravity collocation travelling waves;
- pure-gravity Stokes-wave linear stability through `LinearStabProblem`;
- fixed-B gravity–capillary target states, including validated analytical and
  finite-difference Jacobian paths;
- finite-Reynolds-number viscous gravity–capillary residuals and sequential
  energy-continuation paths for both derived-B and fixed-B formulations;
- finite-Reynolds-number analytical Jacobians checked against centered finite
  differences;
- two-fluid interfacial Stokes waves (`InterfacialStokesProblem`) with steepness
  continuation and fifth-order Stokes expansion reference (`FifthOrderStokes`);
- interfacial linear stability following Murashige & Choi (2022) via `LinearStabProblem`;
- axisymmetric standing waves in a cylindrical basin (`AxiStandingWaves` submodule),
  using HOSE boundary-integral time integration and Newton shooting continuation.

## In development

- complete documentation and reference validation for gravity–capillary and
  viscous stability (σ vs p sweeps, regression tests);
- standing-wave stability analysis.
