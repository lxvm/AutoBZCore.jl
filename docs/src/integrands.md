# Integrands

The design of AutoBZCore.jl uses multiple dispatch to provide multiple
interfaces for user integrands that allow various optimizations to be compatible
with a common interface for solvers.

```@docs
AutoBZCore.IntegralFunction
AutoBZCore.InplaceIntegralFunction
AutoBZCore.InplaceBatchIntegralFunction
AutoBZCore.CommonSolveIntegralFunction
AutoBZCore.FourierIntegralFunction
AutoBZCore.CommonSolveFourierIntegralFunction
```