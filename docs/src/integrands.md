# Integrands

The design of AutoBZCore.jl uses multiple dispatch to provide multiple
interfaces for user integrands that allow various optimizations to be compatible
with a common interface for solvers.
Unfortunately, not all algorithms support all integrands, since the underlying
libraries must support the same interface.

# Functions

A user can pass an integrand of the form `f(x,p)` in the same way as in Integrals.jl

# `ParameterIntegrand`

```@docs
AutoBZCore.ParameterIntegrand
```

# `InplaceIntegrand`

```@docs
AutoBZCore.InplaceIntegrand
```

# `BatchIntegrand`

```@docs
AutoBZCore.BatchIntegrand
```

# `FourierIntegrand`

```@docs
AutoBZCore.FourierIntegrand
```

# `NestedBatchIntegrand`

```@docs
AutoBZCore.NestedBatchIntegrand
```

# NestedIntegrand

```@docs
AutoBZCore.NestedIntegrand
```