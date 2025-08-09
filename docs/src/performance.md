# Performance Tips

## `executor` interface

The executor interface for integral functions and `NestedQuad` specifies whether the integrand can be evaluated in serial or in parallel and currently allows for either serial or multi-threaded execution.

```@docs
AutoBZCore.AbstractExecutor
AutoBZCore.SerialExecutor
AutoBZCore.ThreadedExecutor
```

## `specialization` interface

The `specialize` interface of CommonSolve integral functions and `NestedQuad` allows control of compilation specialization and inference.

```@docs
AutoBZCore.AbstractSpecialization
AutoBZCore.DefaultSpecialize
AutoBZCore.NoSpecialize
AutoBZCore.FullSpecialize
AutoBZCore.FunctionWrapperSpecialize
```