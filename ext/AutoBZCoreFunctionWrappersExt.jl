module AutoBZCoreFunctionWrappersExt

using AutoBZCore
using FunctionWrappers: FunctionWrapper

function AutoBZCore.init_specialized_integrand(::FunctionWrapperSpecialize, solver, f, x, p, prototype)
    func = AutoBZCore.init_specialized_integrand(DefaultSpecialize(), solver, f, x, p, prototype)
    FunctionWrapper{typeof(prototype), typeof((solver, f, x, p))}(func)
end

end