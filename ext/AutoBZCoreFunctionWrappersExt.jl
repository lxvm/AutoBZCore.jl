module AutoBZCoreFunctionWrappersExt

using AutoBZCore
using FunctionWrappers: FunctionWrapper

function AutoBZCore.init_specialized_integrand(::FunctionWrapperSpecialize, cache, f, x, p, prototype)
    func = AutoBZCore.init_specialized_integrand(DefaultSpecialize(), cache, f, x, p, prototype)
    FunctionWrapper{typeof(prototype), typeof((cache, f, x, p))}(func)
end

end