module CUDAExtensions

using Reexport
@reexport using CUDA

import SpecialFunctions, ForwardDiff, MacroTools
import ForwardDiff.DiffRules
import ForwardDiff: @define_binary_dual_op # required until https://github.com/JuliaDiff/ForwardDiff.jl/pull/491

# Includes
include("utils.jl")
export @cufunc, @cufuncf, @cufunc_register, @cufunc_function

include("functions.jl")
# include("distributions.jl")
# include("forwarddiff.jl")

end
