import ForwardDiff
import ForwardDiff: DiffRules
import ForwardDiff: @define_binary_dual_op # required until https://github.com/JuliaDiff/ForwardDiff.jl/pull/491

#################
# CUDA.jl rules #
#################
# No need to register new adjoints or anything for reverse-mode AD since the use of
# the functions in the adjoints will be correctly replaced in the broadcasting mechanism.

DiffRules.@define_diffrule CUDAExtensions.culoggamma(a) = :(CUDAExtensions.cudigamma($a))
DiffRules.@define_diffrule CUDAExtensions.cudigamma(a) = :(CUDAExtensions.cutrigamma($a))

eval(ForwardDiff.unary_dual_definition(:CUDAExtensions, :culoggamma))
eval(ForwardDiff.unary_dual_definition(:CUDAExtensions, :cudigamma))
