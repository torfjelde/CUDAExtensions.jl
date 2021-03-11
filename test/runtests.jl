using Test
using Zygote, CUDA
import SpecialFunctions
import StatsFuns
import CUDAExtensions

CUDA.allowscalar(false)


function test_ad(f, args...)
    y, ∂_cpu = Zygote.pullback(f, args...)
    _, ∂_gpu = Zygote.pullback(f, map(cu, args)...)

    ∇_cpu = ∂_cpu(one(y))
    ∇_gpu = ∂_gpu(one(y))

    # Convert back to CPU
    ∇_gpu_cpu = map(Array, ∇_gpu)

    for i = 1:length(∇_cpu)
        # Ensure that the values align
        @test ∇_cpu[i] ≈ ∇_gpu_cpu[i] atol=1e-3

        # Ensure that eltype is preserved
        @test eltype(∇_gpu_cpu[i]) == eltype(args[i])
    end
end

test_ad_broadcast(f, args...) = test_ad(args...) do (inner_args...)
    sum(f.(inner_args...))
end

@testset "SpecialFunctions.jl" begin
    x = Float32.(randn(10))

    test_ad_broadcast(SpecialFunctions.erf, x)
    test_ad_broadcast(SpecialFunctions.erfinv, SpecialFunctions.erf.(x))
    test_ad_broadcast(SpecialFunctions.erfc, x)
    test_ad_broadcast(SpecialFunctions.erfcinv, SpecialFunctions.erfc.(x))
    # test_ad_broadcast(SpecialFunctions.erfi, x) # Apparently doesn't have a CUDA equivalent
    test_ad_broadcast(SpecialFunctions.erfcx, x)
end
