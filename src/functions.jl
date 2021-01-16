using SpecialFunctions

# https://github.com/JuliaMath/SpecialFunctions.jl/blob/d18ff04178dd37a60cc716a60ab3caf1a6903f43/src/gamma.jl#L20
@cufunc function SpecialFunctions.digamma(x)
    if x <= 0 # reflection formula
        ψ = -π / CUDA.tan(π * x)
        x = 1 - x
    else
        ψ = zero(x)
    end
    if x < 7
        # shift using recurrence formula
        ν = one(x)
        n = 7 - CUDA.floor(x)
        while ν <= n - 1
            ψ -= inv(x + ν)
            ν += one(x)
        end
        ψ -= inv(x)
        x += n
    end
    t = inv(x)
    ψ += CUDA.log(x) - t / 2
    t *= t # 1/z^2
    # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
    # TODO: Something really weird going on here. Can't use f32; tried
    # [×] `oftype` of coeffs
    # [×] `Float32` of coeffs
    # [×] `...f0` of coeffs
    # [×] Even failed if I just straight up don't even do this computation o.O
    ψ -= (
        t *
        # @evalpoly(
        #     t,
        #     oftype(x, 0.08333333333333333), oftype(x, -0.008333333333333333),
        #     oftype(x, 0.003968253968253968), oftype(x, -0.004166666666666667),
        #     oftype(x, 0.007575757575757576), oftype(x, -0.021092796092796094),
        #     oftype(x, 0.08333333333333333), oftype(x, -0.4432598039215686)
        # )
        @evalpoly(
            t,
            0.08333333333333333, -0.008333333333333333,
            0.003968253968253968, -0.004166666666666667,
            0.007575757575757576, -0.021092796092796094,
            0.08333333333333333, -0.4432598039215686
        )
    )
    return ψ
end

function _cutrigamma(x)
  ψ = zero(x)
  if x < 8
    # shift using recurrence formula
    n = 8 - CUDA.floor(x)
    ψ += inv(x)^2
    ν = one(x)
    while ν <= n - 1
      ψ += inv(x + ν)^2
      ν += one(x)
    end
    x += n
  end
  t = inv(x)
  w = t * t # 1/z^2
  ψ += t + 0.5 * w
  # the coefficients here are Float64(bernoulli[2:9])
  ψ += t * w * @evalpoly(w,0.16666666666666666,-0.03333333333333333,0.023809523809523808,-0.03333333333333333,0.07575757575757576,-0.2531135531135531,1.1666666666666667,-7.092156862745098)
  return ψ
end

@cufunc function SpecialFunctions.trigamma(x)
  if x <= 0 # reflection formula
    return (π / CUDA.sin(π * x))^2 - _cutrigamma(1 - x)
  else
    return _cutrigamma(x)
  end
end

@cufunc SpecialFunctions.loggamma(x) = CUDA.lgamma(x)
# @cufunc SpecialFunctions.logbeta(x, y) = SpecialFunctions.loggamma(x) + SpecialFunctions.loggamma(y) - SpecialFunctions.loggamma(x + y)

@cufunc SpecialFunctions.erf(x) = CUDA.erf(x)
@cufunc SpecialFunctions.erfc(x) = CUDA.erfc(x)
@cufunc SpecialFunctions.erfcx(x) = CUDA.erfcx(x)
@cufunc SpecialFunctions.erfinv(x) = CUDA.erfinv(x)
@cufunc SpecialFunctions.erfcinv(x) = CUDA.erfcinv(x)

#####################
### Distributions ###
#####################
using StatsFuns
import StatsFuns: gammalogpdf, zval, xval, normlogpdf, normcdf, log2π, invsqrt2

@cufunc StatsFuns.gammalogpdf(k::Real, θ::Real, x::Number) = -loggamma(k) - k * log(θ) + (k - 1) * log(x) - x / θ

    
@cufunc function StatsFuns.xval(μ::Real, σ::Real, z::Number)
    if isinf(z) && iszero(σ)
        μ + one(σ) * z
    else
        μ + σ * z
    end
end
@cufunc StatsFuns.zval(μ::Real, σ::Real, x::Number) = (x - μ) / σ

# logpdf
@cufunc_register StatsFuns.normlogpdf
@cufunc_function StatsFuns.normlogpdf(z::Number) = -(abs2(z) + log2π)/2
@cufunc_function function StatsFuns.normlogpdf(μ::Real, σ::Real, x::Number)
    if iszero(σ)
        if x == μ
            z = zval(μ, one(σ), x)
        else
            z = zval(μ, σ, x)
            σ = one(σ)
        end
    else
        z = zval(μ, σ, x)
    end
    normlogpdf(z) - log(σ)
end

# cdf
@cufunc_register StatsFuns.normcdf
@cufunc_function StatsFuns.normcdf(z::Number) = erfc(-z * invsqrt2)/2
@cufunc_function function StatsFuns.normcdf(μ::Real, σ::Real, x::Number)
    if iszero(σ) && x == μ
        z = zval(zero(μ), σ, one(x))
    else        
        z = zval(μ, σ, x)        
    end
    normcdf(z)
end

# logcdf
@cufunc_register StatsFuns.normlogcdf
@cufunc_function StatsFuns.normlogcdf(z::Number) = z < -1.0 ?
    log(erfcx(-z * invsqrt2)/2) - abs2(z)/2 :
    log1p(-erfc(z * invsqrt2)/2)
@cufunc_function function StatsFuns.normlogcdf(μ::Real, σ::Real, x::Number)
    if iszero(σ) && x == μ
        z = zval(zero(μ), σ, one(x))
    else        
        z = zval(μ, σ, x)        
    end
    normlogcdf(z)
end

# truncated normlogpdf
@cufuncf function truncatednormlogpdf(μ, σ, x, lb, ub)
    logtp = StatsFuns.normlogcdf(μ, σ, ub) - StatsFuns.normlogcdf(μ, σ, lb)
    # TODO: deal with outside of boundary
    StatsFuns.normlogpdf(μ, σ, x) - logtp
    # TODO: seems like there's something messed up with the way we return `Inf`
    # if lb <= x <= ub
    #     StatsFuns.normlogpdf(μ, σ, x) - logtp
    # else
    #     TF = float(eltype(x))
    #     -TF(Inf)
    # end
end
