const _blacklisted_replacements = Set([ :*, :/, :+, :-, :\ ])
const _irrationals = Set([:π, :ℯ, :γ, :φ, :catalan])


function should_replace(x)
    MacroTools.isexpr(x, :call) || return false
    (x.args[1] ∉ _blacklisted_replacements) || return false

    # HACK: CUDa can't handle `Irrational` so we gotta do something about that.
    # If uniary function and argument is `Irrational`, don't replace
    if length(x.args) == 2 && x.args[2] ∈ _irrationals
        return false
    end

    return true
end


function replace_device_all(ex)
    MacroTools.postwalk(ex) do x
        x = if should_replace(x)
            :(CUDA.cufunc($(x.args[1]))($(x.args[2:end]...)))
        else
            x
        end

        x = CUDA._cuint(x)
        x = CUDA._cupowliteral(x)
        x
    end
end


function tocuname(f)
    # If we have expression of form `Mod1.Mod2.f` then we define `cufunc(Mod1.Mod2.f) = cuf`
    # not `cuMod1.Mod2.f`.
    if MacroTools.isexpr(f, :.)
        Symbol(:cu, f.args[end].value)
    else
        Symbol(:cu, f)
    end
end


function make_cufunc_function(def)
    f = def[:name]
    def[:name] = tocuname(f) # Expr(:., Symbol(mod), QuoteNode(tocuname(f)))
    def[:body] = replace_device_all(def[:body])
    return :($(esc(MacroTools.combinedef(def))))
end


function make_cufunc_register(fname)
    cuname = tocuname(fname)
    return :(@inline CUDA.cufunc(::typeof($(esc(fname)))) = $(esc(cuname)))
end


function make_cufunc_diffrule(mod, f, nargs)
    # TODO: should default be `Base` or `@__MODULE__`?
    m, fname = MacroTools.isexpr(f, :.) ? (Symbol(f.args[end - 1]), f.args[end].value) : (:Base, f)
    
    # Return early if no diff-rule present
    DiffRules.hasdiffrule(m, fname, nargs) || return nothing

    # TODO: support binary also; currently the issue is that the rule isn't generated
    # properly due to the use of `map` I believe.
    nargs == 1 || throw(ArgumentError("`make_cufunc_diffrule` currently only supports nargs=1"))

    # Get the name
    cuname = tocuname(Expr(:., m, QuoteNode(fname)))

    # Create placeholders for arguments used to define new diffrule
    args = ntuple(i -> gensym(Symbol(:x, i)), nargs)
    # Get the diffrule with `args` as the representations of the arguments
    drule = DiffRules.diffrule(m, fname, args...)

    drule_interpolated = if nargs > 1
        map(drule) do dc
            MacroTools.postwalk(Meta.quot(dc)) do e
                e in args ? Expr(:$, e) : e
            end
        end
    else
        MacroTools.postwalk(Meta.quot(drule)) do e
            e in args ? Expr(:$, e) : e
        end
    end

    # Replace all methods with `cufunc`
    drule_cu = if nargs > 1
        map(replace_device_all, drule_interpolated)
    else
        replace_device_all(drule_interpolated)
    end

    # On the RHS of a DiffRule, we need to make the arguments interpolated, e.g. x -> $x
    rhs = drule_cu

    # Determine which method to use for evaluating the diff-rule created
    def_func = if nargs == 1
        :(ForwardDiff.unary_dual_definition)
    # elseif nargs == 2
    #     :(ForwardDiff.binary_dual_definition)
    else
        throw(ArgumentError("this one you gotta do yourself buddy"))
    end

    return quote
        $(DiffRules).@define_diffrule $(Symbol(mod)).$(cuname)($(esc.(args)...)) = $(esc(rhs))
        eval($(def_func)($(QuoteNode(Symbol(mod))), $(QuoteNode(cuname))))
    end
end


macro cufunc(ex)
    def = MacroTools.splitdef(ex)
    f = def[:name]
    nargs = length(def[:args])
    cufunc_function = make_cufunc_function(def)
    cufunc_register = make_cufunc_register(f)
    # `__module__` corresponds the module of the calling scope, rather than
    # `@__MODULE__` which corresponds to the scope where the code is read, i.e. here.
    cufunc_diffrule = make_cufunc_diffrule(__module__, f, nargs)
    quote
        $cufunc_function
        $cufunc_register
        $cufunc_diffrule
    end
end


"""
    @cufunc_function function f(args...) ... end

Converts the function into a CUDA-broadcastable function by replacing all
functions calls in the body with `cufunc` versions.
"""
macro cufunc_function(ex)
    def = MacroTools.splitdef(ex)
    return make_cufunc_function(def)
end


"""
    @cufunc_register Module.f
    @cufunc_register Module.f(args...)

Registers `Module.f` to its corresponding `cuf` via `CUDA.cufunc`.

Useful in cases where you have multiple different definitions of `Module.f`.
"""
macro cufunc_register(ex)
    MacroTools.@capture(ex, (f_(args_) | f_))
    return make_cufunc_register(f)
end


"""
    @cufunc_diffrule Module.f(args...)
    @cufunc_diffrule TargetModule Module.f(args...)

If it exists, diffrule from `DiffRules` is extracted and `cufunc`ed.

`TargetModule` defines the module for the diffrule to be registered,
as is required by DiffRules.jl.
"""
macro cufunc_diffrule(ex)
    MacroTools.@capture(ex, f_(args__)) || return nothing
    # `__module__` corresponds the module of the calling scope, rather than
    # `@__MODULE__` which corresponds to the scope where the code is read, i.e. here.
    return make_cufunc_diffrule(__module__, f, length(args))
end

macro cufunc_diffrule(mod, ex)
    MacroTools.@capture(ex, f_(args__)) || return nothing
    return make_cufunc_diffrule(mod, f, length(args))
end

"""
    @cufuncf function f(args...) ... end

Same as [CUDAExtensions.@cufunc](@ref) but also includes the function `f`.

This is useful if you're defining a method for the first time and also
want it to be CUDA-broadcastable, rather than making an existing method
CUDA-broadcastable.
"""
macro cufuncf(ex)
    newex = quote
        $ex
        CUDAExtensions.@cufunc($ex)
    end
    return esc(newex)
end
