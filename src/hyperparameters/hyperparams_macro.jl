using Bijectors

export @hyperparams

const _HYPERPARAMS_ALLOWED_OPTIONS = (:transform, :space, :prior_space)
const _HYPERPARAMS_BUILTIN_TRANSFORMS = (:log, :logit, :identity)
const _HYPERPARAMS_CALL_ALIASES = Dict(
    :Log => :log,
    :Logit => :logit,
    :Identity => :identity,
)
const _HYPERPARAMS_VALID_SPACES = (:natural, :working)

@inline function _hyperparams_builtin_transform(sym::Symbol)
    if sym == :log
        return elementwise(log)
    elseif sym == :logit
        return Bijectors.Logit(0.0, 1.0)
    elseif sym == :identity
        return identity
    else
        error("@hyperparams: unknown builtin transform `$(sym)`.")
    end
end

function _hyperparams_is_free_clause(clause::Expr)
    if clause.head == :call && clause.args[1] == :~
        return clause, Any[]
    elseif clause.head == :tuple && !isempty(clause.args)
        base = clause.args[1]
        if base isa Expr && base.head == :call && base.args[1] == :~
            return base, clause.args[2:end]
        end
    end
    return nothing, nothing
end

function _hyperparams_option_dict(option_exprs)
    opts = Dict{Symbol, Any}()
    for opt in option_exprs
        if opt isa LineNumberNode
            continue
        elseif opt isa Expr && opt.head == :(=) && opt.args[1] isa Symbol
            key = opt.args[1]::Symbol
            if key ∉ _HYPERPARAMS_ALLOWED_OPTIONS
                valid_options = join(_HYPERPARAMS_ALLOWED_OPTIONS, ", ", " or ")
                error(
                    "@hyperparams: unsupported option `$key`. " *
                        "Valid options are: $valid_options"
                )
            end
            if haskey(opts, key)
                error("@hyperparams: option `$key` specified more than once.")
            end
            opts[key] = opt.args[2]
        else
            error("@hyperparams: expected `option = value` syntax in options, got `$(opt)`.")
        end
    end
    return opts
end

function _hyperparams_alias_transform(sym::Symbol)
    if sym in _HYPERPARAMS_BUILTIN_TRANSFORMS
        return :(IntegratedNestedLaplace._hyperparams_builtin_transform($(QuoteNode(sym))))
    end
    return nothing
end

function _hyperparams_alias_transform(expr::Expr)
    if expr.head == :call && length(expr.args) == 1 && expr.args[1] isa Symbol
        func_sym = expr.args[1]::Symbol
        if haskey(_HYPERPARAMS_CALL_ALIASES, func_sym)
            mapped = _HYPERPARAMS_CALL_ALIASES[func_sym]
            return :(IntegratedNestedLaplace._hyperparams_builtin_transform($(QuoteNode(mapped))))
        end
    end
    return nothing
end

function _hyperparams_resolve_transform(value)
    value === nothing && return :identity

    expr = value isa QuoteNode ? value.value : value
    expr isa String && (expr = Symbol(expr))

    if expr isa Symbol
        alias = _hyperparams_alias_transform(expr)
        return alias === nothing ? expr : alias
    elseif expr isa Expr
        alias = _hyperparams_alias_transform(expr)
        return alias === nothing ? expr : alias
    else
        return expr
    end
end

function _hyperparams_space_symbol(expr, name::Symbol)
    expr === nothing && return nothing

    literal = expr isa QuoteNode ? expr.value : expr
    if literal isa String
        literal = Symbol(literal)
    end

    literal isa Symbol || error("@hyperparams: prior space for parameter `$name` must be given as a Symbol (e.g. :natural).")
    literal in _HYPERPARAMS_VALID_SPACES || error("@hyperparams: prior space for parameter `$name` must be :natural or :working, got `$(literal)`.")
    return literal
end

function _hyperparams_resolve_space(opts::Dict{Symbol, Any}, name::Symbol)
    space_val = _hyperparams_space_symbol(get(opts, :space, nothing), name)
    prior_space_val = _hyperparams_space_symbol(get(opts, :prior_space, nothing), name)

    if !(space_val === nothing || prior_space_val === nothing || space_val == prior_space_val)
        error("@hyperparams: conflicting `space` / `prior_space` declarations for parameter `$name`.")
    end

    selected = prior_space_val === nothing ? space_val : prior_space_val
    return QuoteNode(selected === nothing ? :working : selected)
end

function _hyperparams_hyperparameter_expr(base::Expr, option_exprs)
    if length(base.args) < 3
        error("@hyperparams: `name ~ prior` clause must provide a prior distribution.")
    end

    name_expr = base.args[2]
    prior_expr = base.args[3]
    name_expr isa Symbol || error("@hyperparams: hyperparameter name must be a Symbol, got `$(name_expr)`.")

    options = _hyperparams_option_dict(option_exprs)
    transform_expr = _hyperparams_resolve_transform(get(options, :transform, nothing))
    prior_space_expr = _hyperparams_resolve_space(options, name_expr)

    hyper_call = Expr(
        :call, :Hyperparameter, prior_expr,
        Expr(:kw, :transform, transform_expr),
        Expr(:kw, :prior_space, prior_space_expr)
    )

    return name_expr, hyper_call
end

function _hyperparams_process_clause(clause)
    if clause isa Expr
        base, option_exprs = _hyperparams_is_free_clause(clause)
        if base !== nothing
            name, hyper_expr = _hyperparams_hyperparameter_expr(base, option_exprs)
            return :free, name, hyper_expr
        elseif clause.head == :(=) && clause.args[1] isa Symbol
            return :fixed, clause.args[1], clause.args[2]
        elseif clause.head == :call && clause.args[1] == :~
            # Likely missing parentheses when trying to add options
            error(
                "@hyperparams: syntax error in clause `$(clause)`. " *
                    "Did you forget parentheses? When adding options like `transform` or `space`, " *
                    "wrap the entire clause in parentheses: `(name ~ prior, transform = log, space = natural)`"
            )
        end
    end

    error(
        "@hyperparams: unsupported clause `$(clause)`. " *
            "Use `name ~ prior` or `(name ~ prior, option = value)` for free parameters, " *
            "or `name = value` for fixed parameters."
    )
end

function _hyperparams_namedtuple(entries::Vector{Expr})
    return Expr(:tuple, Expr(:parameters, entries...))
end

"""
    @hyperparams begin ... end

Convenient macro for creating `HyperparameterSpec` objects with clean, declarative syntax.

# Syntax

**Free parameters** use the `~` operator:
```julia
name ~ prior                                         # Identity transform, working space (default)
(name ~ prior, transform = transform_expr)           # With custom transform
(name ~ prior, transform = log, space = natural)     # With options
```

**Fixed parameters** use `=`:
```julia
name = value
```

# Builtin Transforms

The macro provides convenient shortcuts for common transformations:

- `log` → `elementwise(log)` for positive parameters (σ, τ, κ, etc.)
- `logit` → `Bijectors.Logit(0.0, 1.0)` for parameters in (0,1) (ρ, correlation, etc.)
- `identity` → no transformation (default for parameters already unconstrained)

**Call aliases** are also supported: `Log()`, `Logit()`, `Identity()`

For maximum flexibility, you can also provide any custom bijector expression directly.

# Options

- `transform`: Bijector mapping natural → working space (default: `identity`)
- `space` or `prior_space`: Space in which the prior is specified
  - `:natural` - Prior is on the natural (constrained) parameter space
  - `:working` - Prior is already on the working (unconstrained) space (default)

**Important:** When specifying options, wrap the entire clause in parentheses!

# Examples

```julia
using Distributions, Bijectors

# Example 1: PC priors (Penalizing Complexity priors)
# Exponential prior on σ, optimized on log(σ)
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    μ = 0.0  # Fixed parameter
end

# Example 2: Using call aliases
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = Log(), space = natural)
    (ρ ~ Beta(2, 2), transform = Logit(), space = natural)
end

# Example 3: Custom bijector expression
spec = @hyperparams begin
    (τ ~ Gamma(2, 1), transform = elementwise(x -> log(x + 1)), space = natural)
end

# Example 4: Identity transform (default) - prior already in working space
spec = @hyperparams begin
    μ ~ Normal(0, 10)      # No transformation needed
    σ ~ Gamma(2, 1)        # Prior specified directly in working space
end

# Example 5: Mixed free and fixed parameters
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    μ = 0.0    # Fixed intercept
    n = 100    # Fixed sample size
end

# Example 6: Using prior_space keyword (synonym for space)
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, prior_space = natural)
end
```

# Comparison with Manual Construction

The macro provides a much more concise syntax compared to manual construction:

```julia
# With macro (recommended)
spec = @hyperparams begin
    (σ ~ Exponential(1.0), transform = log, space = natural)
    (ρ ~ Beta(2, 2), transform = logit, space = natural)
    μ = 0.0
end

# Manual construction (verbose)
spec = HyperparameterSpec(
    free = (
        σ = Hyperparameter(Exponential(1.0), transform=elementwise(log), prior_space=:natural),
        ρ = Hyperparameter(Beta(2, 2), transform=Bijectors.Logit(0.0, 1.0), prior_space=:natural)
    ),
    fixed = (μ = 0.0,)
)
```

# Notes

- At least one free parameter (using `~`) is required
- When specifying options, use parentheses: `(name ~ prior, option = value)`
- Options use unquoted symbols: `space = natural`, not `space = :natural`
- Parameters cannot be specified multiple times
- The `space` and `prior_space` keywords are synonyms - use whichever is more natural

# See Also

- [`HyperparameterSpec`](@ref): The underlying type created by this macro
- [`Hyperparameter`](@ref): Individual hyperparameter specification
"""
macro hyperparams(block)
    clauses = block isa Expr && block.head == :block ? block.args : Any[block]

    free_entries = Expr[]
    fixed_entries = Expr[]
    seen = Set{Symbol}()

    for clause in clauses
        clause isa LineNumberNode && continue

        kind, name, value_expr = _hyperparams_process_clause(clause)

        if name in seen
            error("@hyperparams: parameter `$name` specified more than once.")
        end
        push!(seen, name)

        if kind == :free
            push!(free_entries, Expr(:kw, name, value_expr))
        elseif kind == :fixed
            push!(fixed_entries, Expr(:kw, name, value_expr))
            # Detect if user forgot parentheses and put options on separate lines
            if name in _HYPERPARAMS_ALLOWED_OPTIONS
                error(
                    "@hyperparams: found `$name = ...` as a fixed parameter, but `$name` is typically used as an option. " *
                        "Did you forget parentheses? Use: `(param ~ prior, $name = value)` instead of separate lines."
                )
            end
        end
    end

    isempty(free_entries) && error("@hyperparams: at least one free parameter (specified with `~`) is required.")

    free_nt = _hyperparams_namedtuple(free_entries)
    fixed_nt = _hyperparams_namedtuple(fixed_entries)

    call_expr = Expr(
        :call, :HyperparameterSpec,
        Expr(:kw, :free, free_nt),
        Expr(:kw, :fixed, fixed_nt)
    )

    return esc(call_expr)
end
