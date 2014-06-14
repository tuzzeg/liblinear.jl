module liblinear

include("c.jl")
import .c: FeatureNode, Parameter, Problem

immutable Params
  solver::Int
  C::Float64
  eps::Float64
end

Params(; solver::Int=0, C::Float64=1.0, eps::Float64=0.1) = Params(solver, C, eps)

# Cases:
# - Classification/Regression
# - 2 Class/Multiclass
# - Bias/no bias

end # module
