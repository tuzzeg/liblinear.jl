module liblinear

include("c.jl")
import .c: FeatureNode, Parameter, Problem, valid_solver

immutable Params
  solver::Int
  C::Float64
  eps::Float64
  function Params(solver::Int, C::Float64, eps::Float64)
    if eps <= 0
      throw(ArgumentError("eps <= 0"))
    end
    if C <= 0
      throw(ArgumentError("C <= 0"))
    end
    if !valid_solver(solver)
      throw(ArgumentError("Unknown solver, solver=$solver"))
    end
    new(solver, C, eps)
  end
end

Params(; solver::Int=0, C::Float64=1.0, eps::Float64=0.1) = Params(solver, C, eps)

immutable Model
  classes::Array{Int, 1}
  features::Int
  weights::Array{Float64, 2}
end

function problem{X, Y}(x::Array{X, 2}, y::Array{Y, 1})
  rows, cols = size(x)
  if rows != length(y)
    throw(ArgumentError("x and y dimentions should match, x=$(size(x)) y=$(size(y))"))
  end

  # X
  nodes = Array(FeatureNode, rows*(cols+1))
  p_nodes = Array(Ptr{FeatureNode}, rows)
  i = 1
  for r in 1:rows
    p_nodes[r] = pointer(nodes, i)
    for c in 1:cols
      v = x[r, c]
      if !isfinite(v)
        throw(ArgumentError("NaN of Inf value, r=$r c=$c v=$v"))
      elseif v != 0
        nodes[i] = FeatureNode(c, convert(Cfloat, v))
        i += 1
      end
    end
    nodes[i] = FeatureNode(-1, convert(Cfloat, 0))
    i += 1
  end

  # Y
  y_c = convert(Array{Cdouble, 1}, y)

  Problem(rows, cols, pointer(y_c), pointer(p_nodes), -1)
end

function parameter(p::Params)
  Parameter(p.solver, p.eps, p.C, 0, C_NULL, C_NULL, 0.1)
end

function train{X, Y}(params::Params, x::Array{X, 2}, y::Array{Y, 1}; verbose::Bool=false)
  c.set_silence(!verbose)

  prob_c = problem(x, y)
  param_c = parameter(params)
  p_model = c.train(prob_c, param_c)
  model = unsafe_load(p_model)

  # TODO correctly free mem after train
  n_features = int(model.nr_feature)
  n_classes = int(model.nr_class)
  w = pointer_to_array(model.w, (1, n_features))
  labels = pointer_to_array(model.label, n_classes)
  Model(labels, n_features, w)
end

# Cases:
# - Classification/Regression
# - 2 Class/Multiclass
# - Bias/no bias

end # module
