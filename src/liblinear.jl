module liblinear

import StatsBase: StatisticalModel, fit, predict

include("c.jl")
import .c: FeatureNode, Parameter, Problem, valid_solver

immutable ClassificationParams{N, Y}
  solver::Int
  C::Float64
  eps::Float64
  labels::NTuple{N, Y} # expected target values
  function ClassificationParams(solver::Int, C::Float64, eps::Float64, labels::NTuple{N, Y})
    _check_params(solver, C, eps)
    new(solver, C, eps, labels)
  end
end

function ClassificationParams{N, Y}(labels::NTuple{N, Y}; solver::Int=0, C::Float64=1.0, eps::Float64=0.1)
  ClassificationParams{N, Y}(solver, C, eps, labels)
end

immutable RegressionParams
  solver::Int
  C::Float64
  eps::Float64
  function RegressionParams(solver::Int, C::Float64, eps::Float64)
    _check_params(solver, C, eps)
    new(solver, C, eps)
  end
end

function RegressionParams(; solver::Int=0, C::Float64=1.0, eps::Float64=0.1)
  RegressionParams(solver, C, eps)
end

immutable ClassificationModel{Y} <: StatisticalModel
  labels::NTuple{2, Y}
  weights::Array{Float64, 2}
end

immutable RegressionModel
  weights::Array{Float64, 1}
end

# Fit regression model for the given x,y
# x should be of type convertible to Float64
function fit{X, Y}(::Type{RegressionModel}, x::Array{X, 2}, y::Array{Y, 1}, params::ClassificationParams{2, Y}; verbose::Bool=false)
  c.set_silence(!verbose)

  prob_c = _problem(x, y)
  param_c = _parameter(params)
  p_model = c.train(prob_c, param_c)
  model = unsafe_load(p_model)

  # TODO correctly free mem after train
  n_features = int(model.nr_feature)
  n_classes = int(model.nr_class)
  w = pointer_to_array(model.w, n_features)
  # labels = pointer_to_array(model.label, n_classes)
  RegressionModel(w)
end

function predict{X}(model::RegressionModel, x::Array{X, 2})
  w = model.weights
  r = x * w
  o = ones(r)
  o ./ (o+exp(-r))
end

function _check_params(solver, C, eps)
  if eps <= 0
    throw(ArgumentError("eps <= 0, eps=$eps"))
  end
  if C <= 0
    throw(ArgumentError("C <= 0, C=$eps"))
  end
  if !valid_solver(solver)
    throw(ArgumentError("Unknown solver, solver=$solver"))
  end
end

function _problem{X, Y}(x::Array{X, 2}, y::Array{Y, 1})
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

function _parameter(p::ClassificationParams)
  Parameter(p.solver, p.eps, p.C, 0, C_NULL, C_NULL, 0.1)
end

function _parameter(p::RegressionParams)
  Parameter(p.solver, p.eps, p.C, 0, C_NULL, C_NULL, 0.1)
end

# Cases:
# - Classification/Regression
# - 2 Class/Multiclass
# - Bias/no bias

# StatBase.fit(::Type{Model}, x, y, params) :: Model
# StatBase.predict(model::Model, x) -> x'

end # module
