module liblinear

import StatsBase: StatisticalModel, fit, predict

include("c.jl")
using .c: FeatureNode, Parameter, Problem, valid_solver, free_and_destroy_model

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

immutable Model
  nr_class::Int # number of classes
  nr_feature::Int
  weights::Array{Float64, 2}
  labels::Array{Int, 1} # label of each class
  bias::Float32
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
function fit{Y}(::Type{RegressionModel}, x, y::Array{Y, 1}, params::ClassificationParams{2, Y}; verbose::Bool=false)
  y_map = (Y=>Cdouble)[y=>i for (i, y) in enumerate(params.labels)]
  model = _train(_parameter(params), _problem(x, y; map_y=(y)->y_map[y]))
  n_features, n_classes = size(model.weights)
  @assert 1 == n_classes
  RegressionModel(reshape(model.weights, (n_features,)))
end

function fit{Y}(T, x, y::Array{Y, 2}, params; verbose::Bool=false)#={{{=##=}}}=#
  n_examples, n_features = size(x)
  fit(T, x, reshape(y, (n_examples,)), params; verbose=verbose)
end

function predict(model::RegressionModel, x)
  _prob(x, model.weights)
end

# Classification

function fit{Y}(::Type{ClassificationModel{Y}}, x, y::Array{Y, 1}, params::ClassificationParams{2, Y}; verbose::Bool=false)
  y_map = (Y=>Cdouble)[y=>i for (i, y) in enumerate(params.labels)]
  model = _train(_parameter(params), _problem(x, y; map_y=(y)->y_map[y]))
  n_features, n_classes = size(model.weights)

  @assert 1 == n_classes
  @assert [1, 2] == model.labels

  w = reshape(model.weights, (n_features, n_classes))
  ClassificationModel(params.labels, w)
end

function predict{X, Y}(model::ClassificationModel{Y}, x::Array{X, 2})
  probs = _prob(x, model.weights)
  Y[0.5<p ? model.labels[1] : model.labels[2]  for p in probs]
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

function _problem{X, Y}(x::Array{X, 2}, y::Array{Y, 1}; map_y::Union(Function, Nothing)=nothing)
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
  y_c = convert(Array{Cdouble, 1}, (map_y != nothing ? map(map_y, y) : y))

  Problem(rows, cols, pointer(y_c), pointer(p_nodes), -1)
end

function _problem{X, Y}(x::SparseMatrixCSC{X, Int}, y::Array{Y, 1}; map_y::Union(Function, Nothing)=nothing)
  rows, cols = size(x)
  c = nfilled(x)
  if rows != length(y)
    throw(ArgumentError("x and y dimentions should match, x=$(size(x)) y=$(size(y))"))
  end

  # Need CSR
  x = x'

  # X
  # Allocate nodes for each row (count: c) and 1 sentinel at each row (count: rows)
  nodes = Array(FeatureNode, c+rows)
  p_nodes = Array(Ptr{FeatureNode}, rows)
  for r in 1:rows
    i0, i1 = x.colptr[r], x.colptr[r+1]-1
    p_nodes[r] = pointer(nodes, i0+r-1)
    for i in i0:i1
      c = x.rowval[i]
      v = x.nzval[i]
      if !isfinite(v)
        throw(ArgumentError("NaN of Inf value, r=$r c=$c v=$v"))#={{{=##=}}}=#
      end
      nodes[i+r-1] = FeatureNode(c, convert(Cfloat, v))
    end
    nodes[i1+r] = FeatureNode(-1, convert(Cfloat, 0))
  end

  # Y
  y_c = convert(Array{Cdouble, 1}, (map_y != nothing ? map(map_y, y) : y))

  Problem(rows, cols, pointer(y_c), pointer(p_nodes), -1)
end

function _parameter(p::ClassificationParams)
  Parameter(p.solver, p.eps, p.C, 0, C_NULL, C_NULL, 0.1)
end

function _parameter(p::RegressionParams)
  Parameter(p.solver, p.eps, p.C, 0, C_NULL, C_NULL, 0.1)
end

# Train model, correctly free resources allocated by C lib
function _train(param::Parameter, problem::Problem; verbose=false)
  c.set_silence(!verbose)

  p_model = c.train(problem, param)
  model = unsafe_load(p_model)

  n_features = int(model.nr_feature)
  n_classes = int(model.nr_class)
  # Different weight vector in two classes vs multi class classification
  w_dims = (2 == n_classes) ? (n_features,1) : (n_features, n_classes)
  w = copy(pointer_to_array(model.w, w_dims))
  labels = copy(pointer_to_array(model.label, n_classes))
  bias = float(model.bias)

  free_and_destroy_model(p_model)

  Model(n_features, n_classes, w, labels, bias)
end

function _prob(x, w)
  r = x * w
  o = ones(r)
  o ./ (o+exp(-r))
end

# Cases:
# - Classification/Regression
# - 2 Class/Multiclass
# - Bias/no bias
# - dense/sparse

# StatBase.fit(::Type{Model}, x, y, params) :: Model
# StatBase.predict(model::Model, x) -> x'

end # module
