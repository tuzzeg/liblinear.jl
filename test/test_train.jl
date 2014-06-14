using Base.Test

require("src/liblinear.jl")
import liblinear: Params

require("src/c.jl")
import c: FeatureNode, Parameter, Problem

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
      if v != 0
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
  nr_weight = 0
  weight_label = Cint[]
  weight = Cdouble[]
  Parameter(p.solver, p.eps, p.C, nr_weight, pointer(weight_label), pointer(weight), 0.1)
end

function test_train()
  #+1 1:-1 3:1
  #+1 1:-1 4:1
  #-1 2:-1 3:-0.5
  #-1 1:1 2:-1 4:-1

  params = Params()
  param_c = parameter(params)

  x = [ -1  0 1 0;
        -1  0 0 1;
         0 -1 -0.5 0;
         1 -1 0 -1
      ]
  y = [1, 1, -1, -1]
  prob_c = problem(x, y)

  p_model = c.train(pointer(Problem[prob_c]), pointer(Parameter[param_c]))
  model = unsafe_load(p_model)
  w = pointer_to_array(model.w, 4)
  labels = pointer_to_array(model.label, 2)

  @test_approx_eq_eps [-0.67219, 0.49949, 0.41542, 0.41921] w 1e-4
end

test_train()
