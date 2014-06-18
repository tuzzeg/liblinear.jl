using Base.Test

require("src/c.jl")
import c: FeatureNode, Parameter, Problem

function test_train()
  #+1 1:-1 3:1
  #+1 1:-1 4:1
  #-1 2:-1 3:-0.5
  #-1 1:1 2:-1 4:-1

  n(i, x) = FeatureNode(i, x)
  xx = FeatureNode[
    n(1, -1), n(3, 1), n(-1, 0),
    n(1, -1), n(4, 1), n(-1, 0),
    n(2, -1), n(3, -0.5), n(-1, 0),
    n(1, 1), n(2, -1), n(4, -1), n(-1, 0)
  ]
  x = Ptr{FeatureNode}[
    pointer(xx, 1),
    pointer(xx, 4),
    pointer(xx, 7),
    pointer(xx, 10)
  ]

  l = 4
  n = 4
  y = Cdouble[1, 1, 2, 2]
  bias = -1
  problem = Problem(l, n, pointer(y), pointer(x), bias)

  solver_type = 0
  eps = 0.1
  C = 1.0
  nr_weight = 0
  weight_label = Cint[]
  weight = Cdouble[]
  p = 0.1
  param = Parameter(solver_type, eps, C, nr_weight, pointer(weight_label), pointer(weight), p)

  c.set_silence(false)

  p_model = c.train(problem, param)
  model = unsafe_load(p_model)
  w = pointer_to_array(model.w, int(model.nr_feature))
  labels = pointer_to_array(model.label, int(model.nr_class))

  println(w)
  println(labels)

  @test_approx_eq_eps [-0.67219, 0.49949, 0.41542, 0.41921] w 1e-4
end

test_train()
