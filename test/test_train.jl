using Base.Test

import StatsBase: fit

require("src/liblinear.jl")
import liblinear: ClassificationParams, RegressionModel

x = [ -1  0    1  0;
      -1  0    0  1;
       0 -1 -0.5  0;
       1 -1    0 -1
    ]

function test_train()
  params = ClassificationParams((-1, 1))
  y = [1, 1, -1, -1]

  model = fit(RegressionModel, x, y, params)
  dump(model)

  @test_approx_eq_eps [-0.67219, 0.49949, 0.41542, 0.41921] model.weights 1e-4
end

test_train()
