using Base.Test

import StatsBase: fit, predict

require("src/liblinear.jl")
import liblinear: ClassificationParams, RegressionModel

x_train = [
  5.1 3.5 1.4 0.2;
  4.9 3.0 1.4 0.2;
  7.0 3.2 4.7 1.4;
  6.4 3.2 4.5 1.5
]
y_train = [0, 0, 1, 1]

x_test = [
  4.7 3.2 1.3 0.2;
  4.6 3.1 1.5 0.2;
  6.5 2.8 4.6 1.5;
  5.7 2.8 4.5 1.3
]
y_test = [0, 0, 1, 1]

function test_train()
  params = ClassificationParams((0, 1); eps=0.01)

  model = fit(RegressionModel, x_train, y_train, params)
  @test_approx_eq_eps [0.174465,0.477886,-0.827468,-0.373797] model.weights 1e-4

  y1 = predict(model, x_test)
  @test_approx_eq_eps [0.768302,0.724725,0.130681,0.132732] y1 1e-4
end

test_train()
