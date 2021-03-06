using Base.Test

import StatsBase: fit, predict

require("src/liblinear.jl")
import liblinear: ClassificationParams, ClassificationModel, RegressionModel

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

function test_train_y_2dim()
  params = ClassificationParams((0, 1); eps=0.01)

  model = fit(RegressionModel, x_train, y_train', params)
  @test_approx_eq_eps [0.174465,0.477886,-0.827468,-0.373797] model.weights 1e-4
end

function test_train_unknown_class()
  params = ClassificationParams((-1, 1); eps=0.01)
  @test_throws KeyError fit(RegressionModel, x_train, y_train, params)
end

function test_classification()
  params = ClassificationParams((0, 1); eps=0.01)
  model = fit(ClassificationModel{Int}, x_train, y_train, params)
  @test_approx_eq_eps [0.174465,0.477886,-0.827468,-0.373797] model.weights 1e-4

  y1 = predict(model, x_test)
  @assert [0, 0, 1, 1] == y1
end

function test_classification_strings()
  params = ClassificationParams(("I. setosa", "I. versicolor"); eps=0.01)
  y_str = ["I. setosa", "I. setosa", "I. versicolor", "I. versicolor"]
  model = fit(ClassificationModel{ASCIIString}, x_train, y_str, params)
  y1 = predict(model, x_test)
  @assert ["I. setosa", "I. setosa", "I. versicolor", "I. versicolor"] == y1
end

function test_regression_with_bias()
  params = ClassificationParams((0, 1); eps=0.01, bias=1.0)
  model = fit(RegressionModel, x_train, y_train, params)
  @test_approx_eq_eps [0.157814, 0.464984, -0.826937, -0.372264] model.weights 1e-4
  @test_approx_eq_eps 0.14089087 model.bias 1e-4

  y1 = predict(model, x_test)
  @test_approx_eq_eps [0.772256, 0.729764, 0.130827, 0.134380] y1 1e-4
end

function test_classification_with_bias()
  params = ClassificationParams((0, 1); eps=0.01, bias=1.0)
  model = fit(ClassificationModel{Int}, x_train, y_train, params)
  @test_approx_eq_eps [0.157814, 0.464984, -0.826937, -0.372264] model.weights 1e-4
  @test_approx_eq_eps 0.14089087 model.bias 1e-4

  y1 = predict(model, x_test)
  @assert [0, 0, 1, 1] == y1
end

test_train()
test_train_y_2dim()
test_train_unknown_class()

test_classification()
test_classification_strings()

test_regression_with_bias()
test_classification_with_bias()
