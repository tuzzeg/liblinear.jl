using Base.Test

require("src/liblinear.jl")
import liblinear: Params, Model, train

function test_train()
  #+1 1:-1 3:1
  #+1 1:-1 4:1
  #-1 2:-1 3:-0.5
  #-1 1:1 2:-1 4:-1

  params = Params()

  x = [ -1  0 1 0;
        -1  0 0 1;
         0 -1 -0.5 0;
         1 -1 0 -1
      ]
  y = [1, 1, -1, -1]

  model = train(params, x, y)
  dump(model)

  @test_approx_eq_eps [-0.67219, 0.49949, 0.41542, 0.41921] model.weights 1e-4
end

test_train()
