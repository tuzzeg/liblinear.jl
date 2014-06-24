using Base.Test

require("src/svmlight.jl")

function test_line()
  io = IOBuffer("+1  1:2.0 2:-5 # aa")

  x, y = read_svmlight(io)

  @test_approx_eq [2.0 -5] x
  @test_approx_eq [1] y
end

function test_line1()
  io = IOBuffer("+1  1:2.0 2:-5 ")

  x, y = read_svmlight(io)

  @test_approx_eq [2.0 -5] x
  @test_approx_eq [1] y
end

test_line()
test_line1()
