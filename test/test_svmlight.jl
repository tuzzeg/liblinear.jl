require("src/svmlight.jl")

function test_line()
  l = "+1  1:2.0 2:-5 # aa"

  x = sparse(Int[], Int[], Float64[])
  y = Float64[]

  _update_from_line!(x, y, l)

  dump(x)
  dump(y)

  println(x)

  # @test_approx_eq 
end

function test_line1()
  l = "+1  1:2.0 2:-5 "

  x = sparse(Int[], Int[], Float64[])
  y = Float64[]

  _update_from_line!(x, y, l)

  dump(x)
  dump(y)

  println(x)

  # @test_approx_eq 
end

# test_line()
test_line1()
