using Base.Test

require("src/svmlight.jl")

function test_line()
  io = IOBuffer("+1  1:2.0 2:-5 # aa")

  x, y = read_svmlight(io)

  @test_approx_eq [2.0 -5] x
  @test_approx_eq [1] y
end

function test_line_ending_space()
  io = IOBuffer("+1  1:2.0 2:-5 ")

  x, y = read_svmlight(io)

  @test_approx_eq [2.0 -5] x
  @test_approx_eq [1] y
end

function test_file()
  io = IOBuffer("""
  # comment
  1.0 2:2.5 10:-5.2 15:1.5 # an inline comment
  2.0 5:1.0 12:-3
  # another comment
  3.0 20:27
  """)
  x, y = read_svmlight(io)

  @assert (3, 20) == size(x)
  x_exp = [
    0 2.5 0 0 0 0 0 0 0 -5.2 0  0 0 0 1.5 0 0 0 0  0;
    0 0   0 0 1 0 0 0 0    0 0 -3 0 0   0 0 0 0 0  0;
    0 0   0 0 0 0 0 0 0    0 0  0 0 0   0 0 0 0 0 27
  ]
  @test_approx_eq x_exp x
  @test_approx_eq [1, 2, 3] y
end

function test_comment_only()
  io = IOBuffer("# comment")
  x, y = read_svmlight(io)

  @assert (0, 0) == size(x)
  @assert (0,) == size(y)
end

function test_comments()
  io = IOBuffer("# comment\n  # aa")
  x, y = read_svmlight(io)

  @assert (0, 0) == size(x)
  @assert (0,) == size(y)
end

function test_target_only()
  io = IOBuffer("1 # comment\n  # aa  # bb   ")
  x, y = read_svmlight(io)

  @assert (1, 0) == size(x)
  @assert (1,) == size(y)

  @test_approx_eq [1] y
end

test_line()
test_line_ending_space()
test_file()
test_comment_only()
test_comments()
test_target_only()
