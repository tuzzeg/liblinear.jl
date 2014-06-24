function read_svmlight(io::IO)
  # Build transposed sparse matrix
  x = SparseMatrixCSC(0, 0, Int[], Int[], Float64[])
  y = Float64[]
  push!(x.colptr, 1)
  for l in eachline(io)
    l = chomp(l)
    _update_from_line!(x, y, l)
  end
  x', y
end

# States
# 0 -> 0 [' ']
# 0 -> 1 [<TARGET> ' ']
# 0 -> 3 [<TARGET> '#']
# 0 -> 3 ['#']
# 0 -> 4 [<TARGET> $]
# 0 -> 4 [$]
# 1 -> 2 [<COL> ':']
# 1 -> 3 ['#']
# 1 -> 4 [$]
# 2 -> 1 [<VAL> ' ']
# 2 -> 3 [<VAL> '#']
# 2 -> 4 [<VAL> $]
# 3 -> 4 [$]
# 3 -> 4 [<COMMENT> $]
# 4: end state

function _update_from_line!(x, y, l::String)
  pos = 1
  st = 0
  col = convert(Int, 0)
  val = convert(Float64, 0)
  pos = _skip_space(l, pos)
  while st != 4
    if st == 0
      if done(l, pos)
        st = 4
      elseif l[pos] == '#'
        pos += 1
        st = 3
      else
        nextpos = search(l, [' ', '#'], pos)
        if 0 < nextpos
          push!(y, parsefloat(Float64, l[pos:nextpos]))
          pos = nextpos+1
          st = isspace(l[nextpos]) ? 1 : 3
        else
          target = l[pos:end]
          pos = endof(l)
          st = 4
        end
      end
    elseif st == 1
      pos = _skip_space(l, pos)
      if done(l, pos)
        st = 4
      elseif l[pos] == '#'
        st = 3
      else
        nextpos = search(l, ':', pos)
        if 0 < nextpos
          c = parseint(Int, l[pos:nextpos-1])
          push!(x.rowval, c)
          x.m = max(x.m, c)
          st = 2
          pos = nextpos+1
        else
          throw(ParseError("Expected ':' [$l]"))
        end
      end
    elseif st == 2
      nextpos = search(l, [' ', '#'], pos)
      if 0 < nextpos
        push!(x.nzval, parsefloat(Float64, l[pos:nextpos-1]))
        st = isspace(l[nextpos]) ? 1 : 3
        pos = nextpos+1
      else
        push!(x.nzval, parsefloat(Float64, l[pos:end]))
        st = 4
        pos = endof(l)
      end
    elseif st == 3
      st = 4
      pos = endof(l)
    end
  end
  push!(x.colptr, endof(x.rowval)+1)
  x.n += 1
end

function _skip_space(s, pos)
  while !done(s, pos)
    ch, pos1 = next(s, pos)
    if isspace(ch)
      pos = pos1
    else
      return pos
    end
  end
  pos
end
