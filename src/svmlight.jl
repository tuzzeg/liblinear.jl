function read_svmlight(io::IO; features=0)
  # Build transposed sparse matrix
  x = SparseMatrixCSC(features, 0, Int[], Int[], Float64[])
  y = Float64[]
  push!(x.colptr, 1)
  for l in eachline(io)
    l = chomp(l)
    _update_from_line!(x, y, l; features=features)
  end
  x', y
end

# States
# 0 -> 0 [' ']
# 0 -> 1 [<TARGET> ' ']
# 0 -> 4 [<TARGET> '#']
# 0 -> 4 ['#']
# 0 -> 5 [<TARGET> $]
# 0 -> 5 [$]
# 1 -> 2|3 [<COL> ':'] # 3 if features<COL 
# 1 -> 4 ['#']
# 1 -> 5 [$]
# 2|3 -> 1 [<VAL> ' ']
# 2|3 -> 4 [<VAL> '#']
# 2|3 -> 5 [<VAL> $]
# 4 -> 5 [$]
# 4 -> 5 [<COMMENT> $]
# 5: end state

function _update_from_line!(x, y, l::String; features=0)
  pos = 1
  st = 0
  col = convert(Int, 0)
  val = convert(Float64, 0)
  pos = _skip_space(l, pos)
  while st != 5
    if st == 0
      if done(l, pos)
        st = 5
      elseif l[pos] == '#'
        pos += 1
        st = 4
      else
        nextpos = search(l, [' ', '#'], pos)
        if 0 < nextpos
          push!(y, parsefloat(Float64, l[pos:nextpos]))
          pos = nextpos+1
          st = isspace(l[nextpos]) ? 1 : 4
        else
          target = l[pos:end]
          pos = endof(l)
          st = 5
        end
        if 4 <= st
          _finished_row(x)
        end
      end
    elseif st == 1
      pos = _skip_space(l, pos)
      if done(l, pos)
        st = 5
      elseif l[pos] == '#'
        st = 4
      else
        nextpos = search(l, ':', pos)
        if 0 < nextpos
          c = parseint(Int, l[pos:nextpos-1])
          if features==0 || c <= features
            push!(x.rowval, c)
            x.m = max(x.m, c)
            st = 2
          else
            st = 3
          end
          pos = nextpos+1
        else
          throw(ParseError("Expected ':' [$l]"))
        end
      end
      if 4 <= st
        _finished_row(x)
      end
    elseif st == 2 || st == 3
      nextpos = search(l, [' ', '#'], pos)
      if 0 < nextpos
        if st == 2
          push!(x.nzval, parsefloat(Float64, l[pos:nextpos-1]))
        end
        st = isspace(l[nextpos]) ? 1 : 4
        pos = nextpos+1
      else
        if st == 2
          push!(x.nzval, parsefloat(Float64, l[pos:end]))
        end
        st = 5
        pos = endof(l)
      end
      if 4 <= st
        _finished_row(x)
      end
    elseif st == 4
      st = 5
      pos = endof(l)
    end
  end
end

function _finished_row(x)
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
