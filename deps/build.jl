using BinDeps

@BinDeps.setup

deps = [
  liblinear = library_dependency("liblinear", aliases = ["liblinear"])
]

@osx_only begin
  if Pkg.installed("Homebrew") === nothing
    error("Homebrew package not installed, please run Pkg.add(\"Homebrew\")")
  end
  using Homebrew
  provides(Homebrew.HB, "liblinear", liblinear, os = :Darwin )
end

# System Package Managers

# liblinear1 installs a lot of packages
# provides(AptGet, {
#   "liblinear1" => liblinear
# })

const ver = "1.94"

provides(Sources, {
  URI("http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-$(ver).tar.gz") => liblinear
})

prefix = joinpath(BinDeps.depsdir(liblinear), "usr")
srcdir = joinpath(BinDeps.depsdir(liblinear), "src", "liblinear-$(ver)")

provides(BuildProcess,
  (@build_steps begin
    GetSources(liblinear)
    CreateDirectory(prefix)
    CreateDirectory(joinpath(prefix, "lib"))
    @build_steps begin
      ChangeDirectory(srcdir)
      FileRule(joinpath(prefix, "lib", "liblinear.so"), @build_steps begin
        MakeTargets(["lib"])
        `cp $srcdir/liblinear.so.1 $prefix/lib/liblinear.so`
      end)
    end
  end), liblinear, os = :Unix)

@BinDeps.install [:liblinear => :liblinear]
