#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
Pkg.instantiate()

include("make.jl")

using LiveServer
serve(dir=joinpath(@__DIR__, "build"))
