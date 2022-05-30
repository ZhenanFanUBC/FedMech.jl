module FedMech

using LinearAlgebra
using Flux
using Zygote
using Printf
using Statistics
using StatsBase
using Random
using SparseArrays
# Random.seed!(1234)
foreach(i -> Random.seed!(Random.default_rng(i), 1234), 1:Threads.nthreads())

export load_data, read_libsvm, train_test_split
export label_transformation
export mechanism1, mechanism2
export getMetrics, r2_score
export Client, update!, performance, get_predict
export Server, select!, sendModel!, aggregate!, training!

include("utils.jl")
include("client.jl")
include("server.jl")

end