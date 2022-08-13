module FedMech

using LinearAlgebra
using Flux
using Zygote
using Printf
using Statistics
using StatsBase
using Random
using SparseArrays


export load_data, read_libsvm, train_test_split, splitDataByClass, split_data, splitDataByClassImg
export label_transformation
export mechanism1, mechanism2
export LeNet5small, LeNet5
export buildPredModel, buildRangeModel
export buildPredModelImg, buildRangeModelImg
export getMetrics, r2_score
export Client, ClientImg, update!, performance
export Server, select!, sendModel!, aggregate!, training!


include("utils.jl")
include("client.jl")
include("server.jl")

end