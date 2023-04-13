module FedMech

using LinearAlgebra
using Flux
using Printf
using Statistics
using StatsBase
using Random
using SparseArrays
using CUDA
using Revise 
using Dates#, Logging LoggingExtras, 
using MLDatasets
import Random: seed!, randperm
import StatsBase: sample
const date_format = "mm-dd HH:MM:SS"
timestamp_logger(logger) = TransformerLogger(logger) do log
  merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
end

export load_data, read_libsvm, train_test_split, splitDataByClass, split_data, splitDataByClassImg
export label_transformation
export mechanism1, mechanism2
export LeNet5small, LeNet5
export buildPredModel, buildRangeModel
export buildPredModelImg, buildRangeModelImg
export getMetrics, r2_score
export Client, ClientImg, update!, performance, performance_2 
export Server, select!, sendModel!, aggregate!, training!

CUDA.allowscalar(true) 
device=cpu 
include("utils.jl")
include("my_client.jl")
include("server.jl")

end