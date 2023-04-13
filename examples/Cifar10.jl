
using FedMech
using LinearAlgebra
using MLDatasets
import Random: seed!, randperm
using Printf
using CUDA, Flux
import StatsBase: sample
import Zygote: dropgrad
using LoggingExtras, Dates
const date_format = "mm-dd HH:MM:SS"
using ArgParse
device = gpu

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--batchsize"
        arg_type = Int64
        default = 32
        "--datarat"
        arg_type = Float64
        default = 0.01
        "--lambda"
        arg_type = Float64
        default = 0.0
        "--withMech"
        arg_type = Int64
        default = 1
        "--withFed"
        arg_type = Int64
        default = 0
    end
    return parse_args(s)
end

function main(λ, p, withMech, withFed; numClients=5, verbose=true)
    "withFed=0 for ML, withFed=1 for FL, withFed=2 for proximal reg, withFed=3 for adaptive "
    withFed = Int64(withFed)
    seed!(9999)
    tic = now()
    println("start: $(Dates.format(now(), date_format))")

    Xtrain, Ytrain = MLDatasets.CIFAR10(:train)[:]
    Xtrain = convert(Array{Float32,4}, Xtrain)
    Xtest, Ytest = MLDatasets.CIFAR10(:test)[:]
    Xtest = convert(Array{Float32,4}, Xtest)

    numClass = 10
    # separate data to clients in non-iid fashion
    XtrainSplit, YtrainSplit, XtestSplit, YtestSplit = splitDataByClassImg(Xtrain, Ytrain, Xtest, Ytest, numClients, numClass, 6)

    # initialize clients
    clients = Vector{Union{Client,ClientImg}}(undef, numClients)
    for i = 1:numClients
        clients[i] = ClientImg(i, XtrainSplit[i], YtrainSplit[i], XtestSplit[i], YtestSplit[i], numClass, λ, p, withMech; withAdap=withFed == 3)
    end
    @printf("prepared clients\n ")

    # initialize server
    if withFed != 0
        server = Server(clients, numClients)
        @printf("Start federated learning process:\n")
        training!(server, 50; with_proximal_term=withFed == 2, numEpoches=3)
    else
        for i = 1:numClients
            update!(clients[i], numEpoches=50)
        end
    end

    # check performance
    show_name(withMech, withFed)

    @printf("\n --- final f --- below: \n")
    accs = []
    povs = []
    for i = 1:numClients
        c = clients[i]
        acc, pov = performance(c, use_g=false; verbose=verbose)
        push!(accs, acc)
        push!(povs, pov)
        # @printf("dbg: client %d, test: ", i); performance_2(c.f, c.Xtest, c.Ytest)
    end
    println("TA ", mean_plusminus_std(accs), " POV ", mean_plusminus_std(povs))

    @printf("\n --- predictive g --- below: \n")
    accs = []
    povs = []
    for i = 1:numClients
        acc, pov = performance(clients[i], use_g=true; verbose=verbose)
        push!(accs, acc)
        push!(povs, pov)
    end
    println("TA ", mean_plusminus_std(accs), " POV ", mean_plusminus_std(povs))

    for λ1 in [0.3] # [.0,.1,.2,.3,.4,.5]
        println("\n --- recompose f, lambda = $(λ1) --- below: \n")
        accs = []
        povs = []
        for i = 1:numClients
            c = clients[i]
            c.f = x -> (1 - λ1) * NNlib.softmax(c.W(x) + dropgrad(c.h(x))) + λ1 * dropgrad(c.g(x))
            # @printf("dbg: client %d, test: ", i); performance_2(c.f, c.Xtest, c.Ytest)
            acc, pov = performance(c; verbose=verbose)
            push!(accs, acc)
            push!(povs, pov)
        end
        println("TA ", mean_plusminus_std(accs), " POV ", mean_plusminus_std(povs))
    end

    # for i = 1:numClients
    #     c=clients[i]; 
    #     for (func,nm) in zip([c.f, c.g], ["final", "predictive"])
    #         X,Y = c.Xtest, c.Ytest  
    #         @printf("client %d, %s, test: ", i, nm); performance_2(func, X,Y) 
    #         X,Y = c.Xtrain, c.Ytrain  
    #         @printf("client %d, %s, train: ", i, nm); performance_2(func, X,Y) 
    #     end
    # end

    println("finish: $(Dates.format(now(), date_format))")
    println("elapse: $(now()-tic)")
    return clients
end

# if abspath(PROGRAM_FILE) == @__FILE__
#     parsed_args = parse_commandline()
#     println("Parsed args:")
#     for (arg,val) in parsed_args
#         println("  $arg  =>  $val")
#     end 
#     # hyperparameters
#     λ = parsed_args["lambda"]#0. # 0.3
#     p = parsed_args["datarat"] #0.01 # 0.3# 
#     @printf("hyper: λ = %.2f, p = %.2f\n", λ, p) # @info "hyper: " λ=λ p=p 
#     withMech = Bool(parsed_args["withMech"])# tag for whether using mechanism models
#     withFed = Bool(parsed_args["withFed"]) # tag for whether using federated learning
#     main(λ,p,withMech, withFed)
# end
