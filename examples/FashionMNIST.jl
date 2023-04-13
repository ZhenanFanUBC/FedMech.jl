using FedMech
using LinearAlgebra
using MLDatasets
import Random: seed!, randperm
using Printf
import Zygote: dropgrad
using CUDA, Flux
import StatsBase: sample

function main(λ, p, withMech, withFed; numClients=5, verbose=true)

    seed!(9999)
    # io=open("/tmp/out.log","w")
    # logger = timestamp_logger( SimpleLogger(io, Logging.Info) ) 
    # global_logger(logger)
    # @info("start ") 

    # load data
    Xtrain, Ytrain = MLDatasets.FashionMNIST(:train)[:]
    Xtrain = reshape(Xtrain, 28, 28, 1, :)
    Xtrain = convert(Array{Float32,4}, Xtrain)
    Ytrain = convert(Vector{Int64}, Ytrain)
    Xtest, Ytest = MLDatasets.FashionMNIST(:test)[:]
    Xtest = reshape(Xtest, 28, 28, 1, :)
    Xtest = convert(Array{Float32,4}, Xtest)
    Ytest = convert(Vector{Int64}, Ytest)
    # number of classes
    numClass = 10
    # separate data to clients in non-iid fashion
    XtrainSplit, YtrainSplit, XtestSplit, YtestSplit = splitDataByClassImg(Xtrain, Ytrain, Xtest, Ytest, numClients, numClass, 6)

    # initialize clients
    clients = Vector{Union{Client,ClientImg}}(undef, numClients)
    for i = 1:numClients
        clients[i] = ClientImg(i, XtrainSplit[i], YtrainSplit[i], XtestSplit[i], YtestSplit[i], numClass, λ, p, withMech; withAdap=withFed == 3)
    end
    # @printf("prepared clients \n ") 

    # initialize server
    if withFed != 0
        τ = 5
        server = Server(clients, τ)
        # @printf("Start federated learning process:\n")
        training!(server, 200; with_proximal_term=withFed == 2, numEpoches=5)
    else
        for i = 1:numClients
            update!(clients[i]; numEpoches=10)
        end
    end


    # check performance
    show_name(withMech, withFed)
    accs = []
    povs = []
    for i = 1:numClients
        acc, pov = performance(clients[i]; verbose=verbose)
        push!(accs, acc)
        push!(povs, pov)
    end
    println("TA ", mean_plusminus_std(accs), " POV ", mean_plusminus_std(povs))

    if withMech == false && withFed == 0
        @printf("performance for P-KM\n")
        accs = []
        povs = []
        for i = 1:numClients
            acc, pov = performance(clients[i]; use_g=true, verbose=false)
            push!(accs, acc)
            push!(povs, pov)
        end
        println("TA ", mean_plusminus_std(accs), " POV ", mean_plusminus_std(povs))
    end

    # for i = 1:numClients
    #     c=clients[i]; 
    #     for (func,nm) in zip([c.f, c.g], ["final", "predictive"])
    #         X,Y = c.Xtest, c.Ytest  
    #         @printf("client %d, %s, test: ", i, nm); performance_2(func, X, Y) 
    #     end
    # end

    # @info("end ")
    # flush(io)
    # close(io)
end













































