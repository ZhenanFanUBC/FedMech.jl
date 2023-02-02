
using FedMech
using LinearAlgebra
using SparseArrays
using Flux
import Random: seed!, randperm
using Printf
seed!(9999)

function discretize(Ycont, nbin=19)
    mx, mn = maximum(Ycont), minimum(Ycont)
    bin_size = (mx - mn) / nbin
    Y = Int64[]
    for y in Ycont
        cls = Int64(floor((y - mn) / bin_size)) + 1
        push!(Y, cls)
    end
    return Y, xx -> (xx - 1) * bin_size + mn
end
function main(λ=0.3, p=0.3, withMech=true, withFed=true)
    # load data
    X, Ycont = read_libsvm("data/YearPredictionMSD-proc", "regression")
    # X, Ycont = read_libsvm("data/housing_scale.txt", "regression") 
    lookup = Dict()
    for i in 1:length(Ycont)
        lookup[X[:, i]] = Ycont[i]
    end
    Y, mpbk = discretize(Ycont)
    println("Size X: ", size(X))

    # number of clients
    numClients = 5
    # number of classes
    numClass = maximum(Y)
    # separate data to clients in non-iid fashion
    Xsplit, Ysplit = splitDataByClass(X, Y, numClients, numClass, 11)
    # hyperparameters
    @printf("λ = %.2f, p = %.3f\n", λ, p)

    # initialize clients
    clients = Vector{Union{Client,ClientImg}}(undef, numClients)
    for i = 1:numClients
        clients[i] = Client(i, Xsplit[i], Ysplit[i], numClass, λ, p, withMech)
    end

    # initialize server
    if withFed
        τ = 5
        server = Server(clients, τ)
        # @printf("Start federated learning process:\n")
        training!(server, 10)
    else
        for i = 1:numClients
            update!(clients[i], numEpoches=10)
        end
    end

    # check performance
    if withMech && withFed
        @printf("performance for FLwKM\n")
    elseif withMech && !withFed
        @printf("performance for MLwKM\n")
    elseif !withMech && withFed
        @printf("performance for FL\n")
    else
        @printf("performance for ML\n")
    end

    for i = 1:numClients
        performance(clients[i], lookup, mpbk)
    end

    @printf("performance for P-KM\n")
    for i = 1:numClients
        performance(clients[i], lookup, mpbk; use_g=true)
    end
    return clients
end
