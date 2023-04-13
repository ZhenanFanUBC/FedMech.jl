using FedMech
using LinearAlgebra
using SparseArrays
using Flux
import Random: seed!, randperm
using Printf
function main(λ, p, withMech, withFed; numClients=5, numClassPerClient=5, dataPath="data/covtype", verbose=true)
    seed!(9999)
    # load data
    X, Y = read_libsvm(dataPath)
    # number of classes
    numClass = maximum(Y) # 7 for covtype 
    # println("$dataPath ncls $numClass size of X $(size(X)) #cls $numClass") 
    # separate data to clients in non-iid fashion
    Xsplit, Ysplit = splitDataByClass(X, Y, numClients, numClass, numClassPerClient)

    # initialize clients
    clients = Vector{Union{Client,ClientImg}}(undef, numClients)
    for i = 1:numClients
        clients[i] = Client(i, Xsplit[i], Ysplit[i], numClass, λ, p, withMech; withAdap=withFed == 3)
    end

    # initialize server
    if withFed != 0
        τ = 5
        server = Server(clients, τ)
        # @printf("Start federated learning process:\n")
        training!(server, 10; with_proximal_term=withFed == 2, numEpoches=5)
    else
        for i = 1:numClients
            update!(clients[i], numEpoches=5)
        end
    end

    accs = []
    povs = []
    show_name(withMech, withFed)
    for i = 1:numClients
        acc, pov = performance(clients[i]; verbose=verbose)
        push!(accs, acc)
        push!(povs, pov)
    end
    println("TA $(mean_plusminus_std(accs)) POV $(mean_plusminus_std(povs)) ")

    if withMech == false && withFed == 0
        @printf("performance for P-KM\n")
        accs = []
        povs = []
        for i = 1:numClients
            acc, pov = performance(clients[i]; use_g=true, verbose=verbose)
            push!(accs, acc)
            push!(povs, pov)
        end
        println("TA $(mean_plusminus_std(accs)) POV $(mean_plusminus_std(povs)) ")
    end

    if withMech == false && withFed != 0
        # recompose f with λ1, i.e., inject knowledge  
        λ1 = 0.0 # only use R-KM, not use P-KM 
        println("\n --- recompose f, lambda = $(λ1) --- below: \n")
        accs = []
        povs = []
        for i = 1:numClients
            c = clients[i]
            c.f = x -> (1 - λ1) * NNlib.softmax(c.W(x) + dropgrad(c.h(x))) + λ1 * dropgrad(c.g(x))
            acc, pov = performance(c; verbose=verbose)
            push!(accs, acc)
            push!(povs, pov)
        end
        println("TA $(mean_plusminus_std(accs)) POV $(mean_plusminus_std(povs)) ")
    end
end
