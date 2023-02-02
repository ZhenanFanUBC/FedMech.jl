using FedMech
using LinearAlgebra
using MLDatasets
import Random: seed!, randperm
using Printf
import Zygote: dropgrad
using CUDA, Flux 
import StatsBase: sample

seed!(9999)

# load data
Xtrain, Ytrain = MLDatasets.FashionMNIST(:train)[:]
Xtrain = reshape(Xtrain, 28, 28, 1, :)
Xtrain = convert(Array{Float32, 4}, Xtrain)
Ytrain = convert(Vector{Int64},Ytrain)
Xtest, Ytest = MLDatasets.FashionMNIST(:test)[:]
Xtest = reshape(Xtest, 28, 28, 1, :)
Xtest = convert(Array{Float32, 4}, Xtest)
Ytest = convert(Vector{Int64},Ytest)
# number of clients
numClients = 5 # 5 
# number of classes
numClass = 10
# separate data to clients in non-iid fashion
XtrainSplit, YtrainSplit, XtestSplit, YtestSplit = splitDataByClassImg(Xtrain, Ytrain, Xtest, Ytest, numClients, numClass, 6)
# hyperparameters
λ = 0.1
p = 0.3
@printf("λ = %.2f, p = %.2f\n", λ, p)

# tag for whether using mechanism models
withMech = true
# tag for whether using federated learning
withFed = true

# initialize clients
clients = Vector{Union{Client, ClientImg}}(undef, numClients)
for i = 1:numClients
    clients[i] = ClientImg(i, XtrainSplit[i], YtrainSplit[i], XtestSplit[i], YtestSplit[i], numClass, λ, p, withMech) 
end
@printf("prepared clients \n ") 

# initialize server
if withFed
    τ = 5
    server = Server(clients, τ)
    # @printf("Start federated learning process:\n")
    training!(server, 200)  
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
    performance(clients[i])
end

@printf("\n")

for i = 1:numClients
    c=clients[i]; 
    for (func,nm) in zip([c.f, c.g], ["final", "predictive"])
        X,Y = c.Xtest, c.Ytest  
        @printf("client %d, %s, test: ", i, nm); performance_2(func, X,Y) 
        # X,Y = c.Xtrain, c.Ytrain  
        # @printf("client %d, %s, train: ", i, nm); performance_2(func, X,Y) 
    end
end













































