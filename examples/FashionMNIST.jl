using FedMech
using LinearAlgebra
using MLDatasets
import Random: seed!, randperm
using Printf
seed!(1234)

# load data
Xtrain, Ytrain = MLDatasets.FashionMNIST(:train)[:]
Xtrain = reshape(Xtrain, 28, 28, 1, :)
Xtrain = convert(Array{Float64, 4}, Xtrain)
Ytrain = convert(Vector{Int64},Ytrain)
Xtest, Ytest = MLDatasets.FashionMNIST(:test)[:]
Xtest = reshape(Xtest, 28, 28, 1, :)
Xtest = convert(Array{Float64, 4}, Xtest)
Ytest = convert(Vector{Int64},Ytest)
# number of clients
numClients = 5
# number of classes
numClass = 10
# separate data to clients in non-iid fashion
XtrainSplit, YtrainSplit, XtestSplit, YtestSplit = splitDataByClassImg(Xtrain, Ytrain, Xtest, Ytest, numClients, numClass, 6)
# hyperparameters
λ = 0.3
p = 0.01
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

# initialize server
if withFed
    τ = 5
    server = Server(clients, τ)
    # @printf("Start federated learning process:\n")
    training!(server, 10)
else
    for i = 1:numClients
        update!(clients[i], 10)
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

@printf("performance for P-KM\n")
for i = 1:numClients
    performance(clients[i]; use_g=true)
end














































