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
Xtest, Ytest = MLDatasets.FashionMNIST(:test)[:]
Xtest = reshape(Xtest, 28, 28, 1, :)
Xtest = convert(Array{Float64, 4}, Xtest)
# number of clients
numClients = 5
# number of classes
numClass = 10
# separate data to clients in non-iid fashion
XtrainSplit, YtrainSplit, XtestSplit, YtestSplit = splitDataByClassImg(Xtrain, Ytrain, Xtest, Ytest, numClients, numClass, 6)


# initialize clients
clients = Vector{Union{Client, ClientBase, ClientImg}}(undef, numClients)
for i = 1:numClients
    clients[i] = ClientImg(i, XtrainSplit[i], YtrainSplit[i], XtestSplit[i], YtestSplit[i], numClass, 0.7,
    0.1) 
end

# initialize server
server = Server(clients, 3)
@printf("Start federated learning process:\n")
training!(server, 10)

# check performance
for i = 1:numClients
    performance(clients[i])
end














































