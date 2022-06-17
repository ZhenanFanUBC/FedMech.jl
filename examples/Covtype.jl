using FedMech
using LinearAlgebra
using SparseArrays
using Flux
import Random: seed!, randperm
using Printf
seed!(1234)

# load data
X, Y = read_libsvm("data/covtype/covtype")
# number of clients
numClients = 5
# number of classes
numClass = maximum(Y)
# separate data to clients in non-iid fashion
Xsplit, Ysplit = splitDataByClass(X, Y, numClients, numClass, 5)
# Xsplit, Ysplit = split_data(X, Y, numClients)

# tag for whether using mechanism models
withMech = false
# tag for whether using federated learning
withFed = false

# initialize clients
clients = Vector{Union{Client, ClientBase}}(undef, numClients)
if withMech
    for i = 1:numClients
        clients[i] = Client(i, Xsplit[i], Ysplit[i], numClass, 0.7, 0.01)
    end
else
    for i = 1:numClients
        clients[i] = ClientBase(i, Xsplit[i], Ysplit[i], numClass, 0.01)
    end
end

# initialize server
if withFed
    τ = 3
    server = Server(clients, τ)
    @printf("Start federated learning process:\n")
    training!(server, 10)
else
    for i = 1:numClients
        update!(clients[i], 5)
    end
end

# check performance
for i = 1:numClients
    performance(clients[i])
end














































