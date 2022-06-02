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
numClients = 3
# number of classes
numClass = maximum(Y)
# separate data to clients in non-iid fashion
# Xsplit, Ysplit = splitDataByClass(X, Y, numClients, numClass)
Xsplit, Ysplit = split_data(X, Y, numClients)
# initialize clients
clients = Vector{Client}(undef, numClients)
for i = 1:numClients
    clients[i] = Client(i, Xsplit[i], Ysplit[i], numClass, 0.7)
end
# Initialize server
τ = 3
server = Server(clients, τ)
# Training
@printf("Start federated learning process:\n")
training!(server, 3)
# check performance
for i = 1:numClients
    performace(clients[i])
end












































