using FedMech
using LinearAlgebra
using SparseArrays
using Flux
import Random: seed!, randperm
using Printf
seed!(9999)

# load data
X, Y = read_libsvm("data/covtype/covtype")
# number of clients
numClients = 5
# number of classes
numClass = maximum(Y)
# separate data to clients in non-iid fashion
Xsplit, Ysplit = splitDataByClass(X, Y, numClients, numClass, 5)
# hyperparameters
λ = 0.3
p = 0.01
@printf("λ = %.2f, p = %.3f\n", λ, p)

# tag for whether using mechanism models
withMech = true
# tag for whether using federated learning
withFed = true

# initialize clients
clients = Vector{Union{Client, ClientImg}}(undef, numClients)
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

# @printf("performance for P-KM\n")
# for i = 1:numClients
#     performance(clients[i]; use_g=true)
# end













































