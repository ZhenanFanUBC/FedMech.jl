# Coal Mixing Problem
push!(LOAD_PATH, pwd())
using Revise
using FedMech
using LinearAlgebra
using Printf
using Random
foreach(i -> Random.seed!(Random.default_rng(i), 1234), 1:Threads.nthreads())

# Load and split data
numTrain1, Xtrain1, labelTrain1 = load_data("../x_train_m.csv", indexed=false)
numTest1_1, Xtest1_1, labelTest1_1 = load_data("../x_test_m.csv", indexed=false)
numTest1_2, Xtest1_2, labelTest1_2 = load_data("../x_test_g_m.csv", indexed=false)
num2, X2, label2 = load_data("../data_ag.csv", indexed=true)
Xtrain2, labelTrain2, Xtest2, labelTest2 = train_test_split(num2, X2, label2, 0.7)
num3, X3, label3 = load_data("../data_hy.csv", indexed=true)
Xtrain3, labelTrain3, Xtest3, labelTest3 = train_test_split(num3, X3, label3, 0.7)
num4, X4, label4 = load_data("../data_mf.csv", indexed=true)
Xtrain4, labelTrain4, Xtest4, labelTest4 = train_test_split(num4, X4, label4, 0.7)

# Preknowledge
numFeature = size(Xtrain1, 1)
# weight
w = zeros(numFeature)
w[2] = -0.75
w[4] = 0.4
w[5] = 0.4
c1 = 55.0
c2 = 48.48
c3 = 51.42
c4 = 52.77
# lower bound
l = zeros(numFeature); l[2] = 5
# upper bound
u = 100*ones(numFeature)
u[1] = 20; u[2] = 45; u[3] = 5; u[5] = 35; u[16] = 1
# normal vector
d = zeros(numFeature); d[10] = 1; d[11] = 1; d[12] = 1
# minimum and maximum
minVal = round(min( minimum(labelTrain1), minimum(labelTrain2), minimum(labelTrain3), minimum(labelTrain4) ))
maxVal = round(max( maximum(labelTrain1), maximum(labelTrain2), maximum(labelTrain3), maximum(labelTrain4) ))
minVal = convert(Int64, minVal)
maxVal = convert(Int64, maxVal)
# mechanism models
g1 = x -> mechanism1(x, w, c1, minVal, maxVal)
h1 = x -> mechanism2(x, l, u, d, minVal, maxVal)
g2 = x -> mechanism1(x, w, c2, minVal, maxVal)
h2 = x -> mechanism2(x, l, u, d, minVal, maxVal)
g3 = x -> mechanism1(x, w, c3, minVal, maxVal)
h3 = x -> mechanism2(x, l, u, d, minVal, maxVal)
g4 = x -> mechanism1(x, w, c4, minVal, maxVal)
h4 = x -> mechanism2(x, l, u, d, minVal, maxVal)

# Initialize clients
clients = Vector{Client}(undef, 4)
clients[1] = Client(1, Xtrain1, labelTrain1, g1, h1, minVal, maxVal, 0.7, 1e-3)
clients[2] = Client(2, Xtrain2, labelTrain2, g2, h2, minVal, maxVal, 0.7, 1e-3)
clients[3] = Client(3, Xtrain3, labelTrain3, g3, h3, minVal, maxVal, 0.7, 1e-3)
clients[4] = Client(4, Xtrain4, labelTrain4, g4, h4, minVal, maxVal, 0.7, 1e-3)

# training

update!(clients[1], 1090)
update!(clients[2], 1000)
update!(clients[3], 1000)
update!(clients[4], 1000)

# check performance
@printf("Performance of client 1\n")
@printf("Test1\n")
performance(clients[1], Xtest1_1, labelTest1_1, minVal, maxVal)
@printf("Test2\n")
performance(clients[1], Xtest1_2, labelTest1_2, minVal, maxVal)
@printf("Performance of client 2\n")
performance(clients[2], Xtest2, labelTest2, minVal, maxVal)
@printf("Performance of client 3\n")
performance(clients[3], Xtest3, labelTest3, minVal, maxVal)
@printf("Performance of client 4\n")
performance(clients[4], Xtest4, labelTest4, minVal, maxVal)

