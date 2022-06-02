using LinearAlgebra
using SparseArrays
using Flux
using Random
using Printf
Random.seed!(1234)

######################################################################
######################## load data ###################################
######################################################################
function read_libsvm(filename::String)
    numLine = 0
    nnz = 0
    open(filename, "r") do f
        while !eof(f)
            line = readline(f)
            info = split(line, " ")
            numLine += 1
            nnz += ( length(info)-1 )
            if line[end] == ' '
                nnz -= 1
            end
        end
    end
    @printf("number of lines: %i\n", numLine)
    n = numLine
    m = 0
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    y = zeros(Int64, n)
    numLine = 0
    cc = 1
    open(filename, "r") do f
        while !eof(f)
            numLine += 1
            line = readline(f)
            info = split(line, " ")
            value = parse(Int64, info[1] )
            y[numLine] = value
            ll = length(info)
            if line[end] == ' '
                ll -= 1
            end
            for i = 2:ll
                idx, value = split(info[i], ":")
                idx = parse(Int, idx)
                value = parse(Float64, value)
                I[cc] = numLine
                J[cc] = idx
                V[cc] = value
                cc += 1
                m = max(m, idx)
            end
        end
    end
    return sparse( J, I, V, m, n ), y
end
Xtrain, Ytrain = read_libsvm("data/adults/a8a")
numTrain = size(Xtrain, 2)
Xtest, Ytest = read_libsvm("data/adults/a8a.t")
numTest = size(Xtest, 2)

######################################################################
###### build mechanism model g with limited features #################
######################################################################

# randomly select 40 features
numFeatures = size(Xtrain, 1)
selectedFeatures = randperm(numFeatures)[1:40]
Xtrain1 = Xtrain[selectedFeatures,:]
Ytrain1 = Flux.onehotbatch(Ytrain, [-1,1])
Xtest1 = Xtest[selectedFeatures,:]
# train a logistic regression model
model = Chain(Dense(40, 2), softmax)
loss(x, y) = Flux.crossentropy( model(x) , y )
data = Flux.Data.DataLoader( (Xtrain1, Ytrain1), 
                             batchsize=25, 
                             shuffle=true )
opt = ADAM()
@printf "start training model g ... "
for t = 1:5
    Flux.train!(loss, Flux.params(model), data, opt)
end
@printf "done!\n"
# check performance
function accuracy(m, X::SparseMatrixCSC, Y::Vector)
    count = 0
    num = size(X, 2)
    for i = 1:num
        x = X[:,i]
        pred = sign([-1,1]'*m(x))
        if pred == Y[i]
            count += 1
        end
    end
    return count/num
end
@printf "train accuracy: %.2f\n" accuracy(model, Xtrain1, Ytrain)
@printf "test accuracy: %.2f\n" accuracy(model, Xtest1, Ytest)
# build g: ℜ⁴⁰ → [1,2]
g(x) = [-1,1]'*model(x)

######################################################################
## build constraints  ################################################
## fˣ(w) ≤ aᵀx + b  ∀ x ∈ X := {l ≤ x ≤ u}  ##########################
## c₁ ≤ fˣ(w) ≤ c₂  ∀ x  #############################################
######################################################################

# randomly generate l and u
lower = minimum(Xtrain, dims=2)
upper = maximum(Xtrain, dims=2)
l = zeros(numFeatures)
u = zeros(numFeatures)
for i = 1:numFeatures
    li = lower[i]
    ui = upper[i]
    gap = ui - li
    v1 = rand()*gap + li
    v2 = rand()*gap + li
    l[i] = min(v1, v2)
    u[i] = max(v1, v2)
end
# randomly generate vector a (better way?)
a = randn(numFeatures) 
# find b
function find_bias( X::SparseMatrixCSC, 
                    Y::Vector, 
                    a::Vector, 
                    l::Vector, 
                    u::Vector)
    b = -1e8
    num = size(X, 2)
    for i = 1:num
        x = X[:,i]
        if l ≤ x ≤ u
            y = Y[i]
            b = max(b, y - a'x)
        end
    end
    return b
end
b = find_bias(hcat(Xtrain,Xtest), vcat(Ytrain,Ytest), a, l, u)
# choices for c₁ and c₂ are trival
c₁ = -1
c₂ = 1


