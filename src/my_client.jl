# client for general classification dataset
mutable struct Client{T1<:Int64,
    T2<:Float32,
    T3<:Vector{T1},
    T4<:SparseMatrixCSC{T2,T1},
    T5<:Flux.OneHotArray,
    T6<:Flux.Chain,
    T7<:Function}
    id::T1                  # client index
    Xtrain::T4              # training data
    Ytrain::T3              # training label
    YtrainHot::T5           # transformed training label
    Xtest::T4               # test data
    Ytest::T3               # test label
    W::T6                   # model  
    g::T7                   # prediction-type knowledge model
    h::T7                   # range-type knowledge model   
    f::T7                   # personalized model
    function Client(id::Int64,
        X::SparseMatrixCSC{Float32,Int64},
        Y::Vector{Int64},
        numClass::Int64,
        λ::Float64,
        p::Float64,
        withMech::Bool)
        # split train and test
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(X, Y, 0.1)
        # label transformation
        YtrainHot = Flux.onehotbatch(Ytrain, 1:numClass)
        # mechanism models
        g = buildPredModel(Xtrain, YtrainHot, numClass)
        h = buildRangeModel(hcat(Xtrain, Xtest), vcat(Ytrain, Ytest), numClass, 1, g)
        # model
        numFeature = size(Xtrain, 1)
        dim1 = 128
        dim2 = 128
        W = Chain(Dense(numFeature, dim1, relu),
            Dense(dim1, dim2, relu),
            Dense(dim2, numClass))
        # use only subset of training data
        num = size(Xtrain, 2)
        numTrain = Int(floor(p * num))
        perm = Random.randperm(num)
        Idx = perm[1:numTrain]
        # personalized model
        if withMech
            f = x -> (1 - λ) * NNlib.softmax(W(x) + h(x)) + λ * g(x)
        else
            f = x -> NNlib.softmax(W(x))
        end
        new{Int64,Float32,Vector{Int64},SparseMatrixCSC{Float32,Int64},Flux.OneHotArray,Flux.Chain,Function}(id, Xtrain[:, Idx], Ytrain[Idx], YtrainHot[:, Idx], Xtest, Ytest, W, g, h, f)
    end
end

import Zygote: dropgrad
# client for image classification dataset
mutable struct ClientImg{T1<:Int64,
    T2<:Float32,
    T3<:Vector{T1},
    T4<:Array{Float32,4},
    T5<:Flux.OneHotArray,
    T6<:Flux.Chain,
    T7<:Function}
    id::T1                  # client index
    Xtrain::T4              # training data
    Ytrain::T3              # training label
    YtrainHot::T5           # transformed training label
    Xtest::T4               # test data
    Ytest::T3               # test label
    W::T6                   # model                 
    g::T7                   # prediction-type knowledge model
    h::T7                   # range-type knowledge model   
    f::T7                   # personalized model
    function ClientImg(id::Int64,
        Xtrain::Array{Float32,4},
        Ytrain::Vector{Int64},
        Xtest::Array{Float32,4},
        Ytest::Vector{Int64},
        numClass::Int64,
        λ::Float64,
        p::Float64,
        withMech::Bool)
        # label transformation
        YtrainHot = Flux.onehotbatch(Ytrain, 0:9)
        is_cifar = size(Xtrain, 3) == 3
        # mechanism models
        g = buildPredModelImg(Xtrain, YtrainHot)
        h = buildRangeModelImg(cat(Xtrain, Xtest, dims=4), vcat(Ytrain, Ytest), numClass, 2, g)
        # model
        if !is_cifar
            W = LeNet5()
        else
            W = MyModel(large=true)
        end
        W = W |> device
        # use only subset of training data
        num = size(Xtrain, 4)
        numTrain = Int(floor(p * num))
        perm = Random.randperm(num)
        Idx = perm[1:numTrain]
        # model with mechanisms
        if withMech
            f = x -> (1 - λ) * NNlib.softmax(W(x) + dropgrad(h(x))) + λ * dropgrad(g(x))
        else
            f = x -> NNlib.softmax(W(x))
        end
        new{Int64,Float32,Vector{Int64},Array{Float32,4},Flux.OneHotArray,Flux.Chain,Function}(id, Xtrain[:, :, :, Idx], Ytrain[Idx], YtrainHot[:, Idx], Xtest, Ytest, W, g, h, f)
    end
end

using MLUtils: mapobs

function update!(c::Union{Client,ClientImg}; numEpoches::Int64=5, eta=0.1) # todo caller need to be changed accordingly 
    data = Flux.DataLoader(
        mapobs(device, (c.Xtrain, c.YtrainHot)), batchsize=32, shuffle=true)
    loss(x, y) = Flux.crossentropy(c.f(x), y)
    opt = Flux.Optimise.Optimiser(WeightDecay(5.0f-4), Adam())
    # opt = Flux.Optimise.Optimiser(WeightDecay(5f-4), Momentum(eta, 0.9)) 
    for t = 1:numEpoches
        Flux.train!(loss, Flux.params(c.W), data, opt)
        # callback!( (c, t) )
    end
    lss = -1.0
    # lss = loss(c.Xtrain |> device, c.YtrainHot |> device ) # todo batchlize it 
    # @printf "client: %d, lr: %.2f\n" c.id opt[2].eta 
    return lss
end


function performance(c::Client; use_g::Bool=false)
    numPred = 0
    numVio = 0
    numTest = size(c.Xtest, 2)
    for i = 1:numTest
        x = c.Xtest[:, i]
        if use_g
            pred = argmax(c.g(x))
        else
            pred = argmax(c.f(x))
        end
        if pred == c.Ytest[i]
            numPred += 1
        end
        range = c.h(x)
        if range[pred] != 0.0
            numVio += 1
        end
    end
    @printf "client: %d, test accuracy: %.2f, percentage of violation: %.2f\n" c.id numPred / numTest numVio / numTest
end

import StatsBase: rmsd
function performance(c::Client, lookup::Dict{Any,Any}, mpbk::Any; use_g::Bool=false)
    preds=Float32[] 
    gts=Float32[]
    numVio = 0
    numTest = size(c.Xtest, 2)
    for i = 1:numTest
        x = c.Xtest[:, i]
        if use_g
            pred = argmax(c.g(x))
        else
            pred = argmax(c.f(x))
        end
        push!(preds, mpbk(pred)) 
        push!(gts, lookup[x]) 
        range = c.h(x)
        if range[pred] != 0.0
            numVio += 1
        end
    end
    @printf "client: %d, test rmse: %.2f, percentage of violation: %.2f\n" c.id rmsd(preds, gts) numVio / numTest
end

function performance(c::ClientImg; use_g::Bool=false, eval_on_test::Bool=true)
    numPred = 0
    numVio = 0
    if eval_on_test
        X, Y = c.Xtest, c.Ytest
    else
        X, Y = c.Xtrain, c.Ytrain
    end
    numTest = size(X, 4)
    Flux.testmode!(c.W)
    for i = 1:numTest
        x = X[:, :, :, i:i] |> device
        if use_g
            pred = argmax(c.g(x) |> cpu)[1] - 1
        else
            pred = argmax(c.f(x) |> cpu)[1] - 1
        end
        if pred == Y[i]
            numPred += 1
        end
        rg = c.h(x)
        if rg[pred+1] != 0.0
            numVio += 1
            # @printf "i %d\n" i 
            # break  
        end
    end
    @printf "client: %d, test accuracy: %.2f, percentage of violation: %.2f\n" c.id numPred / numTest numVio / numTest
end

function performance_2(func, X, Y)
    numPred = 0
    numTest = size(X, 4)
    for i = 1:numTest
        x = X[:, :, :, i:i] |> device
        pred = argmax(func(x) |> cpu)[1] - 1
        if pred == Y[i]
            numPred += 1
        end
    end
    @printf "acc: %.2f \n" numPred / numTest
end
