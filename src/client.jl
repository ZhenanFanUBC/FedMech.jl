# client for general classification dataset
mutable struct Client{T1<:Int64, 
                      T2<:Float64, 
                      T3<:Vector{T1}, 
                      T4<:SparseMatrixCSC{T2, T1},  
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
                    X::SparseMatrixCSC{Float64, Int64},
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
        h = buildRangeModel(hcat(Xtrain,Xtest), vcat(Ytrain,Ytest), numClass, 1, g)
        # model
        numFeature = size(Xtrain, 1)
        dim1 = 128
        dim2 = 128
        W = Chain(  Dense(numFeature, dim1, relu), 
                Dense(dim1, dim2, relu),
                Dense(dim2, numClass) )
        # use only subset of training data
        num = size(Xtrain, 2)
        numTrain = Int(floor(p*num))
        perm = Random.randperm(num)
        Idx = perm[1:numTrain]
        # personalized model
        if withMech
            f = x -> (1-λ)*NNlib.softmax( W(x) + h(x) ) + λ*g(x)
        else
            f = x -> NNlib.softmax( W(x) )
        end
        new{Int64, Float64, Vector{Int64}, SparseMatrixCSC{Float64, Int64}, Flux.OneHotArray, Flux.Chain, Function}(id, Xtrain[:,Idx], Ytrain[Idx], YtrainHot[:,Idx], Xtest, Ytest, W, g, h, f)
    end
end


# client for image classification dataset
mutable struct ClientImg{T1<:Int64, 
                         T2<:Float64, 
                         T3<:Vector{T1}, 
                         T4<:Array{Float64, 4},  
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
                       Xtrain::Array{Float64, 4},
                       Ytrain::Vector{Int64},
                       Xtest::Array{Float64, 4},
                       Ytest::Vector{Int64},
                       numClass::Int64,
                       λ::Float64,
                       p::Float64,
                       withMech::Bool)           
        # label transformation
        YtrainHot = Flux.onehotbatch(Ytrain, 0:9)
        # mechanism models
        g = buildPredModelImg(Xtrain, YtrainHot)
        h = buildRangeModelImg(cat(Xtrain,Xtest,dims=4), vcat(Ytrain,Ytest), numClass, 2, g)
        # model
        W = LeNet5()
        # use only subset of training data
        num = size(Xtrain, 4)
        numTrain = Int(floor(p*num))
        perm = Random.randperm(num)
        Idx = perm[1:numTrain]
        # model with mechanisms
        if withMech
            f = x -> (1-λ)*NNlib.softmax( W(x) + h(x) ) + λ*g(x)
        else
            f = x -> NNlib.softmax( W(x) )
        end
        new{Int64, Float64, Vector{Int64}, Array{Float64, 4}, Flux.OneHotArray, Flux.Chain, Function}(id, Xtrain[:,:,:,Idx], Ytrain[Idx], YtrainHot[:,Idx], Xtest, Ytest, W, g, h, f)
    end
end

function update!(c::Union{Client,ClientImg}, numEpoches::Int64=5, callback!::Function=identity)
    data = Flux.Data.DataLoader( (c.Xtrain, c.YtrainHot), batchsize=5, shuffle=true)
    loss(x, y) = Flux.crossentropy( c.f(x) , y )
    opt = ADAM()
    for t = 1:numEpoches
        Flux.train!(loss, Flux.params(c.W), data, opt)
        callback!( (c, t) )
    end
    lss = loss(c.Xtrain, c.YtrainHot)
    # @printf "client: %d, loss: %.2f\n" c.id lss
    return lss
end


function performance(c::Client; use_g::Bool=false)
    numPred = 0
    numVio = 0
    numTest = size(c.Xtest, 2)
    for i = 1:numTest
        x = c.Xtest[:,i]
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
    @printf "client: %d, test accuracy: %.2f, percentage of violation: %.2f\n" c.id numPred/numTest numVio/numTest
end

function performance(c::ClientImg; use_g::Bool=false)
    numPred = 0
    numVio = 0
    numTest = size(c.Xtest, 4)
    for i = 1:numTest
        x = c.Xtest[:,:,:,i:i]
        if use_g
            pred = argmax(c.g(x))[1]-1
        else
            pred = argmax(c.f(x))[1]-1
        end
        if pred == c.Ytest[i]
            numPred += 1
        end
        range = c.h(x)
        if range[pred+1] != 0.0
            numVio += 1
        end
    end
    @printf "client: %d, test accuracy: %.2f, percentage of violation: %.2f\n" c.id numPred/numTest numVio/numTest
end

