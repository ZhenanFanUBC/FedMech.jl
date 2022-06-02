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
    f::T7                   # model with mechanisms
    function Client(id::Int64,
                    X::SparseMatrixCSC{Float64, Int64},
                    Y::Vector{Int64},
                    numClass::Int64,
                    λ::Float64)
        # split train and test
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(X, Y, 0.01)            
        # label transformation
        YtrainHot = Flux.onehotbatch(Ytrain, 1:numClass)
        # mechanism models
        g = buildPredModel(Xtrain, YtrainHot, numClass)
        h = buildRangeModel(hcat(Xtrain,Xtest), vcat(Ytrain,Ytest), numClass::Int64)
        # model
        numFeature = size(Xtrain, 1)
        dim1 = 128
        dim2 = 128
        W = Chain(  Dense(numFeature, dim1, relu), 
                Dense(dim1, dim2, relu),
                Dense(dim2, numClass) )
        # model with mechanisms
        f = x -> λ*NNlib.softmax( W(x) + h(x) ) + (1-λ)*g(x)
        new{Int64, Float64, Vector{Int64}, SparseMatrixCSC{Float64, Int64}, Flux.OneHotArray, Flux.Chain, Function}(id, Xtrain, Ytrain, YtrainHot, Xtest, Ytest, W, f)
    end
end

function update!(c::Client, numEpoches::Int64=5, callback!::Function=identity)
    data = Flux.Data.DataLoader( (c.Xtrain, c.YtrainHot), batchsize=1, shuffle=true)
    loss(x, y) = Flux.crossentropy( c.f(x) , y )
    opt = ADAM()
    for t = 1:numEpoches
        Flux.train!(loss, Flux.params(c.W), data, opt)
        callback!( (c, t) )
    end
    lss = loss(c.Xtrain, c.YtrainHot)
    @printf "client: %d, loss: %.2f\n" c.id lss
    return lss
end


function performance(c::Client)
    count = 0
    numTest = size(c.Xtest, 2)
    for i = 1:numTest
        x = c.Xtest[:,i]
        pred = argmax(c.f(x))
        if pred == c.Ytest[i]
            count += 1
        end
    end
    @printf "client: %d, test accuracy: %.2f\n" c.id count/numTest
end

