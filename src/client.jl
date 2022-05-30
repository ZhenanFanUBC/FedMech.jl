mutable struct Client{T1<:Int64, 
                      T2<:Float64, 
                      T3<:Vector{T2}, 
                      T4<:Matrix{T2},  
                      T5<:Flux.OneHotArray, 
                      T6<:Flux.Chain, 
                      T7<:Function}
    id::T1                  # client index
    Xtrain::T4              # training data
    labelTrain::T3          # training label
    Ytrain::T5              # transformed training label
    W::T6                   # model                 
    f::T7                   # model with mechanisms
    μ::T2                   # hyperparameter for L2 regularization
    function Client(id::Int64,
                    Xtrain::Matrix{Float64},
                    labelTrain::Vector{Float64},
                    g::Function,
                    h::Function,
                    minVal::Int64,
                    maxVal::Int64,
                    λ::Float64,
                    μ::Float64)
        numClass = maxVal - minVal + 1
        numFeature = size(Xtrain, 1)
        Ytrain = label_transformation(labelTrain, minVal, maxVal)
        # model
        dim1 = 128
        dim2 = 128
        W = Chain(  Dense(numFeature, dim1, relu), 
                    Dense(dim1, dim2, relu),
                    Dense(dim2, numClass) )
        # model with mechanisms
        f = x -> λ*NNlib.softmax( W(x) + h(x) ) + (1-λ)*g(x)
        new{Int64, Float64, Vector{Float64}, Matrix{Float64}, Flux.OneHotArray, Flux.Chain, Function}(id, Xtrain, labelTrain, Ytrain, W, f, μ)
    end
end

function update!(c::Client, numEpoches::Int64=5, callback!::Function=identity)
    data = Flux.Data.DataLoader( (c.Xtrain, c.Ytrain), batchsize=25, shuffle=true)
    sqnorm(x) = sum(abs2, x)
    loss(x, y) = Flux.crossentropy( c.f(x) , y )
    lossReg(x, y) = loss(x, y) + c.μ*sum(sqnorm, params(c.W))
    opt = ADAM()
    for t = 1:numEpoches
        Flux.train!(lossReg, params(c.W), data, opt)
        callback!( (c, t) )
    end
    lss = loss(c.Xtrain, c.Ytrain)
    @printf "client: %d, loss: %.2f\n" c.id lss
    return lss
end

function get_predict(c::Client, X::Matrix{Float64}, minVal::Int64, maxVal::Int64)
    num = size(X, 2)
    vals = collect(minVal:maxVal)
    predict = Vector{Float64}(undef, num)
    for i = 1:num
        p = c.f(X[:,i])
        val = p'vals
        predict[i] = val
    end
    return predict
end

function performance(c::Client, X::Matrix{Float64}, label::Vector{Float64}, minVal::Int64, maxVal::Int64)
    predict = get_predict(c, X, minVal, maxVal)
    getMetrics(predict, label)
end

