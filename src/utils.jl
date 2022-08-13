#########################################################
# Helper functions
#########################################################

# split data into train and test
function train_test_split(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, p::Float64)
    num = size(X, 2)
    numTrain = Int(floor(p*num))
    perm = Random.randperm(num)
    trainIdx = perm[1:numTrain]
    testIdx = perm[numTrain+1:end]
    Xtrain = X[:, trainIdx]; Ytrain = Y[trainIdx]
    Xtest = X[:, testIdx]; Ytest = Y[testIdx]
    return Xtrain, Ytrain, Xtest, Ytest
end

# horizontally split data
function split_data(Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, num_clients::Int64)
    num_data = size(Xtrain, 2)
    num_data_client = div(num_data, num_clients)
    Xtrain_split = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
    Ytrain_split = Vector{ Vector{Int64} }(undef, num_clients)
    t = 1
    for i = 1:num_clients
        if i < num_clients
            ids = collect(t: t+num_data_client-1)
        else
            ids = collect(t: num_data)
        end
        Xtrain_split[i] = Xtrain[:, ids]
        Ytrain_split[i] = Ytrain[ids]
        t += num_data_client
    end
    return Xtrain_split, Ytrain_split
end

# label Transformation
function label_transformation(Label::Vector{Float64}, minVal::Int64, maxVal::Int64)
    num = length(Label)
    Y = Vector{Int64}(undef, num)
    for i = 1:num
        Y[i] = convert(Int64, round(Label[i]))
    end
    return Flux.onehotbatch(Y, minVal:maxVal)
end

# read data from libsvm
function read_libsvm(filename::String, tag::String="classification")
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
    if tag == "classification"
        y = zeros(Int64, n)
    else
        y = zeros(Float64, n)
    end
    numLine = 0
    cc = 1
    open(filename, "r") do f
        while !eof(f)
            numLine += 1
            line = readline(f)
            info = split(line, " ")
            if tag == "classification"
                value = parse(Int64, info[1] )
            else
                value = parse(Float64, info[1] )
            end
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

function splitDataByClass(X::SparseMatrixCSC{Float64, Int64}, 
                          Y::Vector{Int64}, 
                          num_clients::Int64, 
                          num_classes::Int64,
                          num_classes_per_client::Int64)
    Xsplit = Vector{ SparseMatrixCSC{Float64, Int64} }(undef, num_clients)
    Ysplit = Vector{ Vector{Int64} }(undef, num_clients)
    # assign num_classes_per_client classes to each client 
    classes_clients = Vector{Vector{Int64}}(undef, num_clients)
    for i in 1:num_clients
        classes = sample(1:num_classes, num_classes_per_client, replace=false)
        classes_clients[i] = classes
    end
    # clients in each class
    clients_in_classes = [ [] for _ = 1:num_classes]
    for i = 1:num_classes
        for j = 1:num_clients
            if i in classes_clients[j]
                push!(clients_in_classes[i], j)
            end
        end
    end
    # intialize indices
    indices = [ [] for _ = 1:num_clients]
    for i = 1:length(Y)
        class = Y[i]
        j = rand(clients_in_classes[class])
        push!(indices[j], i)
    end
    # fill in
    for i in 1:num_clients
        ids = indices[i]
        Xsplit[i] = copy( X[:,ids] )
        Ysplit[i] = Y[ids]
    end
    return Xsplit, Ysplit
end

function buildPredModel(X::SparseMatrixCSC{Float64, Int64}, Yhot::Flux.OneHotArray, numClass::Int64)
    # logistic regression with limited features
    numFeatures = size(X, 1)
    numSelectedFeatures = floor(Int, 0.3*numFeatures)
    selectedFeatures = randperm(numFeatures)[1:numSelectedFeatures]
    mask = zeros(numFeatures); mask[selectedFeatures] .= 1.0
    model = Chain(Dense(numFeatures, numClass), softmax)
    g0(x) = model(x.*mask)
    loss(x, y) = Flux.crossentropy( g0(x) , y )
    data = Flux.Data.DataLoader( (X, Yhot), 
                                batchsize=25, 
                                shuffle=true )
    opt = ADAM()
    for t = 1:5
        Flux.train!(loss, Flux.params(model), data, opt)
    end
    function g(x::SparseVector{Float64, Int64})
        idx = argmax( g0(x) )
        return Flux.onehot(idx, 1:numClass)
    end
    function g(x::SparseMatrixCSC{Float64, Int64})
        num = size(x, 2)
        out = map(i->g(x[:,i]), collect(1:num))
        return hcat(out...)
    end
    return g
end

function buildRangeModel(X::SparseMatrixCSC{Float64, Int64}, 
                         Y::Vector{Int64}, 
                         numClass::Int64,
                         numClassSub::Int64,
                         g::Function)
    DictData = Dict{SparseVector{Float64}, Vector{Float64}}()
    numData = size(X, 2)
    classes = collect(Set(Y))
    for i = 1:numData
        v = -1e8*ones(numClass)
        trueLabel = Y[i]
        v[trueLabel] = 0.0
        predLabel = argmax(g(X[:,i]))
        v[predLabel] = 0.0
        cs = sample( classes, numClassSub, replace=false )
        v[cs] .= 0.0
        DictData[X[:,i]] = v
    end
    function h(x::SparseVector{Float64, Int64})
        if haskey(DictData, x) 
            return DictData[x]
        else
            return zeros(numClass)
        end
    end
    function h(x::SparseMatrixCSC{Float64, Int64})
        num = size(x, 2)
        out = map(i->h(x[:,i]), collect(1:num))
        return hcat(out...)
    end
    return h
end

function LeNet5(; imgsize=(28,28,1), nclasses=10) 

    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

function LeNet5small(; imgsize=(14,14,1), nclasses=10) 
    
    return Chain(
            MaxPool((2, 2)),
            Conv((5, 5), imgsize[end]=>6, relu),
            Flux.flatten,
            Dense(600, 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses),
            NNlib.softmax
          )
end

function buildPredModelImg(X::Array{Float64, 4}, Yhot::Flux.OneHotArray)
    model = LeNet5small()
    loss(x, y) = Flux.crossentropy( model(x) , y )
    data = Flux.Data.DataLoader( (X, Yhot), 
                                batchsize=25, 
                                shuffle=true )
    opt = ADAM()
    for t = 1:10
        Flux.train!(loss, Flux.params(model), data, opt)
    end
    function g(x::Array{Float64, 4})
       idxs = [p[1] for p in argmax(model(x), dims=1)]
       idxs = reshape(idxs, length(idxs))
       return Flux.onehotbatch(idxs, 1:10)
    end
end           
              
function buildRangeModelImg(X::Array{Float64, 4}, 
                            Y::Vector{Int64}, 
                            numClass::Int64,
                            numClassSub::Int64,
                            g::Function)
    DictData = Dict{Matrix{Float64}, Vector{Float64}}()
    numData = size(X, 4)
    classes = collect(Set(Y))
    for i = 1:numData
        v = -1e8*ones(numClass)
        x = X[:,:,:,i:i]
        trueLabel = Y[i]
        v[trueLabel+1] = 0.0
        predLabel = argmax(g(x))[1]
        v[predLabel] = 0.0
        cs = sample( classes, numClassSub, replace=false )
        cs .+= 1
        v[cs] .= 0.0
        DictData[MaxPool((2, 2))(x)[:,:,1,1]] = v
    end
    function h(x::Matrix{Float64})
        if haskey(DictData, x) 
            return DictData[x]
        else
            return zeros(numClass)
        end
    end
    function h(x::Array{Float64, 4})
        num = size(x, 4)
        out = map(i->h( MaxPool((2, 2))(x[:,:,:,i:i])[:,:,1,1] ), collect(1:num))
        return hcat(out...)
    end
    return h
end

function splitDataByClassImg(Xtrain::Array{Float64, 4}, 
                             Ytrain::Vector{Int64},
                             Xtest::Array{Float64, 4}, 
                             Ytest::Vector{Int64}, 
                             num_clients::Int64, 
                             num_classes::Int64,
                             num_classes_per_client::Int64)
    XtrainSplit = Vector{ Array{Float64, 4} }(undef, num_clients)
    YtrainSplit = Vector{ Vector{Int64} }(undef, num_clients)
    XtestSplit = Vector{ Array{Float64, 4} }(undef, num_clients)
    YtestSplit = Vector{ Vector{Int64} }(undef, num_clients)
    # assign num_classes_per_client classes to each client 
    classes_clients = Vector{Vector{Int64}}(undef, num_clients)
    for i in 1:num_clients
        classes = sample(1:num_classes, num_classes_per_client, replace=false)
        classes_clients[i] = classes
    end
    # clients in each class
    clients_in_classes = [ [] for _ = 1:num_classes]
    for i = 1:num_classes
        for j = 1:num_clients
            if i in classes_clients[j]
                push!(clients_in_classes[i], j)
            end
        end
    end
    # intialize indices
    trainIndices = [ [] for _ = 1:num_clients]
    for i = 1:length(Ytrain)
        class = Ytrain[i]+1
        j = rand(clients_in_classes[class])
        push!(trainIndices[j], i)
    end
    testIndices = [ [] for _ = 1:num_clients]
    for i = 1:length(Ytest)
        class = Ytest[i]+1
        j = rand(clients_in_classes[class])
        push!(testIndices[j], i)
    end
    # fill in
    for i in 1:num_clients
        ids1 = trainIndices[i]
        ids2 = testIndices[i]
        XtrainSplit[i] = copy( Xtrain[:,:,:,ids1] )
        YtrainSplit[i] = Ytrain[ids1]
        XtestSplit[i] = copy( Xtest[:,:,:,ids2] )
        YtestSplit[i] = Ytest[ids2]
    end
    return XtrainSplit, YtrainSplit, XtestSplit, YtestSplit
end

                       
               
                     
                     
                     