#########################################################
# Helper functions
#########################################################

# split data into train and test
function train_test_split(X::SparseMatrixCSC{Float32,Int64}, Y::Vector{Int64}, p::Float64)
    num = size(X, 2)
    numTrain = Int(floor(p * num))
    perm = Random.randperm(num)
    trainIdx = perm[1:numTrain]
    testIdx = perm[numTrain+1:end]
    Xtrain = X[:, trainIdx]
    Ytrain = Y[trainIdx]
    Xtest = X[:, testIdx]
    Ytest = Y[testIdx]
    return Xtrain, Ytrain, Xtest, Ytest
end

# horizontally split data
function split_data(Xtrain::SparseMatrixCSC{Float32,Int64}, Ytrain::Vector{Int64}, num_clients::Int64)
    num_data = size(Xtrain, 2)
    num_data_client = div(num_data, num_clients)
    Xtrain_split = Vector{SparseMatrixCSC{Float32,Int64}}(undef, num_clients)
    Ytrain_split = Vector{Vector{Int64}}(undef, num_clients)
    t = 1
    for i = 1:num_clients
        if i < num_clients
            ids = collect(t:t+num_data_client-1)
        else
            ids = collect(t:num_data)
        end
        Xtrain_split[i] = Xtrain[:, ids]
        Ytrain_split[i] = Ytrain[ids]
        t += num_data_client
    end
    return Xtrain_split, Ytrain_split
end

# label Transformation
function label_transformation(Label::Vector{Float32}, minVal::Int64, maxVal::Int64)
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
            nnz += (length(info) - 1)
            if line[end] == ' '
                nnz -= 1
            end
        end
    end
    # @printf("number of lines: %i\n", numLine)
    n = numLine
    m = 0
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float32, nnz)
    if tag == "classification"
        y = zeros(Int64, n)
    else
        y = zeros(Float32, n)
    end
    numLine = 0
    cc = 1
    open(filename, "r") do f
        while !eof(f)
            numLine += 1
            line = readline(f)
            info = split(line, " ")
            if tag == "classification"
                value = parse(Int64, info[1])
            else
                value = parse(Float32, info[1])
            end
            y[numLine] = value
            ll = length(info)
            if line[end] == ' '
                ll -= 1
            end
            for i = 2:ll
                if !occursin(":", info[i]) 
                    # println(numLine, " ", i," ",ll," ",info[i])
                    println("Warn: check input format")
                    continue 
                end
                idx, value = split(info[i], ":")
                idx = parse(Int, idx)
                value = parse(Float32, value)
                I[cc] = numLine
                J[cc] = idx
                V[cc] = value
                cc += 1
                m = max(m, idx)
            end
        end
    end
    return sparse(J, I, V, m, n), y
end

function splitDataByClass(X::SparseMatrixCSC{Float32,Int64},
    Y::Vector{Int64},
    num_clients::Int64,
    num_classes::Int64,
    num_classes_per_client::Int64 
    )
    Xsplit = Vector{SparseMatrixCSC{Float32,Int64}}(undef, num_clients)
    Ysplit = Vector{Vector{Int64}}(undef, num_clients)
    # assign num_classes_per_client classes to each client 
    classes_clients = Vector{Vector{Int64}}(undef, num_clients)
    for i in 1:num_clients
        classes = sample(1:num_classes, num_classes_per_client, replace=false)
        classes_clients[i] = classes
    end
    # clients in each class # possible that a class not covered by any client
    clients_in_classes = [[] for _ = 1:num_classes]
    for i = 1:num_classes
        for j = 1:num_clients
            if i in classes_clients[j]
                push!(clients_in_classes[i], j)
            end
        end
    end
    # intialize indices
    indices = [[] for _ = 1:num_clients]
    for i = 1:length(Y)
        class = Y[i]
        j = rand(clients_in_classes[class])
        push!(indices[j], i)
    end
    # fill in data
    for i in 1:num_clients
        ids = indices[i]
        Xsplit[i] = copy(X[:, ids])
        Ysplit[i] = copy(Y[ids])
    end
    return Xsplit, Ysplit
end

function buildPredModel(X::SparseMatrixCSC{Float32,Int64}, Yhot::Flux.OneHotArray, numClass::Int64)
    # logistic regression with limited features
    numFeatures = size(X, 1)
    numSelectedFeatures = floor(Int, 0.3 * numFeatures)
    selectedFeatures = randperm(numFeatures)[1:numSelectedFeatures]
    mask = zeros(numFeatures)
    mask[selectedFeatures] .= 1.0
    model = Chain(Dense(numFeatures, numClass), softmax)
    g0(x) = model(x .* mask)
    loss(x, y) = Flux.crossentropy(g0(x), y)
    batchsize=32
    nsamples = size(X,2); 
    if nsamples < batchsize 
        batchsize=nsamples; #println("warn bs 32 -> ", nsamples) 
    end
    # limited_data_idxs = randperm(nsamples)[1:(nsamples÷10)]
    data = Flux.Data.DataLoader((X, Yhot), # [:,limited_data_idxs]
                                batchsize=batchsize,
                                shuffle=true)
    opt = Descent()
        Flux.train!(loss, Flux.params(model), data, opt)
    function g(x::SparseVector{Float32,Int64})
        idx = argmax(g0(x))
        return Flux.onehot(idx, 1:numClass)
    end
    function g(x::SparseMatrixCSC{Float32,Int64})
        num = size(x, 2)
        out = map(i -> g(x[:, i]), collect(1:num))
        return hcat(out...)
    end
    return g
end

function buildRangeModel(X::SparseMatrixCSC{Float32,Int64},
    Y::Vector{Int64},
    numClass::Int64,
    numClassSub::Int64,
    g::Function)
    DictData = Dict{SparseVector{Float32},Vector{Float32}}()
    numData = size(X, 2)
    classes = collect(Set(Y))
    for i = 1:numData
        v = -1e8 * ones(numClass)
        trueLabel = Y[i]
        v[trueLabel] = 0.0
        predLabel = argmax(g(X[:, i]))
        v[predLabel] = 0.0
        cs = sample(classes, numClassSub, replace=false)
        v[cs] .= 0.0
        DictData[X[:, i]] = v
    end
    function h(x::SparseVector{Float32,Int64})
        if haskey(DictData, x)
            return DictData[x]
        else
            return zeros(numClass)
        end
    end
    function h(x::SparseMatrixCSC{Float32,Int64})
        num = size(x, 2)
        out = map(i -> h(x[:, i]), collect(1:num))
        return hcat(out...)
    end
    return h
end

function LeNet5(; imgsize=(28, 28, 1), nclasses=10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)
    # @printf("%d %d ", out_conv_size[1], out_conv_size[2]) 
    return Chain(
        Conv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses), NNlib.softmax  # i think sofxmax needed here
    )
end

function LeNet5small(; imgsize=(14, 14, 1), nclasses=10) # actually accept 28x28 size input 
    return Chain(
        MaxPool((2, 2)),
        Conv((5, 5), imgsize[end] => 6, relu),
        Flux.flatten,
        Dense(600, 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses),
        NNlib.softmax
    )
end

using NNlib, Metalhead, MLUtils

function MyModel(; large=true)
    if large 
        model = Metalhead.ResNet(34; pretrain = true) # 50 
        m2=Chain(
            X->NNlib.upsample_bilinear(X, (7,7) ), 
            model.layers[1], 
            AdaptiveMeanPool((1, 1)), MLUtils.flatten, Dropout(.1), 
            Dense(512=>10) # 2048 
        ) 
    else
        model = Metalhead.ResNet(18; pretrain = true) 
        m2=Chain(
            X->NNlib.meanpool(X, (4,4)), 
            X->NNlib.upsample_bilinear(X, (28,28)),  
            # X->NNlib.upsample_bilinear(X, (14,14)), # this save memory
            model.layers[1], 
            AdaptiveMeanPool((1, 1)), MLUtils.flatten, Dropout(.1), 
            Dense(512=>10), NNlib.softmax 
        )
    end
    Flux.trainmode!(m2)
    return m2
end

using MLUtils: mapobs
function buildPredModelImg(X::Array{Float32,4}, Yhot::Flux.OneHotArray; epochs=2) 
    is_cifar=size(X, 3)==3 
    if !is_cifar 
        # @printf("not cifar")
        model = LeNet5small() 
        batchsize=1000
    else
        model = MyModel(large=false) 
        batchsize=32
    end
    model = model |> device 
    loss(x, y) = Flux.crossentropy(model(x), y)
    nsamples = size(X,2); 
    if nsamples < batchsize 
        batchsize=nsamples; #println("warn bs 32 -> ", nsamples) 
    end
    data = Flux.DataLoader(
        mapobs(device, (X, Yhot)),
        batchsize=batchsize, 
        shuffle=true)
    # opt = Flux.Optimise.Optimiser(WeightDecay(1f-4), Adam())
    opt = Descent()  
    for t = 1:epochs 
        Flux.train!(loss, Flux.params(model), data, opt)
    end
    Flux.testmode!(model) 
    # @printf("a prediction model built, train acc: "); performance_2(model, X, Flux.onecold(Yhot, 0:9)) 
    # @printf("a prediction model built\n") 
    function g(x::Array{Float32,4})
        x = x |> device 
        idxs = [p[1] for p in argmax(cpu(model(x)), dims=1)]
        idxs = reshape(idxs, length(idxs))
        return Flux.onehotbatch(idxs, 1:10) 
    end
    function g(x::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}) 
        idxs = [p[1] for p in argmax(cpu(model(x)), dims=1)]
        idxs = reshape(idxs, length(idxs))
        return Flux.onehotbatch(idxs, 1:10) |> device 
    end 
    return g 
end

function buildRangeModelImg(X::Array{Float32,4},
    Y::Vector{Int64},
    numClass::Int64,
    numClassSub::Int64,
    g::Function)

    DictData = Dict{Matrix{Float32},Vector{Float32}}()
    numData = size(X, 4)
    classes = collect(Set(Y))
    for i = 1:numData
        v = -1e8 * ones(numClass)
        x = X[:, :, :, i:i]
        trueLabel = Y[i]
        v[trueLabel+1] = 0.0
        predLabel = argmax(g(x))[1] 
        v[predLabel] = 0.0
        cs = sample(classes, numClassSub, replace=false)
        cs .+= 1
        v[cs] .= 0.0 
        k=MaxPool((2, 2))(x)[:, :, 1, 1]
        if haskey(DictData, k) 
            @printf("warn: overide dict")
        end
        DictData[k] = v
    end
    function h(x::Union{Matrix{Float32}, CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}})
        x=x|>cpu 
        if haskey(DictData, x)
            return DictData[x]
        else
            @printf("warning: range model not for this x") 
            return zeros(numClass)
        end
    end
    function h(x::Array{Float32,4})
        num = size(x, 4)
        out = map(i -> h(MaxPool((2, 2))(x[:, :, :, i:i])[:, :, 1, 1]), collect(1:num))
        return hcat(out...)
    end
    function h(x::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}) 
        x=x|>cpu 
        num = size(x, 4)
        out = map(i -> h(MaxPool((2, 2))(x[:, :, :, i:i])[:, :, 1, 1]), collect(1:num))
        return hcat(out...) |> device 
    end 
    return h
end

function splitDataByClassImg(Xtrain::Array{Float32,4},
    Ytrain::Vector{Int64},
    Xtest::Array{Float32,4},
    Ytest::Vector{Int64},
    num_clients::Int64,
    num_classes::Int64,
    num_classes_per_client::Int64)
    XtrainSplit = Vector{Array{Float32,4}}(undef, num_clients)
    YtrainSplit = Vector{Vector{Int64}}(undef, num_clients)
    XtestSplit = Vector{Array{Float32,4}}(undef, num_clients)
    YtestSplit = Vector{Vector{Int64}}(undef, num_clients)
    # assign num_classes_per_client classes to each client 
    classes_clients = Vector{Vector{Int64}}(undef, num_clients)
    for i in 1:num_clients
        classes = sample(1:num_classes, num_classes_per_client, replace=false)
        classes_clients[i] = classes
    end
    # clients in each class
    clients_in_classes = [[] for _ = 1:num_classes]
    for i = 1:num_classes
        for j = 1:num_clients
            if i in classes_clients[j]
                push!(clients_in_classes[i], j)
            end
        end
    end
    # intialize indices
    trainIndices = [[] for _ = 1:num_clients]
    for i = 1:length(Ytrain)
        class = Ytrain[i] + 1
        j = rand(clients_in_classes[class])
        push!(trainIndices[j], i)
    end
    testIndices = [[] for _ = 1:num_clients]
    for i = 1:length(Ytest)
        class = Ytest[i] + 1
        j = rand(clients_in_classes[class])
        push!(testIndices[j], i)
    end
    # fill in data
    for i in 1:num_clients
        ids1 = trainIndices[i]
        ids2 = testIndices[i]
        XtrainSplit[i] = copy(Xtrain[:, :, :, ids1])
        YtrainSplit[i] = Ytrain[ids1]
        XtestSplit[i] = copy(Xtest[:, :, :, ids2])
        YtestSplit[i] = Ytest[ids2]
    end
    return XtrainSplit, YtrainSplit, XtestSplit, YtestSplit
end

function show_name(withMech, withFed)
    if withMech == false && withFed == 0
        # ML
        println("ML")
    elseif withMech == true && withFed == 0
        # MLwKM
        println("MLwKM")
    elseif withMech == false && withFed == 1
        # FL
        println("FL")
    elseif withMech == true && withFed == 1
        # FLwKM
        println("FLwKM")
    elseif withMech == false && withFed == 2
        # AD
        println("ADAP")
    elseif withMech == true && withFed == 2
        # ADwKM
        println("ADAPwKM")
    elseif withMech == false && withFed == 3
        # with proximal term (ditto)
        println("DITTO")
    elseif withMech == true && withFed == 3
        println("DITTOwKM")
    end
end

function mean_plusminus_std(data)
    μ = mean(data)
    delta = std(data)
    return     string(@sprintf("%.2f", μ), "±", @sprintf("%.2f", delta))
end

