#########################################################
# Helper functions
#########################################################

# read data
function load_data(filepath::String; indexed::Bool=false)
    lines = readlines(filepath)
    numData = length(lines) - 1
    if indexed
        numFeature = length( split(lines[2], ",") ) - 2
    else
        numFeature = length( split(lines[2], ",") ) - 1
    end
    X = zeros(Float64, numFeature, numData)
    label = zeros(Float64, numData)
    for i = 1:numData
        line = lines[i+1]
        info = split(line, ",")
        for j = 1:numFeature
            if indexed
                val = parse(Float64, info[j+1])
            else
                val = parse(Float64, info[j])
            end
            X[j,i] = val
        end
        label[i] = parse(Float64, info[end])
    end
    return numData, X, label
end

# split data into train and test
function train_test_split(num::Int64, X::Matrix{Float64}, label::Vector{Float64}, p::Float64)
    numTrain = Int(floor(p*num))
    perm = Random.randperm(num)
    trainIdx = perm[1:numTrain]
    testIdx = perm[numTrain+1:end]
    Xtrain = X[:, trainIdx]; labelTrain = label[trainIdx]
    Xtest = X[:, testIdx]; labelTest = label[testIdx]
    return Xtrain, labelTrain, Xtest, labelTest
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
        t += num_features_client
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

# type I mechanism model 
# single data point version
function mechanism1(x::Vector{Float64}, 
                    w::Vector{Float64},
                    c::Float64,
                    minVal::Int64,
                    maxVal::Int64)
    val = c + w'x
    val = round(val)
    val = min( max(val, minVal), maxVal)
    return Flux.onehotbatch(val, minVal:maxVal)
end
# data batch version
function mechanism1(x::Matrix{Float64}, 
                    w::Vector{Float64},
                    c::Float64,
                    minVal::Int64,
                    maxVal::Int64)
    numData = size(x, 2)
    out = map(i->mechanism1(x[:,i],w,c,minVal,maxVal), collect(1:numData))
    return hcat(out...)
end

# type II mechanism model 
# single data point version
function mechanism2(x::Vector{Float64},
                    l::Vector{Float64},
                    u::Vector{Float64},
                    d::Vector{Float64},
                    minVal::Int64, 
                    maxVal::Int64)
    numClass = maxVal - minVal + 1
    out = zeros(Float64, numClass)
    if (x ≥ l) && (x ≤ u) # x ∈ X 
        bound = 82.5263 - 80.26 * x[16] 
        if d'x ≤ 50 # x ∈ H
            bound = min(bound, 60)
        end
        bound = min( max(bound, minVal), maxVal)
        bound = convert(Int64, round(bound))
        idx = bound - minVal + 1
        if idx < numClass
            out = vcat( zeros(Float64, idx), -1e8*ones(Float64, numClass - idx))
        else
            out = zeros(Float64, numClass)
        end
    else
        out = zeros(Float64, numClass)
    end
    return out
end
# data batch version
function mechanism2(x::Matrix{Float64},
                    l::Vector{Float64},
                    u::Vector{Float64},
                    d::Vector{Float64},
                    minVal::Int64, 
                    maxVal::Int64)
    numData = size(x, 2)
    out = map(i->mechanism2(x[:,i],l,u,d,minVal,maxVal), collect(1:numData))
    return hcat(out...)
end


# Evaluation Metrics
function r2_score(predict::Vector{Float64}, label::Vector{Float64})
    v1 = 0
    v2 = 0
    y = mean(label)
    for i = 1:length(predict)
        v1 += (label[i] - predict[i])^2
        v2 += (label[i] - y)^2
    end
    return 1 - (v1/v2)
end

function user_satisfication(predict::Vector{Float64}, label::Vector{Float64})
    diff = predict - label
    mape = abs.( diff ./ label)
    ns = count( x->(x≤0.04), mape)
    return ns / length(predict)
end

function getMetrics(predict::Vector{Float64}, label::Vector{Float64})
    N = length(predict)
    R2 = r2_score(predict, label)
    MAE = L1dist(predict, label) / N
    RMSE = sqrt( sqL2dist(predict, label) / N )
    diff = predict - label
    MAPE = mean( abs.(diff ./ label) )
    PCCS = cor(predict, label)
    US = user_satisfication(predict, label)
    @printf "N: %d, R2: %.2f, MAE: %.2f, RMSE: %.2f, 1-MAPE: %.3f, PCCS: %.2f, US: %.2f\n" N R2 MAE RMSE 1-MAPE PCCS US 
    return nothing
end

# read data from libsvm
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
            if value < 0
                value = Int64(2)
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


