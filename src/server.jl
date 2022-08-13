mutable struct Server{  T1<:Flux.Chain, 
                        T2<:Vector{Union{Client,ClientImg}},
                        T3<:Vector{Int64},
                        T4<:Int64}
    W::T1                   # model
    clients::T2             # clients
    selectedIndices::T3     # indices of selected clients 
    τ::T4                   # number of selected cients
    function Server(clients::Vector{Union{Client,ClientImg}}, τ::Int64)
        W = deepcopy( clients[1].W )
        selectedIndices = Vector{Int64}(undef, τ)
        new{Flux.Chain, Vector{Union{Client,ClientImg}}, Vector{Int64}, Int64}(W, clients, selectedIndices, τ)
    end
end 

function select!(s::Server)
    numClients = length(s.clients)
    s.selectedIndices = randperm(numClients)[1:s.τ]
end

function sendModel!(s::Server)
    for i in s.selectedIndices
        c = s.clients[i]
        for j = 1:length(Flux.params(s.W))
            Flux.params(c.W)[j] .= deepcopy( Flux.params(s.W)[j] )
        end
    end
end

function aggregate!(s::Server)
    l = length(Flux.params(s.W))
    for j = 1:l
        fill!(Flux.params(s.W)[j], 0.0)
    end
    for i in s.selectedIndices
        c = s.clients[i]
        for j = 1:l
            Flux.params(s.W)[j] .+= (1/s.τ) * deepcopy( Flux.params(c.W)[j] )
        end
    end
end

function training!(s::Server, T::Int64)
    for t = 1:T
        # @printf "round: %d\n" t
        # select clients
        select!(s)
        # send global model to selected clients
        sendModel!(s)
        # local update for selected clients
        lss = 0.0
        for i in s.selectedIndices
            c = s.clients[i]
            local_lss = update!(c)
            lss += local_lss
        end
        # @printf "global loss: %.2f\n" lss
        # global aggregation
        aggregate!(s)
    end
end
