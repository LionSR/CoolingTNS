using ITensors

ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) = 2 * op("Sx", s)
ITensors.op(::OpName"σy", ::SiteType"S=1/2", s::Index) = 2 * op("Sy", s)
ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) = 2 * op("Sz", s)

N = 10
pe = 0.01

sites = siteinds("S=1/2", 2N)

for i in 1:100
    println("i: ", i)
    os = Tuple{String,Int64}[]
    for j in eachindex(sites)
        p = rand()
        if p < pe / 3
            push!(os, ("σx", j))
        elseif pe / 3 <= p < 2 * pe / 3
            push!(os, ("σy", j))
        elseif 2 * pe / 3 <= p < pe
            push!(os, ("σz", j))
        end
    end
    println("sites: ", typeof(sites))
    println("os: ", os)
    println("os: ", typeof(os))
    noise_gates = ops(os, sites)
    println("noise_gates: ", typeof(noise_gates))
end


