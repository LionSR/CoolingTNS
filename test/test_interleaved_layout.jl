using Test
using CoolingTNS

@testset "Interleaved system-bath layout" begin
    @test CoolingTNS.interleaved_total_sites(0) == 0
    @test CoolingTNS.interleaved_total_sites(3) == 6

    @test CoolingTNS.interleaved_system_site(1) == 1
    @test CoolingTNS.interleaved_bath_site(1) == 2
    @test CoolingTNS.interleaved_system_site(4) == 7
    @test CoolingTNS.interleaved_bath_site(4) == 8

    @test CoolingTNS.interleaved_system_sites(4) == [1, 3, 5, 7]
    @test CoolingTNS.interleaved_bath_sites(4) == [2, 4, 6, 8]

    sites = [:s1, :b1, :s2, :b2, :s3, :b3]
    @test CoolingTNS.interleaved_system_indices(sites, 3) == [:s1, :s2, :s3]
    @test CoolingTNS.interleaved_bath_indices(sites, 3) == [:b1, :b2, :b3]

    @test_throws ArgumentError CoolingTNS.interleaved_total_sites(-1)
    @test_throws ArgumentError CoolingTNS.interleaved_system_site(0)
    @test_throws ArgumentError CoolingTNS.interleaved_bath_site(0)
end
