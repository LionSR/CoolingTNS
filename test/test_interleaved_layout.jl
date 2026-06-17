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

    @test CoolingTNS.interleaved_bit_position(1) == 0
    @test CoolingTNS.interleaved_system_bit(1) == 0
    @test CoolingTNS.interleaved_bath_bit(1) == 1
    @test CoolingTNS.interleaved_system_bits(4) == [0, 2, 4, 6]
    @test CoolingTNS.interleaved_bath_bits(4) == [1, 3, 5, 7]

    @test CoolingTNS.interleaved_system_basis_state(0b10, 2) == 0b0100
    @test CoolingTNS.interleaved_basis_state(0b10, 0b01, 2) == 0b0110

    @test_throws ArgumentError CoolingTNS.interleaved_total_sites(-1)
    @test_throws ArgumentError CoolingTNS.interleaved_system_site(0)
    @test_throws ArgumentError CoolingTNS.interleaved_bath_site(0)
    @test_throws ArgumentError CoolingTNS.interleaved_bit_position(0)
    @test_throws ArgumentError CoolingTNS.interleaved_basis_state(0b100, 0, 2)
    @test_throws ArgumentError CoolingTNS.interleaved_basis_state(0, 0b100, 2)
end
