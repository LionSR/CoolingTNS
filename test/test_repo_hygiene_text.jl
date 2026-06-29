using Test

@testset "Repository hygiene ignore rules" begin
    gitignore_lines = Set(strip.(split(read(joinpath(@__DIR__, "..", ".gitignore"), String), '\n')))

    @test "/build/" in gitignore_lines
    @test "Notes/*/build/" in gitignore_lines
    @test "slides/build/" in gitignore_lines
    @test ".worktree/" in gitignore_lines
    @test ".worktrees/" in gitignore_lines
    @test ".claude/worktrees/" in gitignore_lines
end
