using LinearAlgebra, SparseArrays, SparseMatricesCSR, ThreadedSparseCSR
using MKLSparse # to enable multithreaded Sparse CSC Matrix-Vec multiplication
using BenchmarkTools, PyPlot

function benchmark_csr_mv(sizes, densities)
    
    times_csc = zeros(length(sizes), length(densities))
    times_csc_transpose = zeros(length(sizes), length(densities))
    times_csr_mul = zeros(length(sizes), length(densities))
    times_csr_bmul = zeros(length(sizes), length(densities))
    times_csr_tmul = zeros(length(sizes), length(densities))
    
    for (j, d) in enumerate(densities)
        for (i, n) in enumerate(sizes)
            num_nzs = floor(Int, n*n*d)
            rows = rand(1:n, num_nzs)
            cols = rand(1:n, num_nzs)
            vals = rand(num_nzs)
            
            cscA = sparse(rows, cols, vals, n, n)
            cscAt = transpose(cscA)
            csrA = sparsecsr(rows, cols, vals, n, n)
            
            x = rand(n)
            y1 = zeros(n)
            y2 = zeros(n)
            y3 = zeros(n)
            y4 = zeros(n)
            y5 = zeros(n)
            
            b_csc = @benchmark mul!($y1, $cscA, $x, true, false)
            times_csc[i, j] = minimum(b_csc).time/1000 # time in microseconds
            
            b_csc_transpose = @benchmark mul!($y2, $cscAt, $x, true, false)
            times_csc_transpose[i, j] = minimum(b_csc_transpose).time/1000 # time in microseconds
            
            b_csr_mul = @benchmark mul!($y3, $csrA, $x, true, false)
            times_csr_mul[i, j] = minimum(b_csr_mul).time/1000 # time in microseconds
            
            b_csr_bmul = @benchmark bmul!($y4, $csrA, $x, true, false)
            times_csr_bmul[i, j] = minimum(b_csr_bmul).time/1000 # time in microseconds
            
            b_csr_tmul = @benchmark tmul!($y5, $csrA, $x, true, false)
            times_csr_tmul[i, j] = minimum(b_csr_tmul).time/1000 # time in microseconds
            
        end
    end
    
    return times_csc, times_csc_transpose, times_csr_mul, times_csr_bmul, times_csr_tmul
    
end

sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
densities = [0.01, 0.001, 0.0001]

times_csc, times_csc_transpose, times_csr_mul, times_csr_bmul, times_csr_tmul = benchmark_csr_mv(sizes, densities)

f, ax = subplots(1, 3, figsize = (13, 5))

for (i, d) in enumerate(densities)
    ax[i].loglog(sizes, times_csc[:, i], marker = "v", label = "MKLSparse, CSC")
    ax[i].loglog(sizes, times_csc_transpose[:, i], marker = "^", label = "MKLSparse, transpose(CSC)")
    ax[i].loglog(sizes, times_csr_mul[:, i], marker = "h", label = "non-threaded mul (CSR)")
    ax[i].loglog(sizes, times_csr_bmul[:, i], marker = "s", label = "bmul (CSR)")
    ax[i].loglog(sizes, times_csr_tmul[:, i], marker = "o", label = "tmul (CSR)")
    
    ax[i].set_title("non-zero density = $(d)")
    ax[i].set_xlabel("matrix size")
    ax[i].set_ylabel("minimum time [μs]")
    ax[i].set_xticks(sizes)
    ax[i].set_xticklabels(sizes)
end

legend()
tight_layout()
savefig("benchmark_csr_matvec.png", dpi = 300)