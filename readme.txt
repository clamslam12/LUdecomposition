1) Compile with g++ -O -fopenmp lu-omp.cpp -o lu-omp
2) Run with ./lu-omp [matrix_size] [#_threads] (ex ./lu-omp 1024 16) matrix_size = 1024; #_threads = 16
3) Proof of correctness, runtimes, and speedup are in the luDecompStats.pdf file.