all:
	make build
	make run

build:
	nvcc -Xptxas -O3 -Xcompiler -O3 -v src/main.cu -I include -ccbin g++-12 -lcudart_static -lcublas -ldl -lrt -lpthread -lstdc++fs -O3 -o main.gpu -std=c++17 --expt-relaxed-constexpr

run:
	./main.gpu
