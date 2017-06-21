NVCC ?= nvcc

HEADERS := image.h haar.h stdio-wrapper.h

all: build

build: vj

image.o: image.c $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

stdio.o: stdio-wrapper.c $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

main.o: main.cpp $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

haar.o: haar.cu $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

rectangles.o: rectangles.cpp $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

gpu_functions.o: gpu_functions.cu $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

helpers.o: helpers.cu $(HEADERS)
	$(NVCC) -o $@ -c $< -O3

vj: main.o haar.o image.o stdio-wrapper.o rectangles.o gpu_functions.o helpers.o
	$(NVCC) -o $@ $+ $(LDFLAGS) -O3

run: build
	./vj

clean:
	rm -f vj *.o Output.pgm
