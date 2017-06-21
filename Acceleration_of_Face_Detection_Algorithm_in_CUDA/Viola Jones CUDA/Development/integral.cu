/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
	int x, y, s, sq, t, tq;
	unsigned char it;
	int height = src->height;
	int width = src->width;
	unsigned char *data = src->data;
	int * sumData = sum->data;
	int * sqsumData = sqsum->data;

	// CUDA Point
	for( y = 0; y < height; y++)
		{
			s = 0;
			sq = 0;
			/* loop over the number of columns */
			for( x = 0; x < width; x ++)
			{
				it = data[y*width+x];
				/* sum of the current row (integer)*/
				s += it; 
				sq += it*it;

				t = s;
				tq = sq;
				if (y != 0)
				{
					t += sumData[(y-1)*width+x];
					tq += sqsumData[(y-1)*width+x];
				}
				sumData[y*width+x]=t;
				sqsumData[y*width+x]=tq;
			}
		}
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum){

	int height = src->height;
	int width = src->width;

	int size = height*width;

	//arrays in main memory
	int* sumData = sum->data;
	int* sqsumData = sqsum->data;
	unsigned char * data = src->data;
	
	//pointers for arrays to be put on cuda memory
	int *sumData_cuda;
	int *sqsumData_cuda;
	unsigned char * data_cuda;

	
	//allocate memory in cuda device
	cudaMalloc((void **)&sumData_cuda,sizeof(int)*size);
	cudaMalloc((void **)&sqsumData_cuda,sizeof(int)*size);		
	cudaMalloc((void **)&data_cuda,sizeof(unsigned char)*size);
	
	//copy contents from main memory to cuda device memory
	cudaMemcpy(data_cuda, data, sizeof(unsigned char)*size, cudaMemcpyHostToDevice);
	checkCudaError();
	
	//thread configuration

	int blocks, threadsPerBlock;
	
	if (height>THREADS){
		blocks = ceil(height/THREADS);
		threadsPerBlock = THREADS;
	} else {
		blocks = 1;
		threadsPerBlock = THREADS;
	}

	//call the cuda kernel
	integralImageRows_GPU<<<blocks,threadsPerBlock>>>(sumData_cuda, sqsumData_cuda, data_cuda, width, height);
	cudaDeviceSynchronize();
	checkCudaError();

	if (width>THREADS){
		blocks = ceil(width/THREADS);
		threadsPerBlock = THREADS;
	} else {
		blocks = 1;
		threadsPerBlock = THREADS;
	}
	integralImageCols_GPU<<<blocks,threadsPerBlock>>>(sumData_cuda, sqsumData_cuda, width, height);

	cudaDeviceSynchronize();
	checkCudaError();
	
	//copy back the results from cuda memory to main memory
	cudaMemcpy(sumData, sumData_cuda, sizeof(int)*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(sqsumData, sqsumData_cuda, sizeof(int)*size, cudaMemcpyDeviceToHost);
	checkCudaError();
	cudaFree(sumData_cuda);
	cudaFree(sqsumData_cuda);
	cudaFree(data_cuda);

}



__global__  void integralImageRows_GPU(int *sumData_cuda, int *sqsumData_cuda, unsigned char *data_cuda, int width, int height){

	int bidx = blockIdx.x*1024+threadIdx.x;

	int i;
	unsigned char temp;

	int sh = 0, sqh = 0;

	if (bidx < height){
		for (i = 0; i<width; i++){
			temp = data_cuda[bidx*width + i];
			sh += temp;
			sqh += temp*temp;
			sumData_cuda[bidx*width + i] = sh;
			sqsumData_cuda[bidx*width + i] = sqh;
		}
	}
}

__global__  void integralImageCols_GPU(int *sumData_cuda, int *sqsumData_cuda, int width, int height){	

	int bidx = blockIdx.x*1024+threadIdx.x;

	int i;

	if (bidx < width){
		for (i = 1; i<height; i++){
			sumData_cuda[width*i + bidx] += sumData_cuda[width*(i-1) + bidx];
			sqsumData_cuda[width*i + bidx] += sqsumData_cuda[width*(i-1) + bidx];
		}
	}

}