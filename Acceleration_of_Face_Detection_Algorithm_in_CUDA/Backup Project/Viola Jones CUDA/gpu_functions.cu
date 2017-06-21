#include "gpu_functions.cuh"
#include <stdio.h>

/** CUDA kernel to add two vectors*/

// USELESS
__global__  void integralImageRows_GPU(int *sumData_cuda, int *sqsumData_cuda, unsigned char *data_cuda, int width, int height){

	int bidx = blockIdx.x*THREADS2+threadIdx.x;

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

	int bidx = blockIdx.x*THREADS2+threadIdx.x;

	int i;

	if (bidx < width){
		for (i = 1; i<height; i++){
			sumData_cuda[width*i + bidx] += sumData_cuda[width*(i-1) + bidx];
			sqsumData_cuda[width*i + bidx] += sqsumData_cuda[width*(i-1) + bidx];
		}
	}

}



__global__ void runCascadeClassifier(int* result_cuda, int start_stage, int cascade_n_stages, int cascade_inv_window_area, int cascade_sum_width, int cascade_sqsum_width, int x2, int y2,
									int* cascade_pq0_cuda, int cascade_pq1, int cascade_pq2, int cascade_pq3, 
									int* cascade_p0_cuda, int cascade_p1, int cascade_p2, int cascade_p3,
									int* tree_thresh_array_cuda, int* scaled_rectangles_array_cuda, int* weights_array_cuda, int* alpha1_array_cuda, int* alpha2_array_cuda, int* stages_thresh_array_cuda, int* stages_array_cuda)
{

	int p_offset, pq_offset;
	int i, j;
	unsigned int mean;
	unsigned int variance_norm_factor;
	int haar_counter = 0;
	int w_index = 0;
	int r_index = 0;
	int stage_sum;
	bool end = false;

	int* cascade_pq1_cuda = cascade_pq0_cuda + cascade_pq1;
	int* cascade_pq2_cuda = cascade_pq0_cuda + cascade_pq2;
	int* cascade_pq3_cuda = cascade_pq0_cuda + cascade_pq3;
	int* cascade_p1_cuda = cascade_p0_cuda + cascade_p1;
	int* cascade_p2_cuda = cascade_p0_cuda + cascade_p2;
	int* cascade_p3_cuda = cascade_p0_cuda + cascade_p3;

	// blockId
	int ptx = blockIdx.x*blockDim.x + threadIdx.x;
	int pty = blockIdx.y*blockDim.y + threadIdx.y;

	if (ptx <= x2 && pty <= y2){

		int index = pty*x2 + ptx;
		
		p_offset = pty * (cascade_sum_width) + ptx;
		pq_offset = pty * (cascade_sqsum_width) + ptx;

		variance_norm_factor =  (cascade_pq0_cuda[pq_offset] - cascade_pq1_cuda[pq_offset] - cascade_pq2_cuda[pq_offset] + cascade_pq3_cuda[pq_offset]);
		mean = (cascade_p0_cuda[p_offset] - cascade_p1_cuda[p_offset] - cascade_p2_cuda[p_offset] + cascade_p3_cuda[p_offset]);

		variance_norm_factor = (variance_norm_factor*cascade_inv_window_area);
		variance_norm_factor =  variance_norm_factor - mean*mean;

		if( variance_norm_factor > 0 )
			variance_norm_factor = (int)sqrtf((float)variance_norm_factor);		// GPU Function
		else
			variance_norm_factor = 1;

		for( i = start_stage; i < cascade_n_stages; i++ )
			{

				stage_sum = 0;

				for( j = 0; j < stages_array_cuda[i]; j++ )
				{

					// the node threshold is multiplied by the standard deviation of the image 
					int t = tree_thresh_array_cuda[haar_counter] * variance_norm_factor;
					int sum;


					sum = (*(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index] + p_offset)
							 - *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 1] + p_offset)
							 - *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 2] + p_offset)
							 + *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 3] + p_offset))
						* weights_array_cuda[w_index];


					sum += (*(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index+4] + p_offset)
						- *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 5] + p_offset)
						- *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 6] + p_offset)
						+ *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 7] + p_offset))
						* weights_array_cuda[w_index + 1];

					if ((scaled_rectangles_array_cuda[r_index+8] != 0))
						sum += (*(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index+8] + p_offset)
							- *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 9] + p_offset)
							- *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 10] + p_offset)
							+ *(cascade_p0_cuda + scaled_rectangles_array_cuda[r_index + 11] + p_offset))
							* weights_array_cuda[w_index + 2];

					if(sum >= t)
						stage_sum += alpha2_array_cuda[haar_counter];
					else
						stage_sum += alpha1_array_cuda[haar_counter];

					

					//n_features++;
					haar_counter++;
					w_index+=3;
					r_index+=12;
				} // end of j loop 


				if( stage_sum < 0.4*stages_thresh_array_cuda[i] ){
				 	result_cuda[index] =  -i;
				 	end = true;

				 	break;
				} // end of the per-stage thresholding 
			} // end of i loop 

		if (!end)
			result_cuda[index] = 1;

	}

}