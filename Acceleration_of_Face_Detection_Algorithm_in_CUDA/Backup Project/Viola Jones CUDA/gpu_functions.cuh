#define THREADS 16
#define THREADS2 256

__global__  void integralImageRows_GPU(int *sumData_cuda, int *sqsumData_cuda, unsigned char *data_cuda, int width, int height);

__global__  void integralImageCols_GPU(int *sumData_cuda, int *sqsumData_cuda, int width, int height);

__global__ void runCascadeClassifier(int* result_cuda, int start_stage, int cascade_n_stages, int cascade_inv_window_area, int cascade_sum_width, int cascade_sqsum_width, int x2, int y2,
									int* cascade_pq0_cuda, int cascade_pq1, int cascade_pq2, int cascade_pq3, 
									int* cascade_p0_cuda, int cascade_p1, int cascade_p2, int cascade_p3,
									int* tree_thresh_array_cuda, int* scaled_rectangles_array_cuda, int* weights_array_cuda, int* alpha1_array_cuda, int* alpha2_array_cuda, int* stages_thresh_array_cuda, int* stages_array_cuda);
