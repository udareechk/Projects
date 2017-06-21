int runCascadeClassifier( myCascade* _cascade, MyPoint pt, int start_stage )
{

	int p_offset, pq_offset;
	int i, j;
	unsigned int mean;
	unsigned int variance_norm_factor;
	int haar_counter = 0;
	int w_index = 0;
	int r_index = 0;
	int stage_sum;
	myCascade* cascade;
	cascade = _cascade;
	
	p_offset = pt.y * (cascade->sum.width) + pt.x;
	pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

	/**************************************************************************
	 * Image normalization
	 * mean is the mean of the pixels in the detection window
	 * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
	 * inv_window_area is 1 over the total number of pixels in the detection window
	 *************************************************************************/

	variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
	mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

	variance_norm_factor = (variance_norm_factor*cascade->inv_window_area);
	variance_norm_factor =  variance_norm_factor - mean*mean;

	/***********************************************
	 * Note:
	 * The int_sqrt is softwar integer squre root.
	 * GPU has hardware for floating squre root (sqrtf).
	 * In GPU, it is wise to convert the variance norm
	 * into floating point, and use HW sqrtf function.
	 * More info:
	 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
	 **********************************************/
	if( variance_norm_factor > 0 )
		variance_norm_factor = int_sqrt(variance_norm_factor);
	else
		variance_norm_factor = 1;

	/**************************************************
	 * The major computation happens here.
	 * For each scale in the image pyramid,
	 * and for each shifted step of the filter,
	 * send the shifted window through cascade filter.
	 *
	 * Note:
	 *
	 * Stages in the cascade filter are independent.
	 * However, a face can be rejected by any stage.
	 * Running stages in parallel delays the rejection,
	 * which induces unnecessary computation.
	 *
	 * Filters in the same stage are also independent,
	 * except that filter results need to be merged,
	 * and compared with a per-stage threshold.
	 *************************************************/
	for( i = start_stage; i < cascade->n_stages; i++ )
		{

			stage_sum = 0;

			for( j = 0; j < stages_array[i]; j++ )
			{

				/* the node threshold is multiplied by the standard deviation of the image */
				int t = tree_thresh_array[haar_counter] * variance_norm_factor;

				int sum = (*(scaled_rectangles_array[r_index] + p_offset)
						 - *(scaled_rectangles_array[r_index + 1] + p_offset)
						 - *(scaled_rectangles_array[r_index + 2] + p_offset)
						 + *(scaled_rectangles_array[r_index + 3] + p_offset))
					* weights_array[w_index];


				sum += (*(scaled_rectangles_array[r_index+4] + p_offset)
					- *(scaled_rectangles_array[r_index + 5] + p_offset)
					- *(scaled_rectangles_array[r_index + 6] + p_offset)
					+ *(scaled_rectangles_array[r_index + 7] + p_offset))
					* weights_array[w_index + 1];

				if ((scaled_rectangles_array[r_index+8] != NULL))
					sum += (*(scaled_rectangles_array[r_index+8] + p_offset)
						- *(scaled_rectangles_array[r_index + 9] + p_offset)
						- *(scaled_rectangles_array[r_index + 10] + p_offset)
						+ *(scaled_rectangles_array[r_index + 11] + p_offset))
						* weights_array[w_index + 2];

				if(sum >= t)
					stage_sum += alpha2_array[haar_counter];
				else
					stage_sum += alpha1_array[haar_counter];

				

				n_features++;
				haar_counter++;
				w_index+=3;
				r_index+=12;
			} /* end of j loop */

			/**************************************************************
			 * threshold of the stage. 
			 * If the sum is below the threshold, 
			 * no faces are detected, 
			 * and the search is abandoned at the i-th stage (-i).
			 * Otherwise, a face is detected (1)
			 **************************************************************/

			/* the number "0.4" is empirically chosen for 5kk73 */
			if( stage_sum < 0.4*stages_thresh_array[i] ){
			 	return -i;
			} /* end of the per-stage thresholding */
		} /* end of i loop */
	return 1;
}



void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{

	myCascade* cascade = _cascade;

	float factor = _factor;
	MyPoint p;
	int result;
	int y1, y2, x2, x, y, step;
	std::vector<MyRect> *vec = &_vec;

	MySize winSize0 = cascade->orig_window_size;
	MySize winSize;

	winSize.width =  myRound(winSize0.width*factor);
	winSize.height =  myRound(winSize0.height*factor);
	y1 = 0;

	/********************************************
	* When filter window shifts to image boarder,
	* some margin need to be kept
	*********************************************/
	y2 = sum_row - winSize0.height;
	x2 = sum_col - winSize0.width;

	/********************************************
	 * Step size of filter window shifting
	 * Reducing step makes program faster,
	 * but decreases quality of detection.
	 * example:
	 * step = factor > 2 ? 1 : 2;
	 * 
	 * For 5kk73, 
	 * the factor and step can be kept constant,
	 * unless you want to change input image.
	 *
	 * The step size is set to 1 for 5kk73,
	 * i.e., shift the filter window by 1 pixel.
	 *******************************************/ 
	step = 1;

	/**********************************************
	 * Shift the filter window over the image.
	 * Each shift step is independent.
	 * Shared data structure may limit parallelism.
	 *
	 * Some random hints (may or may not work):
	 * Split or duplicate data structure.
	 * Merge functions/loops to increase locality
	 * Tiling to increase computation-to-memory ratio
	 *********************************************/
	for( x = 0; x <= x2; x += step )
		for( y = y1; y <= y2; y += step )
		{
			p.x = x;
			p.y = y;

			/*********************************************
			 * Optimization Oppotunity:
			 * The same cascade filter is used each time
			 ********************************************/
			result = runCascadeClassifier( cascade, p, 0 );

			/*******************************************************
			 * If a face is detected,
			 * record the coordinates of the filter window
			 * the "push_back" function is from std:vec, more info:
			 * http://en.wikipedia.org/wiki/Sequence_container_(C++)
			 *
			 * Note that, if the filter runs on GPUs,
			 * the push_back operation is not possible on GPUs.
			 * The GPU may need to use a simpler data structure,
			 * e.g., an array, to store the coordinates of face,
			 * which can be later memcpy from GPU to CPU to do push_back
			 *******************************************************/
			if( result > 0 )
			{
				MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
				vec->push_back(r);
			}
		}
}

//---------------------------------------------------------------------------------------------------

void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{

	myCascade* cascade = _cascade;

	float factor = _factor;
	MyPoint p;
	int result;
	int y1, y2, x2, x, y, step;
	std::vector<MyRect> *vec = &_vec;

	MySize winSize0 = cascade->orig_window_size;
	MySize winSize;

	winSize.width =  myRound(winSize0.width*factor);
	winSize.height =  myRound(winSize0.height*factor);
	y1 = 0;

	y2 = sum_row - winSize0.height;
	x2 = sum_col - winSize0.width;

	step = 1;

	dim3 blocks(x2, y2);
	threadsPerBlock = 1;

	// GPU STUFF
	int* result = (int*)malloc(sizeof(int)*x2*y2);
	int* result_cuda, tree_thresh_array_cuda, scaled_rectangles_array_cuda, weights_array_cuda, alpha1_array_cuda, alpha2_array_cuda, stages_thresh_array_cuda;
	int* cascade_pq0_cuda, cascade_p0_cuda;

	int cascade_pq1 = cascade->pq1 - cascade->pq0;
	int cascade_pq2 = cascade->pq2 - cascade->pq0;
	int cascade_pq3 = cascade->pq3 - cascade->pq0;
	int cascade_p1 = cascade->p1 - cascade->p0;
	int cascade_p2 = cascade->p2 - cascade->p0;
	int cascade_p3 = cascade->p3 - cascade->p0;

	int imageSize = cascade->sum.width* cascade->sum.height; 

	cudaMalloc((void**) &result_cuda, sizeof(int)*x2*y2);
	cudaMalloc((void**) &tree_thresh_array_cuda, sizeof(int)*cascade->total_nodes*12);
	cudaMalloc((void**) &scaled_rectangles_array_cuda, sizeof(int)*cascade->total_nodes*12);
	cudaMalloc((void**) &weights_array_cuda, sizeof(int)*cascade->total_nodes*3);
	cudaMalloc((void**) &alpha1_array_cuda, sizeof(int)*cascade->total_nodes);
	cudaMalloc((void**) &alpha2_array_cuda, sizeof(int)*cascade->total_nodes);
	cudaMalloc((void**) &stages_thresh_array_cuda, sizeof(int)*cascade->n_stages);

	cudaMalloc((void**) &cascade_pq0_cuda, sizeof(int)*imageSize);
	cudaMalloc((void**) &cascade_p0_cuda, sizeof(int)*imageSize);


	int cascade_n_stages = cascade->n_stages, cascade_inv_window_area = cascade->inv_window_area, 
				cascade_sum_width = cascade->sum.width, cascade_sqsum_width = cascade->sqsum.width;

	cudaMemcpy(tree_thresh_array_cuda, tree_thresh_array, sizeof(int)cascade->total_nodes*12, cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_rectangles_array_cuda, scaled_rectangles_array, sizeof(int)cascade->total_nodes*12, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_array_cuda, weights_array, sizeof(int)cascade->total_nodes*3, cudaMemcpyHostToDevice);
	cudaMemcpy(alpha1_array_cuda, alpha1_array, sizeof(int)cascade->total_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(alpha2_array_cuda, alpha2_array, sizeof(int)cascade->total_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(stages_thresh_array_cuda, stages_thresh_array, sizeof(int)cascade->n_stages, cudaMemcpyHostToDevice);

	cudaMemcpy(cascade_pq0_cuda, cascade_pq0, sizeof(int)*imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(cascade_p0_cuda, cascade_p0, sizeof(int)*imageSize, cudaMemcpyHostToDevice);


	runCascadeClassifier<<<blocks, threadsPerBlock>>>(result_cuda, 0, cascade_n_stages, cascade_inv_window_area, cascade_sum_width, cascade_sqsum_width, 
									cascade_pq0_cuda, cascade_pq1, cascade_pq2, cascade_pq3, 
									cascade_p0_cuda, cascade_p1, cascade_p2, cascade_p3);

	cudaMemcpy(result, result_cuda, sizeof(int)*x2*y2, cudaMemcpyDeviceToHost);

	cudaFree(result_cuda);
	cudaFree(tree_thresh_array_cuda);
	cudaFree(scaled_rectangles_array_cuda);
	cudaFree(weights_array_cuda);
	cudaFree(alpha1_array_cuda);
	cudaFree(alpha2_array_cuda);
	cudaFree(stages_thresh_array_cuda);

	for (x = 0; i<=x2; x+=step){
		for (y = y1; y<=y2; y+=step){
			printf("%d ",result[x*x2+y]);
		}
	}
	printf("\n");

}


// GUDA Function
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

	// add offsets to the data array pointer to identify squares
	int* cascade_pq1_cuda = cascade_pq0_cuda + cascade_pq1;
	int* cascade_pq2_cuda = cascade_pq0_cuda + cascade_pq2;
	int* cascade_pq3_cuda = cascade_pq0_cuda + cascade_pq3;
	int* cascade_p1_cuda = cascade_p0_cuda + cascade_p1;
	int* cascade_p2_cuda = cascade_p0_cuda + cascade_p2;
	int* cascade_p3_cuda = cascade_p0_cuda + cascade_p3;

	// blockId to identify threads
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

		/***********************************************
	   * Note:
	   * The int_sqrt is softwar integer squre root.
	   * GPU has hardware for floating squre root (sqrtf).
	   * In GPU, it is wise to convert the variance norm
	   * into floating point, and use HW sqrtf function.
	   * More info:
	   * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
	   **********************************************/

		if( variance_norm_factor > 0 )
			variance_norm_factor = (int)sqrtf((float)variance_norm_factor);		// GPU Function
		else
			variance_norm_factor = 1;

		/**************************************************
		* The major computation happens here.
		* For each scale in the image pyramid,
		* and for each shifted step of the filter,
		* send the shifted window through cascade filter.
		*
		* Note:
		*
		* Stages in the cascade filter are independent.
		* However, a face can be rejected by any stage.
		* Running stages in parallel delays the rejection,
		* which induces unnecessary computation.
		*
		* Filters in the same stage are also independent,
		* except that filter results need to be merged,
		* and compared with a per-stage threshold.
		*************************************************/

		for( i = start_stage; i < cascade_n_stages; i++ )
			{
     			/****************************************************
		       * A shared variable that induces false dependency
		       * 
		       * To avoid it from limiting parallelism,
		       * we can duplicate it multiple times,
		       * e.g., using stage_sum_array[number_of_threads].
		       * Then threads only need to sync at the end
		       ***************************************************/

				stage_sum = 0;

				for( j = 0; j < stages_array_cuda[i]; j++ )
				{

					/**************************************************
			         * Send the shifted window to a haar filter.
			         **************************************************/

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

				/**************************************************************
		       * threshold of the stage. 
		       * If the sum is below the threshold, 
		       * no faces are detected, 
		       * and the search is abandoned at the i-th stage (-i).
		       * Otherwise, a face is detected (1)
		       **************************************************************/


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