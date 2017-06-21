/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "helpers.cuh"

/* include the gpu functions */
#include "gpu_functions.cuh"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int *scaled_rectangles_array;


int clock_counter = 0;
float n_features = 0;


int iter_counter = 0;

/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound( float value )
{
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
					 float scaleFactor, int minNeighbors)
{

	// cudaSetDevice(0);
	/* group overlaping windows */
	const float GROUP_EPS = 0.4f;
	/* pointer to input image */
	MyImage *img = _img;
	/***********************************
	 * create structs for images
	 * see haar.h for details 
	 * img1: normal image (unsigned char)
	 * sum1: integral image (int)
	 * sqsum1: square integral image (int)
	 **********************************/
	MyImage image1Obj;
	MyIntImage sum1Obj;
	MyIntImage sqsum1Obj;
	/* pointers for the created structs */
	MyImage *img1 = &image1Obj;
	MyIntImage *sum1 = &sum1Obj;
	MyIntImage *sqsum1 = &sqsum1Obj;

	/********************************************************
	 * allCandidates is the preliminaray face candidate,
	 * which will be refined later.
	 *
	 * std::vector is a sequential container 
	 * http://en.wikipedia.org/wiki/Sequence_container_(C++) 
	 *
	 * Each element of the std::vector is a "MyRect" struct 
	 * MyRect struct keeps the info of a rectangle (see haar.h)
	 * The rectangle contains one face candidate 
	 *****************************************************/
	std::vector<MyRect> allCandidates;

	/* scaling factor */
	float factor;

	/* maxSize */
	if( maxSize.height == 0 || maxSize.width == 0 )
		{
			maxSize.height = img->height;
			maxSize.width = img->width;
		}

	/* window size of the training set */
	MySize winSize0 = cascade->orig_window_size;

	/* malloc for img1: unsigned char */
	createImage(img->width, img->height, img1);
	/* malloc for sum1: unsigned char */
	createSumImage(img->width, img->height, sum1);
	/* malloc for sqsum1: unsigned char */
	createSumImage(img->width, img->height, sqsum1);

	/* initial scaling factor */
	factor = 1;

	/* iterate over the image pyramid */
	for( factor = 1; ; factor *= scaleFactor )
		{
			/* iteration counter */
			iter_counter++;

			/* size of the image scaled up */
			MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

			/* size of the image scaled down (from bigger to smaller) */
			MySize sz = { ( img->width/factor ), ( img->height/factor ) };

			/* difference between sizes of the scaled image and the original detection window */
			MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

			/* if the actual scaled image is smaller than the original detection window, break */
			if( sz1.width < 0 || sz1.height < 0 )
				break;

			/* if a minSize different from the original detection window is specified, continue to the next scaling */
			if( winSize.width < minSize.width || winSize.height < minSize.height )
				continue;

			/*************************************
			 * Set the width and height of 
			 * img1: normal image (unsigned char)
			 * sum1: integral image (int)
			 * sqsum1: squared integral image (int)
			 * see image.c for details
			 ************************************/
			setImage(sz.width, sz.height, img1);
			setSumImage(sz.width, sz.height, sum1);
			setSumImage(sz.width, sz.height, sqsum1);

			/***************************************
			 * Compute-intensive step:
			 * building image pyramid by downsampling
			 * downsampling using nearest neighbor
			 **************************************/
			nearestNeighbor(img, img1);

			/***************************************************
			 * Compute-intensive step:
			 * At each scale of the image pyramid,
			 * compute a new integral and squared integral image
			 ***************************************************/
			integralImages(img1, sum1, sqsum1);

			/* sets images for haar classifier cascade */
			/**************************************************
			 * Note:
			 * Summing pixels within a haar window is done by
			 * using four corners of the integral image:
			 * http://en.wikipedia.org/wiki/Summed_area_table
			 * 
			 * This function loads the four corners,
			 * but does not do compuation based on four coners.
			 * The computation is done next in ScaleImage_Invoker
			 *************************************************/
			setImageForCascadeClassifier( cascade, sum1, sqsum1);

			/* print out for each scale of the image pyramid */
			printf("detecting faces, iter := %d\n", iter_counter);

			/****************************************************
			 * Process the current scale with the cascaded fitler.
			 * The main computations are invoked by this function.
			 * Optimization oppurtunity:
			 * the same cascade filter is invoked each time
			 ***************************************************/
			ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
			 allCandidates);

		} /* end of the factor loop, finish all scales in pyramid*/

	if( minNeighbors != 0)
		{
			groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
		}

	freeImage(img1);
	freeSumImage(sum1);
	freeSumImage(sqsum1);
	return allCandidates;

}

void setImageForCascadeClassifier( myCascade* _cascade, MyIntImage* _sum, MyIntImage* _sqsum)
{
	MyIntImage *sum = _sum;
	MyIntImage *sqsum = _sqsum;
	myCascade* cascade = _cascade;
	int i, j, k;
	MyRect equRect;
	int r_index = 0;
	int w_index = 0;
	MyRect tr;

	cascade->sum = *sum;
	cascade->sqsum = *sqsum;

	equRect.x = equRect.y = 0;
	equRect.width = cascade->orig_window_size.width;
	equRect.height = cascade->orig_window_size.height;

	cascade->inv_window_area = equRect.width*equRect.height;

	cascade->p0 = (sum->data) ;
	cascade->p1 = (sum->data +  equRect.width - 1) ;
	cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
	cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
	cascade->pq0 = (sqsum->data);
	cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
	cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
	cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

	/****************************************
	 * Load the index of the four corners 
	 * of the filter rectangle
	 **************************************/

	/* loop over the number of stages */
	for( i = 0; i < cascade->n_stages; i++ )
	{
		/* loop over the number of haar features */
		for( j = 0; j < stages_array[i]; j++ )
		{
			int nr = 3;
			/* loop over the number of rectangles */
			for( k = 0; k < nr; k++ )
			{
				tr.x = rectangles_array[r_index + k*4];
				tr.width = rectangles_array[r_index + 2 + k*4];
				tr.y = rectangles_array[r_index + 1 + k*4];
				tr.height = rectangles_array[r_index + 3 + k*4];
				if (k < 2)
				{
					scaled_rectangles_array[r_index + k*4] = (sum->width*(tr.y ) + (tr.x )) ;
					scaled_rectangles_array[r_index + k*4 + 1] = (sum->width*(tr.y ) + (tr.x  + tr.width)) ;
					scaled_rectangles_array[r_index + k*4 + 2] = (sum->width*(tr.y  + tr.height) + (tr.x ));
					scaled_rectangles_array[r_index + k*4 + 3] = (sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
				}
				else
				{
					if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
					{
						scaled_rectangles_array[r_index + k*4] = 0 ;
						scaled_rectangles_array[r_index + k*4 + 1] = 0 ;
						scaled_rectangles_array[r_index + k*4 + 2] = 0;
						scaled_rectangles_array[r_index + k*4 + 3] = 0;
					}
					else
					{
						scaled_rectangles_array[r_index + k*4] = (sum->width*(tr.y ) + (tr.x )) ;
						scaled_rectangles_array[r_index + k*4 + 1] = (sum->width*(tr.y ) + (tr.x  + tr.width)) ;
						scaled_rectangles_array[r_index + k*4 + 2] = (sum->width*(tr.y  + tr.height) + (tr.x ));
						scaled_rectangles_array[r_index + k*4 + 3] = (sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
					}
				} /* end of branch if(k<2) */
			} /* end of k loop*/
			r_index+=12;
			w_index+=3;
		} /* end of j loop */
	} /* end i loop */
}

void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{

	myCascade* cascade = _cascade;

	float factor = _factor;
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

	// dim3 blocks(x2, y2);
	// int threadsPerBlock = 1;

	int blockx, blocky;
	if (x2>THREADS){
		blockx = ceil(x2/(float)THREADS);
	} else {
		blockx = 1;
	}

	if (y2>THREADS){
		blocky = ceil(y2/(float)THREADS);
	} else {
		blocky = 1;
	}

	dim3 blocks(blockx, blocky);
	dim3 threadsPerBlock (THREADS,THREADS);

	// GPU STUFF
	int* result = (int*)malloc(x2*y2*sizeof(int));
	int* result_cuda, *tree_thresh_array_cuda, *scaled_rectangles_array_cuda, *weights_array_cuda, *alpha1_array_cuda, *alpha2_array_cuda, *stages_thresh_array_cuda, *stages_array_cuda;
	int* cascade_pq0_cuda, *cascade_p0_cuda;

	int* cascade_pq0 = cascade->pq0;
	int* cascade_p0 = cascade->p0;

	int cascade_pq1 = cascade->pq1 - cascade->pq0;
	int cascade_pq2 = cascade->pq2 - cascade->pq0;
	int cascade_pq3 = cascade->pq3 - cascade->pq0;
	int cascade_p1 = cascade->p1 - cascade->p0;
	int cascade_p2 = cascade->p2 - cascade->p0;
	int cascade_p3 = cascade->p3 - cascade->p0;

	int imageSize = cascade->sum.width* cascade->sum.height; 

	// Malloc Device Memory 

	cudaMalloc((void**) &result_cuda, sizeof(int)*x2*y2); checkCudaError();
	cudaMalloc((void**) &tree_thresh_array_cuda, sizeof(int)*cascade->total_nodes*12);checkCudaError();
	cudaMalloc((void**) &scaled_rectangles_array_cuda, sizeof(int)*cascade->total_nodes*12);checkCudaError();
	cudaMalloc((void**) &weights_array_cuda, sizeof(int)*cascade->total_nodes*3);checkCudaError();
	cudaMalloc((void**) &alpha1_array_cuda, sizeof(int)*cascade->total_nodes);checkCudaError();
	cudaMalloc((void**) &alpha2_array_cuda, sizeof(int)*cascade->total_nodes);checkCudaError();
	cudaMalloc((void**) &stages_thresh_array_cuda, sizeof(int)*cascade->n_stages);checkCudaError();
	cudaMalloc((void**) &stages_array_cuda, sizeof(int)*cascade->n_stages);checkCudaError();

	cudaMalloc((void**) &cascade_pq0_cuda, sizeof(int)*imageSize);checkCudaError();
	cudaMalloc((void**) &cascade_p0_cuda, sizeof(int)*imageSize);checkCudaError();

	checkCudaError();

	int cascade_n_stages = cascade->n_stages, cascade_inv_window_area = cascade->inv_window_area, 
				cascade_sum_width = cascade->sum.width, cascade_sqsum_width = cascade->sqsum.width;

	cudaMemcpy(tree_thresh_array_cuda, tree_thresh_array, sizeof(int)*cascade->total_nodes*12, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(scaled_rectangles_array_cuda, scaled_rectangles_array, sizeof(int)*cascade->total_nodes*12, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(weights_array_cuda, weights_array, sizeof(int)*cascade->total_nodes*3, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(alpha1_array_cuda, alpha1_array, sizeof(int)*cascade->total_nodes, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(alpha2_array_cuda, alpha2_array, sizeof(int)*cascade->total_nodes, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(stages_thresh_array_cuda, stages_thresh_array, sizeof(int)*cascade->n_stages, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(stages_array_cuda, stages_array, sizeof(int)*cascade->n_stages, cudaMemcpyHostToDevice);checkCudaError();

	cudaMemcpy(cascade_pq0_cuda, cascade_pq0, sizeof(int)*imageSize, cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(cascade_p0_cuda, cascade_p0, sizeof(int)*imageSize, cudaMemcpyHostToDevice);checkCudaError();

	checkCudaError();


	runCascadeClassifier<<<blocks, threadsPerBlock>>>(result_cuda, 0, cascade_n_stages, cascade_inv_window_area, cascade_sum_width, cascade_sqsum_width, x2, y2,
									cascade_pq0_cuda, cascade_pq1, cascade_pq2, cascade_pq3, 
									cascade_p0_cuda, cascade_p1, cascade_p2, cascade_p3,
									tree_thresh_array_cuda, scaled_rectangles_array_cuda, weights_array_cuda, alpha1_array_cuda, alpha2_array_cuda, stages_thresh_array_cuda, stages_array_cuda);
	checkCudaError();

	cudaDeviceSynchronize();
	checkCudaError(); 

	cudaMemcpy(result, result_cuda, sizeof(int)*x2*y2, cudaMemcpyDeviceToHost);
	checkCudaError();

	cudaFree(result_cuda); checkCudaError();
	cudaFree(tree_thresh_array_cuda); checkCudaError();
	cudaFree(scaled_rectangles_array_cuda); checkCudaError();
	cudaFree(weights_array_cuda); checkCudaError();
	cudaFree(alpha1_array_cuda); checkCudaError();
	cudaFree(alpha2_array_cuda); checkCudaError();
	cudaFree(stages_thresh_array_cuda); checkCudaError();


	for( x = 0; x < x2; x += step ) {
    	for( y = y1; y < y2; y += step ) {

			if (result[y*x2+x] > 0)
			{
				MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
				vec->push_back(r);
				// printf("%d\n", result[y*x2+x]);
			}
		}
	}

}


/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/

  // GPU FUNCTION
  // USELESS
// void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum){

// 	int height = src->height;
// 	int width = src->width;

// 	int size = height*width;

// 	//arrays in main memory
// 	int* sumData = sum->data;
// 	int* sqsumData = sqsum->data;
// 	unsigned char * data = src->data;
	
// 	//pointers for arrays to be put on cuda memory
// 	int *sumData_cuda;
// 	int *sqsumData_cuda;
// 	unsigned char * data_cuda;

	
// 	//allocate memory in cuda device
// 	cudaMalloc((void **)&sumData_cuda,sizeof(int)*size);
// 	cudaMalloc((void **)&sqsumData_cuda,sizeof(int)*size);		
// 	cudaMalloc((void **)&data_cuda,sizeof(unsigned char)*size);
	
// 	//copy contents from main memory to cuda device memory
// 	cudaMemcpy(data_cuda, data, sizeof(unsigned char)*size, cudaMemcpyHostToDevice);
// 	checkCudaError();
	
// 	//thread configuration

// 	int blocks, threadsPerBlock;
	
// 	if (height>THREADS2){
// 		blocks = ceil(height/(float)THREADS2);
// 	} else {
// 		blocks = 1;

// 	}

// 	threadsPerBlock = THREADS2;

// 	//call the cuda kernel
// 	integralImageRows_GPU<<<blocks,threadsPerBlock>>>(sumData_cuda, sqsumData_cuda, data_cuda, width, height);
// 	cudaDeviceSynchronize();
// 	checkCudaError();

// 	if (width>THREADS2){
// 		blocks = ceil(width/THREADS2);
// 	} else {
// 		blocks = 1;
// 	}

// 	integralImageCols_GPU<<<blocks,threadsPerBlock>>>(sumData_cuda, sqsumData_cuda, width, height);

// 	cudaDeviceSynchronize();
// 	checkCudaError();
	
// 	//copy back the results from cuda memory to main memory
// 	cudaMemcpy(sumData, sumData_cuda, sizeof(int)*size, cudaMemcpyDeviceToHost);
// 	cudaMemcpy(sqsumData, sqsumData_cuda, sizeof(int)*size, cudaMemcpyDeviceToHost);
// 	checkCudaError();
// 	cudaFree(sumData_cuda);
// 	cudaFree(sqsumData_cuda);
// 	cudaFree(data_cuda);

// } 

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
      //* loop over the number of columns 
      for( x = 0; x < width; x ++)
      {
        it = data[y*width+x];
        //* sum of the current row (integer)
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



/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{

	int y;
	int j;
	int x;
	int i;
	unsigned char* t;
	unsigned char* p;
	int w1 = src->width;
	int h1 = src->height;
	int w2 = dst->width;
	int h2 = dst->height;

	int rat = 0;

	unsigned char* src_data = src->data;
	unsigned char* dst_data = dst->data;


	int x_ratio = (int)((w1<<16)/w2) +1;
	int y_ratio = (int)((h1<<16)/h2) +1;


	for (i=0;i<h2;i++)
	{
		t = dst_data + i*w2;
		y = ((i*y_ratio)>>16);
		p = src_data + y*w1;
		rat = 0;
		for (j=0;j<w2;j++)
		{
			x = (rat>>16);
			*t++ = p[x];
			rat += x_ratio;
		}
	}
}

void readTextClassifier()//(myCascade * cascade)
{
	/*number of stages of the cascade classifier*/
	int stages;
	/*total number of weak classifiers (one node each)*/
	int total_nodes = 0;
	int i, j, k, l;
	char mystring [12];
	int r_index = 0;
	int w_index = 0;
	int tree_index = 0;
	FILE *finfo = fopen("info.txt", "r");

	/**************************************************
	/* how many stages are in the cascaded filter? 
	/* the first line of info.txt is the number of stages 
	/* (in the 5kk73 example, there are 25 stages)
	**************************************************/
	if ( fgets (mystring , 12 , finfo) != NULL )
		{
			stages = atoi(mystring);
		}
	i = 0;

	stages_array = (int *)malloc(sizeof(int)*stages);

	/**************************************************
	 * how many filters in each stage? 
	 * They are specified in info.txt,
	 * starting from second line.
	 * (in the 5kk73 example, from line 2 to line 26)
	 *************************************************/
	while ( fgets (mystring , 12 , finfo) != NULL )
		{
			stages_array[i] = atoi(mystring);
			total_nodes += stages_array[i];
			i++;
		}

	fclose(finfo);


	/* TODO: use matrices where appropriate */
	/***********************************************
	 * Allocate a lot of array structures
	 * Note that, to increase parallelism,
	 * some arrays need to be splitted or duplicated
	 **********************************************/
	rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
	scaled_rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
	weights_array = (int *)malloc(sizeof(int)*total_nodes*3);
	alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
	alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
	tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
	stages_thresh_array = (int*)malloc(sizeof(int)*stages);
	FILE *fp = fopen("class.txt", "r");

	/******************************************
	 * Read the filter parameters in class.txt
	 *
	 * Each stage of the cascaded filter has:
	 * 18 parameter per filter x tilter per stage
	 * + 1 threshold per stage
	 *
	 * For example, in 5kk73, 
	 * the first stage has 9 filters,
	 * the first stage is specified using
	 * 18 * 9 + 1 = 163 parameters
	 * They are line 1 to 163 of class.txt
	 *
	 * The 18 parameters for each filter are:
	 * 1 to 4: coordinates of rectangle 1
	 * 5: weight of rectangle 1
	 * 6 to 9: coordinates of rectangle 2
	 * 10: weight of rectangle 2
	 * 11 to 14: coordinates of rectangle 3
	 * 15: weight of rectangle 3
	 * 16: threshold of the filter
	 * 17: alpha 1 of the filter
	 * 18: alpha 2 of the filter
	 ******************************************/

	/* loop over n of stages */
	for (i = 0; i < stages; i++)
		{    /* loop over n of trees */
			for (j = 0; j < stages_array[i]; j++)
				 {  /* loop over n of rectangular features */
					 for(k = 0; k < 3; k++)
						 {  /* loop over the n of vertices */
							 for (l = 0; l <4; l++)
								{
									if (fgets (mystring , 12 , fp) != NULL)
										rectangles_array[r_index] = atoi(mystring);
									else
										break;
									r_index++;
								} /* end of l loop */
								 if (fgets (mystring , 12 , fp) != NULL)
									{
										weights_array[w_index] = atoi(mystring);
										/* Shift value to avoid overflow in the haar evaluation */
										/*TODO: make more general */
										/*weights_array[w_index]>>=8; */
									}
								 else
									break;
							 w_index++;
						 } /* end of k loop */
						if (fgets (mystring , 12 , fp) != NULL)
							tree_thresh_array[tree_index]= atoi(mystring);
						else
							break;
						if (fgets (mystring , 12 , fp) != NULL)
							alpha1_array[tree_index]= atoi(mystring);
						else
							break;
						if (fgets (mystring , 12 , fp) != NULL)
							alpha2_array[tree_index]= atoi(mystring);
						else
							break;
						tree_index++;
						if (j == stages_array[i]-1)
							{
							 if (fgets (mystring , 12 , fp) != NULL)
								stages_thresh_array[i] = atoi(mystring);
							 else
								break;
							 }  
				 } /* end of j loop */
		} /* end of i loop */
	fclose(fp);
}


void releaseTextClassifier()
{
	free(stages_array);
	free(rectangles_array);
	free(scaled_rectangles_array);
	free(weights_array);
	free(tree_thresh_array);
	free(alpha1_array);
	free(alpha2_array);
	free(stages_thresh_array);
}
/* End of file. */
