/*
 *  Acceleration of face detection using Graphics Processing Units (GPU)
 *  University of Peradeniya
 *
 *  Name            :   gpu_functions.cuh
 *
 *  Author          :   Kesara Gamlath (kesaradhanushka@gmail.com)
 *
 *  Date            :   December 6, 2016
 *
 *  Function        :   CUDA Functions of IntegralImage and runCascadeClassifier function
 *
 *  History         :
 *      06-12-16    :   Initial version.
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


#define THREADS 16
#define THREADS2 256

__global__  void integralImageRows_GPU(int *sumData_cuda, int *sqsumData_cuda, unsigned char *data_cuda, int width, int height);

__global__  void integralImageCols_GPU(int *sumData_cuda, int *sqsumData_cuda, int width, int height);

__global__ void runCascadeClassifier(int* result_cuda, int start_stage, int cascade_n_stages, int cascade_inv_window_area, int cascade_sum_width, int cascade_sqsum_width, int x2, int y2,
									int* cascade_pq0_cuda, int cascade_pq1, int cascade_pq2, int cascade_pq3, 
									int* cascade_p0_cuda, int cascade_p1, int cascade_p2, int cascade_p3,
									int* tree_thresh_array_cuda, int* scaled_rectangles_array_cuda, int* weights_array_cuda, int* alpha1_array_cuda, int* alpha2_array_cuda, int* stages_thresh_array_cuda, int* stages_array_cuda);
