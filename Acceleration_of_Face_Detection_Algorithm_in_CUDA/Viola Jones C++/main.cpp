/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl); Hasindu Ramanayake (hawarama344@gmail.com)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *		06-12-16	: 	OpenCV avi decoder version
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

#include <stdio.h>
#include <stdlib.h>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"
#include <time.h>
 #include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define INPUT_FILENAME "Face.pgm"
#define OUTPUT_FILENAME "Output.pgm"
#define VIDWIDTH 1920
#define VIDHEIGHT 1080

using namespace std;
using namespace cv;


int main (int argc, char *argv[]){
//-------------------------------------------
	string filename = "test.avi";
    VideoCapture capture(filename);
    cv::Mat greyMat,frame;
     if( !capture.isOpened()){
        throw "Error when reading steam_avi";
    }
    //namedWindow( "w", 1);
//--------------------------------------------

    double totalTime = 0;
	int flag;
	
	int mode = 1;
	int i;

	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;


	printf("-- entering main function --\r\n");

	printf("-- loading image --\r\n");

	MyImage imageObj;
	MyImage *image = &imageObj;

	/*flag = readPgm((char *)"Face.pgm", image);
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}

	printf("-- loading cascade classifier --\r\n");*/



	image->width = VIDWIDTH;
	image->height = VIDHEIGHT;
	image->maxgrey = 255;
	image->data = (unsigned char*)malloc(sizeof(unsigned char)*(image->height*image->width));//new unsigned char[row*col];
	image->flag = 1;


	myCascade cascadeObj;
	myCascade *cascade = &cascadeObj;
	MySize minSize = {20, 20};
	MySize maxSize = {0, 0};

	cascade->n_stages=25;
	cascade->total_nodes=2913;
	cascade->orig_window_size.height = 24;
	cascade->orig_window_size.width = 24;

	std::vector<MyRect> result;


	//------------------------------------
		int frame_width=   image->width;
   		int frame_height=   image->height;
   		VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height),true);

   		if ( !video.isOpened() ){
			cout << "ERROR: Failed to write the video" << endl;
			return -1;
   		}
	//-------------------------------------

   	int iter = 0;

	for( ; ; ){
        capture >> frame;
        cv::cvtColor(frame, greyMat,CV_BGR2GRAY);
    //----------------------------------------------    
        for(int j=0;j<greyMat.rows;j++){
  			for (int i=0;i<greyMat.cols;i++){   
       			image->data[j*greyMat.cols + i] = greyMat.at<uchar>(j,i);
  			}
		}


		readTextClassifier();

		printf("-- detecting faces -- iteration: %d\r\n",iter);

		//the moment at which we should start measuring time
		clock_t start=clock();

		result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);

		//the moment at which we should stop measuring time
		clock_t stop=clock();

		//to find the time taken we must find the difference between the clock cycles and divide by
		//number of clock cycles per second
    	double cputime=(double)((stop-start)/(float)CLOCKS_PER_SEC);

    	totalTime = totalTime + cputime;

		for(i = 0; i < result.size(); i++ ){
			MyRect r = result[i];
			drawRectangle(image, r);
		}

		for(int j=0;j<greyMat.rows;j++){
  			for (int i=0;i<greyMat.cols;i++){   
       			greyMat.at<uchar>(j,i) = image->data[j*greyMat.cols + i];
       			//std::cout<< dataMat[j*greyMat.cols + i];
  			}
		}
		if(greyMat.empty())
            break;
        //imshow("w",greyMat);
        //std::cout<< greyMat;
        cv::cvtColor(greyMat,frame,CV_GRAY2BGR);
        video.write(frame);
     //   waitKey(5); // waits to display frame

        releaseTextClassifier();
        iter++;
 	}
    



	waitKey(0);
	/*printf("-- saving output --\r\n"); 
	flag = writePgm((char *)OUTPUT_FILENAME, image); 

	printf("-- image saved --\r\n");

	/* delete image and free classifier */
	freeImage(image);

	
	
	
    printf("Time taken for the operation is %1.5f seconds\n",totalTime);

	return 0;
}
