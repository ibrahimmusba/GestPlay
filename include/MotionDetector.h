#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

class MotionDetector
{
private:
    int numFrames; //Number of consecutive frames to track the motion

    //Will divide the image into blocks to find local motion
    int blockRows;  
    int blockCols;

    Mat prevFrame;
    Mat prevMean; //will be size blockRows x blockCols
    
    Mat findBlockMean(Mat frame);    

public:
    MotionDetector(int blockRows=4, int blockCols=4);
	
    //Initialize with numFrames - 1 frames so that when you give next frame, it has all the information to detect motion
    void init(Mat frame);
    
    void detectMotion(Mat frame);

};

