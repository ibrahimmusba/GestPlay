#include <MotionDetector.h>

using namespace cv;
using namespace std;


MotionDetector::MotionDetector(int blockRows_, int blockCols_)
{
//	this.numFrames = numFrames;
	blockRows = blockRows_;
	blockCols = blockCols_;
}

void MotionDetector::init(Mat frame)
{
    prevFrame = frame;
    findBlockMean(prevFrame);
}

Mat MotionDetector::findBlockMean(Mat frame)
{
    int height = frame.rows;
    int width = frame.cols;
    
    Mat blockMeans = Mat(blockRows, blockCols, CV_32FC1);
    cout << "width = " << width << " height = " << height << endl;
    for (int y = 0; y < blockRows; y++)
    {
        for (int x = 0; x < blockCols; x++)
        {
            int startX = int(width/blockCols*x);
            int endX   = int(width/blockCols*(x+1))-1;
            
            int startY = int(height/blockRows*y);
            int endY   = int(height/blockRows*(y+1))-1;
            
//            cout << "Startx " << startX << ", EndX " << endX << endl;
//            cout << "StartY " << startY << ", EndY " << endY << endl;
            Rect roi = Rect(startX, startY, endX-startX, endY-startY);
            
            Mat frameRoi = frame(roi);
           
		   /*
		   	int i,j;
			uchar* p;
			for( i = 0; i < nRows; ++i)
			{
				p = I.ptr<uchar>(i);
				for ( j = 0; j < nCols; ++j)
				{
					p[j] = table[p[j]];
				}
			}
			*/
            CvScalar avg = mean(frameRoi);
            cout << "Mean: ";
            for (int i = 0; i < 3; i++)
            {
                cout << i <<": " << avg.val[i] << "\t";
            }
            cout << endl;
            
        }
    }
    return frame;
    
}

void MotionDetector::detectMotion(Mat frame)
{
    
    
}
