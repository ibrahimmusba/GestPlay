#include <iostream>
#include <fstream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <stdio.h>
#include <MotionDetector.h>

using namespace cv;
using namespace std;


int process(VideoCapture& capture) {
    int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
    Mat frame;
    
    bool firstFrame = true;
    MotionDetector motionDetector(4,4);
    
    for (;;) {
        capture >> frame;
        if (frame.empty())
            break;
        imshow(window_name, frame);
        
        if (firstFrame)
        {
            motionDetector.init(frame);
            firstFrame = false;
        }
        char key = (char)waitKey(5); //delay N millis, usually long enough to display and capture input
        switch (key) {
            case 'q':
            case 'Q':
            case 27: //escape key
                return 0;
            case ' ': //Save an image
                sprintf(filename,"filename%.3d.jpg",n++);
                imwrite(filename,frame);
                cout << "Saved " << filename << endl;
                break;
            default:
            break;
        }
    }
    return 0;
}



int main(int argc, char** argv) {

    if (argc != 2) {
        cout << "Not enough input arguments" << endl;
		return 1;
    }

    std::string arg = argv[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open a video device or video file!\n" << endl;
        return 1;
    }

    return process(capture);
}
