#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
#include <iostream>
#include "ObjectTracker.h"

using namespace std;
CvPoint VertexOne,VertexThree;//長方形的左上點和右下點

void onMouse(int event,int x,int y,int flag,void* param){
	if(event==CV_EVENT_LBUTTONDOWN||event==CV_EVENT_RBUTTONDOWN){//得到左上角座標
		VertexOne=cvPoint(x,y);
		cout <<"set point one at " << x << " " << y << endl;
	}
	if(event==CV_EVENT_LBUTTONUP||event==CV_EVENT_RBUTTONUP){//得到右下角座標
		VertexThree=cvPoint(x,y);
		cout <<"set point three at " << x << " " << y << endl;
	}
}

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap("france3.avi");
	//VideoCapture cap("australia.avi");
	if ( !cap.isOpened() )  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		getchar();
		return -1;
	}
	cap.set(CV_CAP_PROP_POS_MSEC, 1050); //start the video at 300ms
	double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;
	namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	//namedWindow("MyVideo2",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	/*get initial pos by clicking mouse*/
	VertexOne=cvPoint(0,0);
	VertexThree=cvPoint(0,0);
	cvSetMouseCallback("MyVideo",onMouse);//設定滑鼠callback函式
	
	/*background substractor*/
// 	Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
// 	Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
// 	pMOG2 = new BackgroundSubtractorMOG2(10,196,true);
	
	Mat firstFrame;
	if(!cap.read(firstFrame))
	{
		cout << "Cannot read the frame from video file" << endl;
		return 0;
	}
// 	pMOG2->operator()(firstFrame, fgMaskMOG2,0.5);
// 	cap.read(firstFrame);
// 	pMOG2->operator()(firstFrame, fgMaskMOG2,0.5);
// 
// 	for(int i = 0; i < firstFrame.size().height; i++)
// 		for(int j = 0; j < firstFrame.size().width; j++){
// 			if(fgMaskMOG2.at<char>(i,j) == 0){
// 				firstFrame.at<Vec3b>(i,j) = Vec3b(0,0,0);
// 			}
// 		}

	CObjectTracker *m_pObjectTracker = new CObjectTracker(firstFrame.cols,firstFrame.rows,IMAGE_TYPE(0));
	
	/*get first object*/
	imshow("MyVideo", firstFrame); //show the frame in "MyVideo" window
	waitKey(0);
	cout << "object1 detected" << endl;
	int iCenterX = (VertexThree.x+VertexOne.x)/2;
	int iCentery = (VertexThree.y+VertexOne.y)/2;
	int iWidth = VertexThree.x-VertexOne.x;
	int iHeight = VertexThree.y-VertexOne.y;
	m_pObjectTracker->ObjectTrackerInitObjectParameters(iCenterX,iCentery,iWidth,iHeight);
	rectangle(firstFrame,VertexOne,VertexThree,cv::Scalar( 0, 0, 255 ));
	imshow("MyVideo", firstFrame);

	/*get second object*/
	waitKey(0);
	cout << "object2 detected" << endl;
	iCenterX = (VertexThree.x+VertexOne.x)/2;
	iCentery = (VertexThree.y+VertexOne.y)/2;
	iWidth = VertexThree.x-VertexOne.x;
	iHeight = VertexThree.y-VertexOne.y;
	m_pObjectTracker->ObjectTrackerInitObjectParameters(iCenterX,iCentery,iWidth,iHeight);
	rectangle(firstFrame,VertexOne,VertexThree,cv::Scalar( 0, 255, 0 ));
	imshow("MyVideo", firstFrame);
	waitKey(500);

	/*apply kalman filter*/
//	m_pObjectTracker->enableKalmanFilter(0);
//	m_pObjectTracker->enableKalmanFilter(1);

	int frame_adjust_for_kalman = 0;
	while(frame_adjust_for_kalman--)
	{
		m_pObjectTracker->ObjeckTrackerHandlerByUser(firstFrame.data);
	}
	
//	CvPoint tracker1Center_pre = m_pObjectTracker->getTrackerCenter(0);
//	CvPoint tracker1Center_current = tracker1Center_pre;
//	Mat trace_frame(firstFrame.cols,firstFrame.rows,CV_32FC3,Scalar(0,0,0));
	

	/*saving video frames*/
	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter oVideoWriter("D:/MyVideo.avi", CV_FOURCC('P','I','M','1'), fps, frameSize, true); //initialize the VideoWriter object 
	if ( !oVideoWriter.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}
	while(1)
	{
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video	
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
// 		pMOG2->operator()(frame, fgMaskMOG2,0.5);
// 		for(int i = 0; i < frame.size().height; i++)
// 			for(int j = 0; j < frame.size().width; j++){
// 				if(fgMaskMOG2.at<char>(i,j) == 0){
// 					frame.at<Vec3b>(i,j) = Vec3b(0,0,0);
// 				}
// 			}

		m_pObjectTracker->ObjeckTrackerHandlerByUser(frame.data);

//		tracker1Center_pre = tracker1Center_current;
//		tracker1Center_current = m_pObjectTracker->getTrackerCenter(0);
//		line(frame, tracker1Center_current, tracker1Center_pre, cv::Scalar( 0, 0, 255 ));
	

		oVideoWriter.write(frame); //writer the frame into the file
		imshow("MyVideo", frame); //show the frame in "MyVideo" window
		if(waitKey(20) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		//waitKey(0);
	}	
	delete m_pObjectTracker, m_pObjectTracker = 0;

	return 0;
}

