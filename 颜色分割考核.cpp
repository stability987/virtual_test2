#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
int main() {
	Mat img = imread("ԭͼ.png");
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);
	Mat img_h, img_s, img_v;
	vector<Mat> channel;
	split(hsv, channel);
	img_h = channel[0];
	img_s = channel[1];
	img_v = channel[2];
	Mat mask_h, mask_s, mask_v;
	inRange(img_h, 150, 177,mask_h);
	inRange(img_s, 59, 245, mask_s);
	inRange(img_v, 61, 255, mask_v);
	Mat mask_h_s,mask,dst;
    bitwise_and(mask_h, mask_s, mask_h_s);
	bitwise_and(mask_h_s, mask_v, mask);
	bitwise_and(img, img, dst, mask);
	imshow("Ч��ͼ", dst);
	waitKey(0);
	return 0; 
}