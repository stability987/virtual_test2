#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// ������� 
Mat camera_matrix = (Mat_<double>(3, 3) << 993.9122, 0, 644.2905,
    0, 995.3944, 357.6626,
    0, 0, 1);
// �������
Mat dist_coeffs = (Mat_<double>(5, 1) << 0.1540, -0.4053, 0.0, 0.0, 0.0);

// �������꣺��ͼ���˳���Ӧ (����, ����, ����, ����)
// �ο���ߴ磺��102����102����94
Mat objectPoints = (Mat_<float>(4, 3) <<
    0, 0, 0,        // ���� (ԭ��)
    102, 0, 0,      // ���� (X�᷽��)
    102, 102, 0,    // ���� (X+Y�᷽��)
    0, 102, 0);     // ���� (Y�᷽��)

// 3D�����ᶨ�� (���ڲο�������)
Mat axis_3d = (Mat_<float>(4, 3) <<
    51, 51, 0,      // ԭ�� (�ο�������)
    51 + 51, 51, 0,   // X�� (��) - ����
    51, 51 + 51, 0,   // Y�� (��) - ��ǰ
    51, 51,-50);    // Z�� (��) - ����

// �����ת��ƽ������
Mat rvec, tvec;

// �Ľ����ı��ζ���������
void sortQuadrilateralPoints(vector<Point2f>& pts) {
    if (pts.size() != 4) {  
        return;
    }

    Point2f center(0, 0);
    for (const auto& p : pts) {
        center += p;
    }
  
    center.x /= 4.0f;  
    center.y /= 4.0f;

    
    sort(pts.begin(), pts.end(), [&center](const Point2f& a, const Point2f& b) {
        float angleA = atan2(a.y - center.y, a.x - center.x);
        float angleB = atan2(b.y - center.y, b.x - center.x);
        return angleA < angleB;
        });
}

int main() {

    VideoCapture cap(0);
    Mat src;
    if (!cap.isOpened()) {
        cout << "�����ʧ�ܣ����飺" << endl;
        return -1;
    }

    if (!cap.read(src)) {
        cout << "�޷���ȡ��һ֡��" << endl;
        return -1;
    }

    bool isColor = (src.type() == CV_8UC3);
    VideoWriter writer;
    int codec = writer.fourcc('M', 'J', 'P', 'G');
    double fps = 25.0;
    string filename = "./live.avi";
    if (!writer.open(filename, codec, fps, src.size(), isColor)) {
        cout << "��Ƶ�ļ�����ʧ�ܣ����·���Ƿ���Ȩ�ޡ�" << endl;
        return -1;
    }

    namedWindow("Live");
    namedWindow("dst");

    while (true) {
        if (!cap.read(src)) {
            cout << "֡��ȡʧ�ܣ��˳�����" << endl;
            break;
        }

        Mat hsv, dst, mask;
        cvtColor(src, hsv, COLOR_BGR2HSV);
      
        inRange(hsv, Scalar(110, 37, 151), Scalar(153, 153, 255), mask);
        src.copyTo(dst, mask);

       
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(dst, dst, MORPH_CLOSE, element, Point(-1, -1), 2);
        Mat finall_dst, finall_dst_1;
        GaussianBlur(dst, finall_dst, Size(7, 7), 0);
        Canny(finall_dst, finall_dst_1, 30, 150);

        vector<vector<Point>> contours;
        vector<Vec4i> hirearchy;
        findContours(finall_dst_1, contours, hirearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int max_contour_idx = -1;
        double max_contour_area = -1;
        for (int t = 0; t < contours.size(); t++) {
            double contour_area = contourArea(contours[t]);
            if (contour_area > 400 && contour_area > max_contour_area) {
                max_contour_area = contour_area;
                max_contour_idx = t;
            }
        }

        bool pnp_success = false;
        if (max_contour_idx != -1) {
            vector<Point> approx;
            double approx_accuracy = 0.03 * arcLength(contours[max_contour_idx], true);
            approxPolyDP(contours[max_contour_idx], approx, approx_accuracy, true);

            if (approx.size() == 4) { // ȷ�����ı���
                vector<Point2f> pts;
                for (int i = 0; i < 4; i++) {
                    pts.push_back(Point2f(approx[i].x, approx[i].y));
                }

               
                sortQuadrilateralPoints(pts);

                Mat imagePoints = (Mat_<float>(4, 2) <<
                    pts[0].x, pts[0].y,  // ����
                    pts[1].x, pts[1].y,  // ����
                    pts[2].x, pts[2].y,  // ����
                    pts[3].x, pts[3].y); // ����

                pnp_success = solvePnP(
                    objectPoints, imagePoints, camera_matrix, dist_coeffs,
                    rvec, tvec, false, SOLVEPNP_SQPNP
                );
            }
        }

        if (pnp_success) {
            // ����3D������
            Mat newImagePoint;
            projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs, newImagePoint);

            Point2i start_pt(cvRound(newImagePoint.at<Vec2f>(0)[0]), cvRound(newImagePoint.at<Vec2f>(0)[1]));

            //x
            Point2i x_end(cvRound(newImagePoint.at<Vec2f>(1)[0]), cvRound(newImagePoint.at<Vec2f>(1)[1]));
            line(src, start_pt, x_end, Scalar(0, 0, 255), 2);
          
            //y
            Point2i y_end(cvRound(newImagePoint.at<Vec2f>(2)[0]), cvRound(newImagePoint.at<Vec2f>(2)[1]));
            line(src, start_pt, y_end, Scalar(0, 255, 0), 2);
           
            //z
            Point2i z_end(cvRound(newImagePoint.at<Vec2f>(3)[0]), cvRound(newImagePoint.at<Vec2f>(3)[1]));
            line(src, start_pt, z_end, Scalar(255, 0, 0), 2);
            
        }

        writer.write(src);
        imshow("Live", src);
        imshow("dst", dst);

        char key = waitKey(40);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}
