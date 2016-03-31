
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int imgnumber = 0;

string filename = "10x_10";
int th = 130; // threshold used. As the environment is a very closed environment, we may just set the threshold manually

int displayPicture(Mat image) {
  // create window
  std::stringstream name;
  name << "Image " << imgnumber++;
  string window_name = name.str();
  namedWindow(window_name, CV_WINDOW_NORMAL);
  imshow(window_name, image);
  return 0;
}

int writeToFile(Mat image) {
  ofstream outputFile;
  outputFile.open("output.txt");
  for (int i=0; i<image.rows; i++) {
    for (int j=0; j<image.cols; j++) {
      int value = image.at<uchar>(i,j);
      outputFile << value << ' ';
    }
    outputFile << endl;
  }
  return 0;
}

int writeToPNG(string filename, Mat image) {
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  imwrite(filename, image, compression_params);
}

Mat getImpurePart(Mat img) {
    // start counting the time
    double t = (double) getTickCount();

    // Load the template
    Mat templ = imread("template.jpg", 1);
    Mat bwtmpl = imread("template_bw.jpg", 1);

    // convert the images into grayscale
    cvtColor(img, img, CV_BGR2GRAY);
    cvtColor(templ, templ, CV_BGR2GRAY);
    cvtColor(bwtmpl, bwtmpl, CV_BGR2GRAY);

    int match_method = 1;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    Mat result;

    result.create(result_rows, result_cols, CV_32FC1);

    matchTemplate(img, templ, result, match_method);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    // localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    if (match_method==CV_TM_SQDIFF || match_method==CV_TM_SQDIFF_NORMED) {
      matchLoc = minLoc;
    } else {
      matchLoc = maxLoc;
    }

    // cropped image
    Mat src = Mat::zeros(templ.size(), templ.type());
    for (int i=0; i<templ.rows; i++) {
      for (int j=0; j<templ.cols; j++) {
        src.at<uchar>(i, j) = img.at<uchar>(matchLoc.y + i, matchLoc.x + j);
      }
    }

    // apply median filter
    int filtersize = 3;
    medianBlur(src, src, filtersize);

    // apply thresholding
    threshold(src, src, th, 255, 0);
    threshold(bwtmpl, bwtmpl, th, 255, 0);

    Mat noclean;
    // remove the die edge part
    bitwise_xor(bwtmpl, src, noclean);

    // clean the edge of the pad
    noclean = noclean & bwtmpl;

    // end of time counter
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "time needed: " << t << endl;

    return noclean;
}

int main(int, char** argv)
{
    // Load the image
    Mat img = imread("./original/"+ filename +".jpg", 1);

    Mat noclean = getImpurePart(img);

    // displayPicture(noclean);

    // open image of manual extraction
    Mat manimg = imread("./manual/" + filename + ".png", 1);
    cvtColor(manimg, manimg, CV_BGR2GRAY);
    threshold(manimg, manimg, th, 255, 0);

    int TP = 0; // true positive
    int TN = 0; // true negative
    int FP = 0; // false positive
    int FN = 0; // false negative

    // compare with manual extraction
    for (int i=0; i<manimg.rows; i++) {
        for (int j=0; j<manimg.cols; j++) {
            if (noclean.at<uchar>(i,j) == manimg.at<uchar>(i,j)) {
                if (noclean.at<uchar>(i,j) == 255) {
                    TP++;
                } else {
                    TN++;
                }
            } else {
                if (noclean.at<uchar>(i,j) == 255) {
                    FP++;
                } else {
                    FN++;
                }
            }
        }
    }

    int pembilang = TN + TP;
    int penyebut = TN + TP + FP + FN;
    cout << "accuracy: " << (float) (pembilang*100) / penyebut << endl;

    waitKey(0);
    return 0;
}
