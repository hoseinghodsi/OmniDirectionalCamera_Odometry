
#include <iostream>
#include <utility>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "data/frames/";
    string imgPrefix = "cap_img_"; // left camera, color
    string imgFileType = ".jpg";
    int imgStartIndex = 0;    // first file index 
    int imgEndIndex = 1275 // 2955 (original) or whatevet the last index is ;   // last file index to load
    int imgIndexWidth = 5;    // no. of digits which make up the file index 

    // Ring buffer implementation 
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;             // visualize results

    // Specify the detector and descriptor types
    string detectorType = "BRISK";
    string descriptor = "ORB";

    // Setting the matching parameters 
    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // Initialize camera relative position vector
    vector<vector<double>> cameraPosition;

    cv::Mat camPos(4, 1, cv::DataType<double>::type);
    camPos.at<double>(3, 0) = 1.00;
            
    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=15)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgIndexWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Display formatted table
        string imgLabel = imgPrefix + imgNumber.str();
        cout << endl;
        cout << "----------------------------------" << endl;
        cout << "Detector  : " << detectorType << endl;
        cout << "Descriptor: " << descriptor << endl;
        cout << "----------------------------------" << endl;
        cout << "Image     : " << imgLabel << endl;
        cout << endl;

        // load image from file 
        cv::Mat img, imgGray; 
        img = cv::imread(imgFullFilename);

        // Scale down the size of the image by 50%
        cv::Size newSize = cv::Size(img.cols/2, img.rows/2);
        cv::resize(img, img, newSize, 0, 0, cv::INTER_LINEAR); 

        // Convert the RGB image into grayscale
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // Record the size of final image
        int width = img.cols;
        int height = img.rows;

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        if (dataBuffer.size() > dataBufferSize) 
        {
            dataBuffer.erase(dataBuffer.begin());
        }

        /**************************/
        /* DETECT IMAGE KEYPOINTS */
        /**************************/
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;   // create empty feature list for current image

        pair<int, double> selectedDetector;
                
        if (detectorType.compare("SHITOMASI") == 0)
        {
            selectedDetector = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            selectedDetector = detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0 || detectorType.compare("BRISK") == 0 || 
                    detectorType.compare("ORG") == 0 || detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT") == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        
        // Limit the area from which keypoints can be detected
        bool limitedKeypoints = true;

        cv::Rect focusRect(width/10, 3*height/16, 4*width/5, 10*height/16);   // These seemingly custom frame is something I came up 
                                                                              // playing around with the detector engine. 

        vector<cv::KeyPoint>::iterator kp;
        vector<cv::KeyPoint> keypoints_ROI; 

        // Implement area limit on keypoints detection
        if (limitedKeypoints)
        {
            for (kp = keypoints.begin(); kp != keypoints.end(); ++kp)
            {
                if (focusRect.contains(kp->pt))
                {
                    cv::KeyPoint pickedKeyPoint;
                    pickedKeyPoint.pt = cv::Point2f(kp->pt);
                    pickedKeyPoint.size = 1;
                    keypoints_ROI.push_back(pickedKeyPoint);
                }
            }
        keypoints = keypoints_ROI;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        /********************************/
        /* EXTRACT KEYPOINT DESCRIPTORS */
        /********************************/
        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptor);
        
        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        double match_time = 0;
        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {   
            vector<double> instantaneousPosition;

            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                            (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                            matches, descriptorType, matcherType, selectorType);                                    

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, cv::WINDOW_NORMAL);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }

            // Convert matched keypoints into Point2f 
            vector<cv::Point2f> ps1, ps2;
            for (vector<cv::DMatch>::const_iterator it = matches.begin(); it!= matches.end(); ++it) 
            {
                ps1.push_back((dataBuffer.end() - 2)->keypoints[it->queryIdx].pt);
                ps2.push_back((dataBuffer.end() - 1)->keypoints[it->trainIdx].pt);
            }

            // The camera matix - I simply assumed an identity matrix for now 
            cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
            cameraMatrix.at<double>(0,0) = 1.00; cameraMatrix.at<double>(0,1) = 0.00; cameraMatrix.at<double>(0,2) = 0.00;// cameraMatrix.at<double>(0,3) = 0.00;
            cameraMatrix.at<double>(1,0) = 0.00; cameraMatrix.at<double>(1,1) = 1.00; cameraMatrix.at<double>(1,2) = 0.00;// cameraMatrix.at<double>(1,3) = 0.00;
            cameraMatrix.at<double>(2,0) = 0.00; cameraMatrix.at<double>(2,1) = 0.00; cameraMatrix.at<double>(2,2) = 1.00;// cameraMatrix.at<double>(2,3) = 0.00;

            // Find the essential matrix between two consequtive images
            cv::Mat inliers;
            cv::Mat essentialMatrix = cv::findEssentialMat(ps1, ps2, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliers);
            
            // Extract relative camera pose (rotation and translation matrices) from the essential matrix
            cv::Mat rotation, translation;
            cv::recoverPose(essentialMatrix, ps1, ps2, cameraMatrix, rotation, translation, inliers);

            // Determine the camera position relative to previous image using R and T 
            cv::Mat C; // Camera matrix 
            cv::hconcat(rotation, translation, C);
            cv::Mat lowerRow(1, 4, cv::DataType<double>::type);
            lowerRow.at<double>(0, 3) = 1.00;
            cv::vconcat(C, lowerRow, C); // Camera matrix constructed with the size of 4x4

            // Update the last know camera pose with the Camera matrix
            camPos = C * camPos;

            // Store instantaneous camera position in a vector
            instantaneousPosition.push_back(camPos.at<double>(0, 0));
            instantaneousPosition.push_back(camPos.at<double>(1, 0));
            instantaneousPosition.push_back(camPos.at<double>(2, 0));

            // Construct 2D vector of camera position for all the images
            cameraPosition.push_back(instantaneousPosition);

            // Output camera position
            cout << endl;
            cout << "Camera Position:  " << endl;
            cout << instantaneousPosition[0] << endl;
            cout << instantaneousPosition[1] << endl;
            cout << instantaneousPosition[2] << endl;
            cout << endl;
        }
    }

    return 0;
}
