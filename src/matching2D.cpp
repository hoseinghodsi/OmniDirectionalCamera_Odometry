#include <numeric>
#include <utility>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if (descriptorType.compare("DES_HOG"))
        {
            normType = cv::NORM_L2;
        }
        else if (descriptorType.compare("DES_BINARY"))
        {
            normType = cv::NORM_HAMMING;
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    double t = (double)cv::getTickCount();
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> tempMatches;
        matcher->knnMatch(descSource, descRef, tempMatches, 2);

        double minDescDistRatio = 0.8;
        for (auto iter = tempMatches.begin(); iter != tempMatches.end(); ++iter)
        {
            if (((*iter)[0].distance) < (*iter)[1].distance * minDescDistRatio)
            {
                matches.push_back((*iter)[0]);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << matches.size() << " number of matches "<< "matched in  " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        extractor = cv::BRISK::create();
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int nFeatures = 1000;
        float scaleFactor = 1.3f;
        int nLevels = 8;
        int edgeThreshold = 100;
        int firstLevel = 0;
        int WTA_K = 3;
        int patchSize = 100;
        int fastThreshold = 50;
        extractor = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);

    }
    else if (descriptorType.compare("FREAK") == 0)
    {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::SIFT::create();
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

}

// Detect keypoints in image using the traditional Shi-Thomasi detector
pair<int, double> detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    return {keypoints.size(), 1000*t/1.0};
}

pair<int, double> detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Assign detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;

    // Monitor time performance of Harris Keypoint detection
    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalizing/scaling the output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Filter the most important keypoints
    double maxOverlap = 0.0;
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            // We need to store only those keypoints with responses above the threshold
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            {
                cv::KeyPoint selectedKeyPoint;
                selectedKeyPoint.pt = cv::Point2f(i, j);
                selectedKeyPoint.size = 2 * apertureSize;
                selectedKeyPoint.response = response;

                // We need to perform non-maximum suppression (NMS) in local neighborhood to not
                // many keypoints very close together
                bool isOverlap = false;
                for (auto iter = keypoints.begin(); iter != keypoints.end(); ++iter)
                {
                    double keypointOverlap = cv::KeyPoint::overlap(selectedKeyPoint, *iter);
                    if (keypointOverlap > maxOverlap)
                    {
                        isOverlap = true;
                        if (selectedKeyPoint.response > (*iter).response)
                        {
                            *iter = selectedKeyPoint;
                            break;
                        }
                    }
                }
                if (!isOverlap)
                {
                    keypoints.push_back(selectedKeyPoint);
                }
            }
        }
    }
    // Finding the time elpased for Harris detector to find the keypoints
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return {keypoints.size(), 1000*t/1.0};
}

void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::Feature2D> detector;

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 60;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType.compare("BRISK") == 0)
    {

        int threshold = 70;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        detector = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nFeatures = 1000;
        float scaleFactor = 1.2f;
        int nLevels = 8;
        int edgeThreshold = 80;
        int firstLevel = 0;
        int WTA_K = 2;
        int patchSize = 80;
        int fastThreshold = 20;
        detector = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);

    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::SIFT::create();
    }

    // Finding the time elpased for any detector to find the keypoints
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoint Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    cout << "Keypoints detected: " << keypoints.size() << endl;
}
