#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (!matcherType.compare("MAT_BF"))
    {
        int normType;
        if(!descriptorType.compare("DES_BINARY"))
        {
            normType=cv::NORM_HAMMING;
        }
        else if(!descriptorType.compare("DES_HOG"))
        {
            normType=cv::NORM_L2;
        }
        else
        {
            throw invalid_argument("invalid descriptorType "+descriptorType);
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (!matcherType.compare("MAT_FLANN"))
    {
        if(!descriptorType.compare("DES_HOG"))
        {
            matcher=cv::FlannBasedMatcher::create();
        }
        else if(!descriptorType.compare("DES_BINARY"))
        {
            matcher=cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20,2));
        }
        else
        {
            throw invalid_argument("invalid descriptorType "+descriptorType);
        }
    }
    else
    {
        throw invalid_argument("invalid matcherType "+matcherType);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector< vector<cv::DMatch> > kmatches;
        matcher->knnMatch(descSource,descRef,kmatches,2);

        double minDistanceRatio=0.8;
        for(auto kmatch: kmatches)
        {
            if(kmatch.size()==2 && kmatch[0].distance<minDistanceRatio*kmatch[1].distance)
            {
                matches.push_back(kmatch[0]);
            }
        }
    }
    else
    {
        throw invalid_argument("invalid selectorType "+selectorType);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (!descriptorType.compare("BRISK"))
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(!descriptorType.compare("BRIEF"))
    {
        extractor=cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(!descriptorType.compare("ORB"))
    {
        extractor=cv::ORB::create();
    }
    else if(!descriptorType.compare("FREAK"))
    {
        extractor=cv::xfeatures2d::FREAK::create();
    }
    else if(!descriptorType.compare("AKAZE"))
    {
        extractor=cv::AKAZE::create();
    }
    else if(!descriptorType.compare("SIFT"))
    {
        extractor=cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
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
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t=(double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    if(bVis)
    {
        // visualize results
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }

    if(!keypoints.empty())
    {
        keypoints.clear();
    }
    double maxOverlap=0.0; //for non-maximum suppression
    for(int j=0;j<dst_norm.rows;++j)
    {
        for(int i=0;i<dst_norm.cols;++i)
        {
            int response=(int)dst_norm.at<float>(j,i);
            if(response>minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt=cv::Point2f(i,j);
                newKeyPoint.size=2*apertureSize;
                newKeyPoint.response=response;

                bool foundOverlap=false;
                for(auto it=keypoints.begin();it!=keypoints.end();++it)
                {
                    double kptOverlap=cv::KeyPoint::overlap(newKeyPoint,*it);
                    if(kptOverlap>maxOverlap){
                        foundOverlap=true;
                        if(newKeyPoint.response>it->response)
                        {
                            *it=newKeyPoint;
                            break;
                        }
                    }
                }
                if(!foundOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }
    t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
        cv::Mat visImage=dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName="Harris corner detection results";
        cv::namedWindow(windowName,5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
	double t=(double)cv::getTickCount();
    cv::Ptr<cv::FeatureDetector> detector;
    if(!detectorType.compare("FAST"))
    {
        detector=cv::FastFeatureDetector::create();
    }
    else if(!detectorType.compare("BRISK"))
    {
        detector=cv::BRISK::create();
    }
    else if(!detectorType.compare("ORB"))
    {
        detector=cv::ORB::create();
    }
    else if(!detectorType.compare("AKAZE"))
    {
        detector=cv::AKAZE::create();
    }
    else if(!detectorType.compare("SIFT"))
    {
        detector=cv::xfeatures2d::SIFT::create();
    }
    else{
        throw invalid_argument("invalid detectorType "+detectorType);
    }
    detector->detect(img,keypoints);
    t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
    cout << detectorType<<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if(bVis)
    {
        cv::Mat visImage=img.clone();
        cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName=detectorType+" keypoint detection results";
        cv::namedWindow(windowName,6);
        cv::imshow(windowName,visImage);
        cv::waitKey(0);
    }
}