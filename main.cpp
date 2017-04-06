#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

// Threshold for OTSU binarizer
#define OTSU_THRESHOLD 0.6
// How to classify two areas as appoximating each other
#define THRESHOLD_AREA_APPROXIMATION 0.8

/**
 * This function computes connected components from an input binary image
 *
 * @param[in]     binary Input binary image with background labeled as 0 and foreground as 1.
 * @param[out]    blobs Output vector of connected components containing all pixels belonging to each component.
 * @param[out]    blob_rects Output vector of connected component bounding boxes.
 */
void connComps(const Mat &binary, vector<vector<Point2i> > &blobs, vector<Rect> &blob_rects) {
    blobs.clear();
    blob_rects.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S!

    int label_count = 2; // starts at 2 because 0,1 are used already

    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            if ((int) label_image.at<float>(y, x) != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x, y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0),
                          4);

            std::vector<cv::Point2i> blob;

            for (int i = rect.y; i < (rect.y + rect.height); i++) {
                for (int j = rect.x; j < (rect.x + rect.width); j++) {
                    if ((int) label_image.at<float>(i, j) != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j, i));
                }
            }
            blobs.push_back(blob);

            label_count++;
        }
    }

    for (size_t i = 0; i < blobs.size(); i++) {
        int top = 10000, bottom = -1, left = 10000, right = -1;
        for (size_t j = 0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
            if (x < left)
                left = x;
            if (x > right)
                right = x;
            if (y < top)
                top = y;
            if (y > bottom)
                bottom = y;
        }
        int w = right - left;
        int h = bottom - top;
        blob_rects.push_back(Rect(left, top, w, h));
    }
}

struct ConnectedComponent {
    Rect boundingBox;
    int height;
    int width;
    Point2i topLeft;
    Point2i bottomRight;
    int numPixels;
    int boundingBoxSize;
};

void findConnectedComponents(const Mat&imageBinary, vector<ConnectedComponent>&connectedComponents) {
    connectedComponents.clear();
    Mat imageBinary2=imageBinary.clone();

    vector<Rect>connectedComponentsRects;
    vector<vector<Point2i>>connectedComponentsBlobs;
    connComps(imageBinary2,connectedComponentsBlobs,connectedComponentsRects);

    int connectedComponentsSize=connectedComponentsBlobs.size();

    for(int i=0;i<connectedComponentsSize;i++) {
        Rect rect=connectedComponentsRects[i];
        vector<Point2i> blob=connectedComponentsBlobs[i];
        ConnectedComponent connectedComponent;
        connectedComponent.boundingBox=rect;
        connectedComponent.width=rect.width;
        connectedComponent.height=rect.height;
        connectedComponent.topLeft=rect.tl();
        connectedComponent.bottomRight=rect.br();
        connectedComponent.numPixels=blob.size();
        connectedComponent.boundingBoxSize=rect.area();

        connectedComponents.push_back(connectedComponent);
    }
}

void classifyTextNonText(const vector<ConnectedComponent>&connectedComponents) {
    vector<ConnectedComponent>textual;
    vector<ConnectedComponent>nonTextual;
    int connectedComponentsSize=connectedComponents.size();

    for (int i = 0; i < connectedComponentsSize; i++) {
        ConnectedComponent connectedComponent = connectedComponents[i];
        bool isTextual=true;
        // 1. If CC has low area (<6 pixels)
        if (connectedComponent.boundingBoxSize < 6) {
            isTextual=false;
        }
        // 3. The ratio of CCi’s area with the area of B(CCi) (the density of CCi) is too low (less than 6%)
        else if ((((float) connectedComponent.numPixels) / connectedComponent.boundingBoxSize) < 0.06) {
            isTextual=false;
        }
        else {
            float ratio = (connectedComponent.width / ((float) connectedComponent.height));
            // 4. The ratio of the width of B(CCi) and the height of B(CCi) is too low or too high (less than 0.06 or greater than 16)
            if (ratio < 0.06 || ratio > 15) {
                isTextual=false;
            }
            else {
                // 2. The bounding box of ith connected components contains many other bounding box (B(CCi) contains more than 3 B(CCj)).
                int numBoxesWithin = 0;
                int area = 0;
                for (int j = 0; j < connectedComponentsSize; j++) {
                    // If both connected components are same, skip
                    if (i == j)
                        continue;
                    ConnectedComponent connectedComponent2 = connectedComponents[j];
                    Rect checkRect = connectedComponent2.boundingBox;
                    // If checkRect is within the connected component
                    if ((connectedComponent.boundingBox & checkRect).area() == checkRect.area()) {
                        numBoxesWithin++;
                        area += connectedComponent2.numPixels;
                    }
                }
                // Add own area
                area += connectedComponent.numPixels;
                if (numBoxesWithin >= 3) {
                    isTextual = false;
                }
                // In an addition to find the color table candidates, the CCi is classified the
                // non-text element if CCi’s filled area is big and approximate
                // AB i .
                else if ((area / ((float) connectedComponent.numPixels))>=THRESHOLD_AREA_APPROXIMATION) {
                    isTextual = false;
                }
            }
        }

        if(isTextual) {
            textual.push_back(connectedComponent);
        }
        else {
            nonTextual.push_back(connectedComponent);
        }
    }
}
bool horizontalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.y + i.height, j.y + j.height) - std::max(i.y, j.y)) > 0;
}

void mergeTextualConnectedComponentsIntoLines(const vector<ConnectedComponent> &textualConnectedComponents) {
    int size = textualConnectedComponents.size();
    vector<bool> classifiedConnectedComponents(size, false);
    vector<vector<ConnectedComponent>> textLines;
    for (int i = 0; i < size; i++) {
        ConnectedComponent connectedComponent = textualConnectedComponents[i];
        vector<ConnectedComponent> textLine;
        classifiedConnectedComponents[i] = true;
        for (int j = 0; j < size; j++) {
            if (classifiedConnectedComponents[j] == true)
                continue;
            ConnectedComponent connectedComponent2 = textualConnectedComponents[i];
            if (!horizontalOverlap(connectedComponent.boundingBox, connectedComponent2.boundingBox))
                continue;
            bool d_ij = (connectedComponent2.topLeft.x - connectedComponent.boundingBox.x) > 0;
            int H_i = connectedComponent.height;
            int H_j = connectedComponent2.height;
            int maxH_ij = max(H_i, H_j);
            int minH_ij = min(H_i, H_j);

            int anded = maxH_ij & minH_ij;

            if (d_ij <= anded && anded >= 0.5 * maxH_ij) {
                textLine.push_back(connectedComponent2);
                classifiedConnectedComponents[j] = true;
            }
        }
        textLines.push_back(textLine);
    }
}

int main(int argc, char**argv) {
    if(argc!=2) {
        cout<<"Error in arguments";
        return -1;
    }
    // Read image in colored format
    string fileName=argv[1];
    Mat imageColored=imread(fileName,1);

    // Convert it to gray scale
    Mat imageGray;
    cvtColor(imageColored,imageGray,CV_BGR2GRAY);

    // Apply OTSU's binarization
    Mat imageBinary;
    cv::threshold(imageGray, imageBinary, OTSU_THRESHOLD, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Get connected components
    vector<ConnectedComponent>connectedCoponents;
    findConnectedComponents(imageBinary,connectedCoponents);

    return 0;
}