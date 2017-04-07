#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

// Threshold for OTSU binarization algorithm
#define OTSU_THRESHOLD 0.6
// How to classify two areas as approximating each other
#define THRESHOLD_AREA_APPROXIMATION 0.8
// How to classify if density of CC is low in Close/Non-close table detection
#define CLOSE_NON_CLOSE_RULING_LINES_LOW_DENSITY_THRESHOLD 0.8
// How to classify two filled rectangles as close to each other
#define DISTANCE_SIZE_RATIO 0.1
// How to flag combined area of components within approach to area of the parent (table area)
#define CLOSE_NON_CLOSE_AREA_APPROACHING_THRESHOLD=0.5

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

void classifyTextNonText(const vector<ConnectedComponent> &connectedComponents, vector<ConnectedComponent> &textual,
                         vector<ConnectedComponent> &nonTextual,
                         vector<ConnectedComponent> &filled) {
    int connectedComponentsSize=connectedComponents.size();

    for (int i = 0; i < connectedComponentsSize; i++) {
        ConnectedComponent connectedComponent = connectedComponents[i];
        bool isTextual=true;
        bool isFilled=false;
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
                if ((area / ((float) connectedComponent.numPixels))>=THRESHOLD_AREA_APPROXIMATION) {
                    isTextual = false;
                    isFilled=true;
                }
            }
        }

        if(isTextual) {
            textual.push_back(connectedComponent);
        }
        else if(!isFilled) {
            nonTextual.push_back(connectedComponent);
        }
        else {
            filled.push_back(connectedComponent);
        }
    }
}
bool horizontalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.y + i.height, j.y + j.height) - std::max(i.y, j.y)) > 0;
}

bool verticalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.x + i.width, j.x + j.width) - std::max(i.x, j.x)) > 0;
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
            if (classifiedConnectedComponents[j])
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

void detectCloseNonCloseTables(const vector<ConnectedComponent> &textualComponents,
                               const vector<ConnectedComponent> &nonTextualComponents,
                               vector<ConnectedComponent> &tabularComponents) {
    int textualSize=textualComponents.size();
    int nonTextualSize=nonTextualComponents.size();
    tabularComponents.clear();
    for(int i=0;i<nonTextualSize;i++) {
        ConnectedComponent nonTextualComponent=nonTextualComponents[i];
        Rect parentConnectedComponentRect=nonTextualComponent.boundingBox;
        float ratio=nonTextualComponent.numPixels/((float)nonTextualComponent.boundingBoxSize);
        // 1. If density is not low
        if(ratio>CLOSE_NON_CLOSE_RULING_LINES_LOW_DENSITY_THRESHOLD) {
            continue;
        }
        // 3. Does not cut any textual or non-textual components
        bool cuts=false;
        {
            // Check if this box is cut by textual connected components
            for (int i = 0; i < textualComponents.size(); i++) {
                Rect rect = textualComponents[i].boundingBox;
                if (!((rect & parentConnectedComponentRect).area() == rect.area() ||
                      (rect | parentConnectedComponentRect).area() == (rect.area() + parentConnectedComponentRect.area()))) {
                    cuts = true;
                    break;
                }
            }
            if(!cuts) {
                // Check if this box is cut by non-textual connected components
                for (int i = 0; i < textualComponents.size(); i++) {
                    Rect rect = textualComponents[i].boundingBox;
                    if (!((rect & parentConnectedComponentRect).area() == rect.area() ||
                          (rect | parentConnectedComponentRect).area() ==
                          (rect.area() + parentConnectedComponentRect.area()))) {
                        cuts = true;
                        break;
                    }
                }
            }
        }
        if(cuts)
            continue;

        int numTextWithin=0;
        int numNonTextWithin=0;
        float areaTextual=0;
        float areaNonTextual=0;
        for (int j=0;j<textualComponents.size();j++) {
            if(i==j) {
                continue;
            }
            Rect textualRect=textualComponents[i].boundingBox;
            if((textualRect&parentConnectedComponentRect).area()==textualRect.area()) {
                numTextWithin++;
                areaTextual+=textualRect.area();
            }
        }

        for (int j=0;j<nonTextualComponents.size();j++) {
            if(i==j) {
                continue;
            }
            Rect nonTextualRect=nonTextualComponents[i].boundingBox;
            if((nonTextualRect&parentConnectedComponentRect).area()==nonTextualRect.area()) {
                numNonTextWithin++;
                areaNonTextual+=nonTextualRect.area();
            }
        }
        if(numTextWithin<2)
            continue;
        if((areaNonTextual+areaTextual/parentConnectedComponentRect.area())<CLOSE_NON_CLOSE_AREA_APPROACHING_THRESHOLD)
            continue;
        tabularComponents.push_back(nonTextualComponent);
    }
}

void detectParallelTables(const vector<ConnectedComponent> &textualComponents,
                               const vector<ConnectedComponent> &nonTextualComponents,
                               vector<ConnectedComponent> &tabularComponents) {

}

float inline euclideanDist(Point p, Point q) {
    Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

float distanceBetweenRectangles(Rect a, Rect b) {
    if((a&b).area()!=0)
        return 0;

    float x1=a.x;
    float y1=a.y;
    float x1b=a.x+a.width;
    float y1b=a.y+a.height;

    float x2=b.x;
    float y2=b.y;
    float x2b=b.x+b.width;
    float y2b=b.y+b.height;

    bool left = x2b < x1;
    bool right = x1b < x2;
    bool bottom = y2b < y1;
    bool top = y1b < y2;
    if(top && left)
        return euclideanDist(Point(x1, y1b), Point(x2b, y2));
    else if(left && bottom)
        return euclideanDist(Point(x1, y1), Point(x2b, y2b));
    else if(bottom && right)
        return euclideanDist(Point(x1b, y1), Point(x2, y2b));
    else if(right && top)
        return euclideanDist(Point(x1b, y1b), Point(x2, y2));
    else if(left)
        return x1 - x2b;
    else if (right)
        return x2 - x1b;
    else if (bottom)
        return y1 - y2b;
    else if (top)
        return y2 - y1b;

}

void detectColoredTable(const vector<ConnectedComponent> &filledConnectedComponents,
                        const vector<ConnectedComponent> &textualConnectedComponents,
                        const vector<ConnectedComponent> &nonTextualConnectedComponents, int width, int height) {
    int size = filledConnectedComponents.size();
    vector<vector<ConnectedComponent>> groupedConnectedComponents;
    vector<Rect> groupRects;
    vector<vector<ConnectedComponent>> tabularGroupedConnectedComponents;
    vector<bool> checkedConnectedComponents(size, false);
    for (int i = 0; i < size; i++) {
        ConnectedComponent connectedComponent = filledConnectedComponents[i];
        checkedConnectedComponents[i]=true;
        vector<ConnectedComponent>group;
        Rect groupBoundingRect=connectedComponent.boundingBox;
        for(int j=0;j<size;j++) {
            // If it has already been grouped (or if it is the same as ith)
            if(checkedConnectedComponents[i])
                continue;
            ConnectedComponent connectedComponent2=filledConnectedComponents[j];
            // Compute minimum distance between two rectangles
            float distance=distanceBetweenRectangles(connectedComponent.boundingBox,connectedComponent2.boundingBox);
            // Comute the min of size of two boxes
            float size=min(connectedComponent.boundingBoxSize,connectedComponent2.boundingBoxSize);
            // If two rectangles are close to each other, we can group them together
            if(distance/size<DISTANCE_SIZE_RATIO) {
                group.push_back(connectedComponent2);
                groupBoundingRect=groupBoundingRect|connectedComponent2.boundingBox;
            }
        }
        group.push_back(connectedComponent);
        groupRects.push_back(groupBoundingRect);
    }
    int numGroups = groupedConnectedComponents.size();
    vector<bool> areGroupsTabular(numGroups, false);
    for (int i = 0; i < numGroups; i++) {
        vector<ConnectedComponent> group = groupedConnectedComponents[i];
        int groupSize=group.size();
        vector<int> verticalProjections(width, 0);
        vector<int> horizontalProjections(height, 0);

        // Compute horizontal and vertical projections of bounding boxes
        for (int j = 0; j < groupSize; j++) {
            Rect box = group[j].boundingBox;
            for (int k = box.x; k < box.x + box.width; k++)
                verticalProjections[k] += box.height;
            for (int k = box.y; k < box.y + box.height; k++)
                verticalProjections[k] += box.width;
        }
        vector<Rect>columnRects;
        {
            bool currentStatus = false;
            int startPoint = -1;

            for (int i = 0; i < verticalProjections.size(); i++) {
                bool newStatus = verticalProjections[i] > 0;
                if (newStatus != currentStatus) {
                    if (newStatus == true) {
                        startPoint = i;
                    } else {
                        // push the rect here
                        int endPoint = i;
                        columnRects.push_back(Rect(startPoint, 0, endPoint - startPoint, height));
                    }
                    currentStatus = newStatus;
                }
            }
        }
        vector<Rect>rowRects;
        {
            bool currentStatus = false;
            int startPoint = -1;

            for (int i = 0; i < horizontalProjections.size(); i++) {
                bool newStatus = horizontalProjections[i] > 0;
                if (newStatus != currentStatus) {
                    if (newStatus == true) {
                        startPoint = i;
                    } else {
                        // push the rect here
                        int endPoint = i;
                        columnRects.push_back(Rect(startPoint, 0, endPoint - startPoint, width));
                    }
                    currentStatus = newStatus;
                }
            }
        }
        int numRows=rowRects.size();
        int numCols=columnRects.size();
        vector<Point2i>rowColNumbers(groupSize,Point2i(-1,-1));
        {
            for(int i=0;i<groupSize;i++) {
                Rect groupRect=group[i].boundingBox;
                int columnNumber=-1;
                int rowNumber=-1;
                for(int j=0;j<numRows;j++) {
                    if((groupRect&rowRects[j]).area()>0) {
                        rowNumber=j;
                    }
                }
                for(int j=0;j<numCols;j++) {
                    if((groupRect&columnRects[j]).area()>0) {
                        columnNumber=j;
                    }
                }
                assert(columnNumber!=0);
                assert(rowNumber!=0);
            }
        }
        bool isGroupATable=true;
        {
            for(int i=0;i<rowColNumbers.size();i++) {
                for(int j=0;j<rowColNumbers.size();j++) {
                    if(i==j)
                        continue;
                    if(rowColNumbers[i]==rowColNumbers[j]) {
                        isGroupATable=false;
                        break;
                    }
                }
            }
        }
        {
            Rect groupRect=groupRects[i];
            // Check if this box is cut by textual connected components
            for(int i=0;i<textualConnectedComponents.size();i++) {
                Rect rect=textualConnectedComponents[i].boundingBox;
                if(!((rect&groupRect).area()==rect.area()||(rect|groupRect).area()==(rect.area()+groupRect.area()))) {
                    isGroupATable=false;
                    break;
                }
            }
            // Check if this box is cut by non-textual connected components
            for(int i=0;i<nonTextualConnectedComponents.size();i++) {
                Rect rect=nonTextualConnectedComponents[i].boundingBox;
                if(!((rect&groupRect).area()==rect.area()||(rect|groupRect).area()==(rect.area()+groupRect.area()))) {
                    isGroupATable=false;
                    break;
                }
            }
        }

        if(isGroupATable) {
            tabularGroupedConnectedComponents.push_back(group);
        }
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