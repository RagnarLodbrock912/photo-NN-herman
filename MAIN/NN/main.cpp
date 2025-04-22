#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

int main() {
    int width, height;
    cin >> width >> height;

    vector<vector<int>> R(height, vector<int>(width));
    vector<vector<int>> G(height, vector<int>(width));
    vector<vector<int>> B(height, vector<int>(width));

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            cin >> R[i][j];

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            cin >> G[i][j];

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            cin >> B[i][j];

    // Создаём OpenCV-матрицу с 3 каналами (BGR)
    Mat image(height, width, CV_8UC3);

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j) {
            image.at<Vec3b>(i, j)[0] = static_cast<uchar>(B[i][j]);
            image.at<Vec3b>(i, j)[1] = static_cast<uchar>(G[i][j]);
            image.at<Vec3b>(i, j)[2] = static_cast<uchar>(R[i][j]);
        }

    imwrite("output.png", image);

    return 0;
}
