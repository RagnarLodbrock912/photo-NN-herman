#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#define INPUT_IMAGE_PATH "input_images/"
#define OUTPUT_IMAGE_PATH "static/images/"

std::vector<cv::Mat> convolution(const std::vector<cv::Mat> &channels, const std::vector<std::vector<float>> &filter, float kernel=1.f) {
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    int height = channels[0].size().height - filterHeight + 1;
    int width = channels[0].size().width - filterWidth + 1;

    std::vector<cv::Mat> result(3);
    for (int i = 0; i < 3; i++) {
        result[i] = cv::Mat(height, width, CV_8UC1);
    }

    for (int layer = 0; layer < 3; layer++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float currentPixelResult = 0;

                for (int offsetH = 0; offsetH < filterHeight; offsetH++) {
                    for (int offsetW = 0; offsetW < filterWidth; offsetW++) {
                        currentPixelResult += filter[offsetH][offsetW] * channels[layer].at<uchar>(i + offsetH, j + offsetW);
                    }
                }

                int pixelValue = static_cast<int>(currentPixelResult / kernel);
                pixelValue = std::max(0, std::min(255, pixelValue));
                result[layer].at<uchar>(i, j) = static_cast<uchar>(pixelValue);
            }
        }
    }

    return result;
}



int main() {

    int filterSize;
    std::string imagePath;

    std::cin >> filterSize;

    std::vector<std::vector<float>> kernel(filterSize, std::vector<float>(filterSize, 0.0f));

    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            std::cin >> kernel[i][j];
        }
    }
    
    std::cin >> imagePath;

    cv::Mat image = cv::imread(INPUT_IMAGE_PATH + imagePath);

    if (image.empty()) {
        std::cerr << "Image error!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> channels;
    cv::split(image, channels);  // channels[0] = Blue, channels[1] = Green, channels[2] = Red

    std::vector<cv::Mat> res = convolution(channels, kernel);
    
    cv::Mat mergedImage;
    cv::merge(res, mergedImage);

    cv::imwrite(OUTPUT_IMAGE_PATH + imagePath, mergedImage);

    return 0;
}
