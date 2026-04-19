#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

void inspect(string path) {
    cout << "Inspecting: " << path << endl;
    Net net = readNetFromONNX(path);
    if (net.empty()) {
        cout << "Failed to load net" << endl;
        return;
    }
    
    vector<String> outLayerNames = net.getUnconnectedOutLayersNames();
    cout << "Outputs:" << endl;
    for (const auto& name : outLayerNames) {
        cout << "  - " << name << endl;
    }
    
    // Check first layer (usually input)
    vector<String> layerNames = net.getLayerNames();
    if (!layerNames.empty()) {
        cout << "First 5 layers:" << endl;
        for (int i = 0; i < min((int)layerNames.size(), 5); ++i) {
            cout << "  - " << layerNames[i] << endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        inspect(argv[1]);
    } else {
        inspect("../models/palm_detection.onnx");
        inspect("../models/handpose_estimation.onnx");
    }
    return 0;
}
