#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

void inspectModel(const string& name, const string& path) {
    cout << "\n=== Inspecting " << name << " (" << path << ") ===" << endl;
    try {
        Net net = readNet(path);
        
        vector<string> inputs = net.getUnconnectedOutLayersNames(); // Not exactly inputs but helpful
        
        // This is the correct way in recent OpenCV to get layer details
        vector<int> layerIds = net.getUnconnectedOutLayers();
        for (int id : layerIds) {
            Ptr<Layer> layer = net.getLayer(id);
            cout << "Output Layer: " << layer->name << " type: " << layer->type << endl;
        }

        // For inputs, we can check the first layer
        Ptr<Layer> first = net.getLayer(0);
        cout << "First Layer Name: " << first->name << " type: " << first->type << endl;

    } catch (const cv::Exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}

int main() {
    inspectModel("Palm Detection", "../models/palm_detection.onnx");
    inspectModel("Handpose Estimation", "../models/handpose_estimation.onnx");
    return 0;
}
