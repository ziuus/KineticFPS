#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <linux/uinput.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// --- 1. Virtual Hardware Mouse ---
class VirtualMouse {
    int fd;
    bool is_pressed = false;
public:
    VirtualMouse() {
        fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        ioctl(fd, UI_SET_EVBIT, EV_KEY);
        ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
        ioctl(fd, UI_SET_EVBIT, EV_REL);
        ioctl(fd, UI_SET_RELBIT, REL_X);
        ioctl(fd, UI_SET_RELBIT, REL_Y);

        struct uinput_setup ussetup;
        memset(&ussetup, 0, sizeof(ussetup));
        ussetup.id.bustype = BUS_USB;
        ussetup.id.vendor  = 0x1234;
        ussetup.id.product = 0x5678;
        strcpy(ussetup.name, "KineticFPS AI Controller");
        ioctl(fd, UI_DEV_SETUP, &ussetup);
        ioctl(fd, UI_DEV_CREATE);
    }
    ~VirtualMouse() { ioctl(fd, UI_DEV_DESTROY); close(fd); }

    void move(int dx, int dy) {
        struct input_event ev[3];
        memset(&ev, 0, sizeof(ev));
        ev[0].type = EV_REL; ev[0].code = REL_X; ev[0].value = dx;
        ev[1].type = EV_REL; ev[1].code = REL_Y; ev[1].value = dy;
        ev[2].type = EV_SYN; ev[2].code = SYN_REPORT; ev[2].value = 0;
        write(fd, &ev, sizeof(ev));
    }

    void set_click(bool pressed) {
        if (pressed == is_pressed) return;
        struct input_event ev[2];
        memset(&ev, 0, sizeof(ev));
        ev[0].type = EV_KEY; ev[0].code = BTN_LEFT; ev[0].value = pressed ? 1 : 0;
        ev[1].type = EV_SYN; ev[1].code = SYN_REPORT; ev[1].value = 0;
        write(fd, &ev, sizeof(ev));
        is_pressed = pressed;
    }
};

// --- 2. Neural Kalman Engine ---
class NeuralKalman {
    KalmanFilter kf;
public:
    NeuralKalman() : kf(4, 2, 0) {
        kf.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(1e-3));
        setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    }
    Point2f update(Point2f m) {
        kf.predict();
        Mat mm = (Mat_<float>(2, 1) << m.x, m.y);
        Mat p = kf.correct(mm);
        return Point2f(p.at<float>(0), p.at<float>(1));
    }
};

int main() {
    cout << "[KineticFPS] Activating Full Gesture Control Engine..." << endl;
    Net palmNet = readNetFromONNX("../models/palm_detection.onnx");
    Net handNet = readNetFromONNX("../models/handpose_estimation.onnx");
    
    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    VirtualMouse mouse;
    NeuralKalman kalman;
    Mat frame, blob;
    Rect roi(170, 90, 300, 300); // Dynamic ROI
    Point2f smoothed_pt(0.5, 0.5);

    while (cap.read(frame)) {
        auto start = chrono::high_resolution_clock::now();
        flip(frame, frame, 1);
        int fw = frame.cols, fh = frame.rows;

        // --- Stage 1: Dynamic ROI (Palm Search) ---
        Mat pBlob = blobFromImage(frame, 1.0/255.0, Size(192, 192), Scalar(0,0,0), true, false);
        Mat pInput; transposeND(pBlob, {0, 2, 3, 1}, pInput);
        palmNet.setInput(pInput);
        vector<Mat> pOuts; palmNet.forward(pOuts, palmNet.getUnconnectedOutLayersNames());
        
        float* scores = (float*)pOuts[1].data;
        float maxS = 0; int maxI = 0;
        for(int i=0; i<pOuts[1].total(); i++) { if(scores[i] > maxS) { maxS = scores[i]; maxI = i; } }

        if (maxS > 0.5) {
            float* b = (float*)pOuts[0].data + (maxI * 18);
            float cx = b[0] * fw / 192.0f, cy = b[1] * fh / 192.0f;
            roi = Rect(cx - 150, cy - 150, 300, 300);
            roi &= Rect(0, 0, fw, fh);
        }

        // --- Stage 2: High-Precision Landmarks ---
        Mat crop = frame(roi);
        Mat lBlob = blobFromImage(crop, 1.0/255.0, Size(224, 224), Scalar(0,0,0), true, false);
        Mat lInput; transposeND(lBlob, {0, 2, 3, 1}, lInput);
        handNet.setInput(lInput);
        Mat land = handNet.forward("Identity");

        if (!land.empty()) {
            float* d = (float*)land.data;
            Point2f thumb_tip(d[4*3]/224.0f, d[4*3+1]/224.0f);
            Point2f index_tip(d[8*3]/224.0f, d[8*3+1]/224.0f);
            
            // 1. Cursor Movement
            Point2f raw((roi.x + index_tip.x*roi.width)/fw, (roi.y + index_tip.y*roi.height)/fh);
            Point2f pred = kalman.update(raw);
            float dx = (pred.x - smoothed_pt.x) * 2500.0f; // Multiplier for speed
            float dy = (pred.y - smoothed_pt.y) * 2500.0f;
            mouse.move((int)dx, (int)dy);
            smoothed_pt = pred;

            // 2. Pinch Detection (Click)
            float dist = sqrt(pow(thumb_tip.x - index_tip.x, 2) + pow(thumb_tip.y - index_tip.y, 2));
            mouse.set_click(dist < 0.08); // Threshold for click

            // 3. Visual Feedback
            rectangle(frame, roi, (dist < 0.08) ? Scalar(0, 255, 0) : Scalar(255, 255, 255), 2);
            for(int i=0; i<21; i++) {
                circle(frame, Point(roi.x + (d[i*3]/224.0f)*roi.width, roi.y + (d[i*3+1]/224.0f)*roi.height), 2, Scalar(104, 186, 127), -1);
            }
        }

        auto end = chrono::high_resolution_clock::now();
        cout << "\r[PERF] Latency: " << chrono::duration<double, milli>(end-start).count() << "ms    " << flush;
        imshow("KineticFPS Pro", frame);
        if (waitKey(1) == 27) break;
    }
    return 0;
}
