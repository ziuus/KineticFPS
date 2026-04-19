import cv2
import numpy as np

def inspect(path, input_size):
    print(f"Inspecting: {path}")
    try:
        net = cv2.dnn.readNetFromONNX(path)
        blob = cv2.dnn.blobFromImage(np.zeros((input_size, input_size, 3), np.uint8), 1.0/255.0, (input_size, input_size))
        
        # Transpose NCHW -> NHWC
        blob_nhwc = np.transpose(blob, (0, 2, 3, 1))
        
        net.setInput(blob_nhwc)
        out_names = net.getUnconnectedOutLayersNames()
        outs = net.forward(out_names)
        
        for i, out in enumerate(outs):
            print(f"  Output {i} ({out_names[i]}): {out.shape}")
            # Print a few values to see range
            print(f"    Sample: {out.flatten()[:20]}")
            
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)

inspect("/home/zius/Projects/KineticFPS/models/palm_detection.onnx", 128)
inspect("/home/zius/Projects/KineticFPS/models/handpose_estimation.onnx", 224)
