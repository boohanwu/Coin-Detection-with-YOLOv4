import cv2
import numpy as np

# Load YOLO
weights = "weights/yolov4-coin_last.weights"
config = "cfg/yolov4-coin.cfg"
classes = "classes/coin.names"

net = cv2.dnn.readNet(weights, config)

with open(classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Video capture
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    height, width, channels = img.shape     # default-> height:480, width:640
    window_scale = 1

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    # 篩選預測框及分類閥值，輸出層有三個，外迴圈跑三次
    # 內迴圈次數依據預測框的數量
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 第二次篩選預測框，參數依序為：預測框參數、存在物件信心度、信心度閥值及IoU閥值
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y-5), font, 2, color, 3)
            print("Class ID: %d, Object: %s, Confidence: %.2f, Center of X: %d, Center of Y: %d" % (class_ids[i], label, round(confidences[i], 2), center_x, center_y))

    frame = cv2.resize(img, (640 * window_scale, 480 * window_scale), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()