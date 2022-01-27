from imageai.Detection.Custom import CustomObjectDetection
import os

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "assets/dataset/models/detection_model-ex-007--loss-0031.655.h5"))
detector.setJsonPath(os.path.join(execution_path, "assets/dataset/json/detection_config.json"))
detector.loadModel()
print(os.path.join(execution_path, 'testImg/p1.jpg'))
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'p3.jpg'), output_image_path=os.path.join(execution_path, "output.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
