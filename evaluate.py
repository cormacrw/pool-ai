from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory='./assets/dataset')

metrics = trainer.evaluateModel(model_path="./assets/dataset/models/detection_model-ex-010--loss-0028.976.h5", json_path="./assets/dataset/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
print(metrics)