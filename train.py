from imageai.Detection.Custom import DetectionModelTrainer
import json
import sys

pretrain_model = sys.argv[2]
data_dir = sys.argv[1]

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_dir)
trainer.setTrainConfig(object_names_array=["pool ball"], batch_size=4, num_experiments=200, train_from_pretrained_model=pretrain_model)

trainer.trainModel()