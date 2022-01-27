from imageai.Detection.Custom import DetectionModelTrainer
import json

f = open('config.json')

config = json.load(f)

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=config['data'])
trainer.setTrainConfig(object_names_array=["pool ball"], batch_size=4, num_experiments=200, train_from_pretrained_model=config['pretrained_model'])

trainer.trainModel()