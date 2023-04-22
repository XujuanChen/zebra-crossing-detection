from ultralytics import YOLO

# load a model
model = YOLO('yolov8n.pt')

# use the model
results = model.train(data="cvlocal.yaml", epochs=10)

