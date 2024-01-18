from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# config.yaml фалй з шляхами до набору данних, епохи, назва моделі
results = model.train(data='config.yaml', epochs=10,name="PingPong_detector")

results = model.val()





