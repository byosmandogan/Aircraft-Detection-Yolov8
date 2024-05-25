from ultralytics import YOLO

model = YOLO('best.pt').to("cuda")

sonuc = model.predict(source="MQ-9.mp4", show=True)
