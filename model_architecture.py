import yolo_pytorch


def model():
	trained_model = yolo_pytorch.Yolov4(n_classes=80, inference=True)

	return trained_model
