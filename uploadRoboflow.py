import roboflow

rf=roboflow.Roboflow(api_key="48TrqHbazjT2kcISiJpD")

project = rf.workspace().project("cube-color-gzmh4")
version = project.version(3)

version.deploy("yolov11", "./runs/detect/cube_training26/weights", "best.pt")
