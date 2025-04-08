import roboflow

rf=roboflow.Roboflow(api_key="48TrqHbazjT2kcISiJpD")

project = rf.workspace().project("cube-color-gzmh4")
version = project.version(4)

version.deploy("yolov11", "./yolov11_cube_640/cube_640_Training1/weights", "best.pt")
