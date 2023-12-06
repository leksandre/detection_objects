from imageai.Detection import ObjectDetection
import os
os.chdir('C:\python_work08052021') 
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")#"model_ex-003_acc-0.718750.h5"
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 
         "image411.jpg"), output_image_path=os.path.join(execution_path , "imagenew411.jpg"))

for eachObject in detections:
    print(eachObject["name"] + " : " + eachObject["percentage_probability"] + " : " + eachObject["percentage_probability"] )