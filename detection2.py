from imageai.Detection import ObjectDetection
# from imageai.Detection.Custom import CustomObjectDetection
import os
# os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ ["CUDA_VISIBLE_DEVICES"] = ""
os.chdir('C:\python_work08052021') 
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "model_ex-005_acc-0.609375.h5"))#"model_ex-003_acc-0.718750.h5"
# detector.setModelPath( os.path.join(execution_path , "model_ex-003_acc-0.657895.h5_111111111111111111.h5"))#resnet50_coco_best_v2.1.0.h5
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
detections = detector.detectObjectsFromImage(input_image="C:\python_work08052021\image.jpg", output_image_path="C:\python_work08052021\imagenew.jpg", minimum_percentage_probability=30)


for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )