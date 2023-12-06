import os
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import pprint
from imageai.Detection import ObjectDetection
from datetime import datetime
os.chdir('C:\python_work08052021') 
probaility = 30

global lastImg
lastImg = ''

execution_path = os.getcwd()
now = datetime.now()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

detector.setModelPath( os.path.join(execution_path , 'resnet50_coco_best_v2.1.0.h5'))#model_ex-003_acc-0.718750.h5#resnet50_coco_03.h5
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
# detector.setModelPath( os.path.join(execution_path , "DenseNet-BC-121-32.h5"))
# detector.setModelPath( os.path.join(execution_path , "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
# detector.setModelPath( os.path.join(execution_path , "resnet50_imagenet_tf.2.0.h5"))
# detector.setModelPath( os.path.join(execution_path , "mobilenet_v2.h5"))
detector.loadModel()

def checkImg(pp=probaility,img1='',name1="image_1new"):
    return detector.detectObjectsFromImage( 
    input_image=os.path.join(execution_path , img1), # "вхдное" изображения для распознования
    output_image_path=os.path.join(execution_path , name1+now.strftime("%m_%d_%Y_%H_%M_%S")+".jpg"), # новое изображение которое является результатом обрабоки "входного изображения"
    minimum_percentage_probability=pp
    )

def drawImg(pp=probaility,img1='',name1="image_1newEdited",additionaltitle1='',additionalbox1=[]):
    global lastImg
    lastImg = name1+now.strftime("%m_%d_%Y_%H_%M_%S")+".jpg"
    return detector.detectObjectsFromImage( 
    input_image=os.path.join(execution_path , img1), # "вхдное" изображения для распознования
    output_image_path=os.path.join(execution_path , lastImg), # новое изображение которое является результатом обрабоки "входного изображения"
    minimum_percentage_probability=pp,
    additionaltitle=additionaltitle1,
    additionalbox=additionalbox1
    )

def multyspectr(detectRGB,detectlFN,probailityEtalon,num1):
    arr1=[]
    for objectIFN in detectlFN:
      if "percentage_probability" in objectIFN:
       if float(objectIFN["percentage_probability"]) < 40:
        box_IFN = objectIFN["detection_details"]
        # print("try to ", objectIFN)
        found = False
        probaility = probailityEtalon
        detect1 = detectRGB
        while (not found) and (probaility>7):
            for objecRGB in detect1:
             if 'detection_details' in objecRGB :
                if 'percentage_probability' not in objecRGB :
                 objecRGB["percentage_probability"] = 0
                box_RGB = objecRGB["detection_details"]
                if (abs(box_IFN[0]-box_RGB[0]))<75  and  (abs(box_IFN[1]-box_RGB[1]))<75 :
                    # print('seems',box_IFN[0],box_RGB[0],box_IFN[1],box_RGB[1],(abs(box_IFN[0]-box_RGB[0])),(abs(box_IFN[1]-box_RGB[1])))
                    newBox=objecRGB["detection_details"]
                    if 'detection_details' in objectIFN:
                     del objectIFN["detection_details"]
                    del objecRGB["detection_details"]
                    percentProbaility = (((float(objectIFN["percentage_probability"])/100*0.86)+(float(objecRGB["percentage_probability"])/100*0.14)) - ((float(objectIFN["percentage_probability"])/100*0.86)*(float(objecRGB["percentage_probability"])/100*0.14)))*100
                    print(objectIFN,newBox)
                    found = True
                    if probaility<probailityEtalon:
                        objectIFN["percentage_probability_new"] = percentProbaility
                        detect1 = drawImg(probailityEtalon,img1,str(num1)+"image_2newEDITED0_",str(objectIFN["name"]) +" "+str(round(percentProbaility, 3))+" (edited)", newBox)
                        print(str(objecRGB["name"])+' was approved with probability '+str(percentProbaility), objectIFN)
            if not found:
                probaility = probaility-2
                detect1 = checkImg(probaility,img1,str(num1)+"image_2newtemp")
       else:
           print(float(objectIFN["percentage_probability"]))
           arr1.append(objectIFN)
    if len(arr1)>0:
     drawImg(probailityEtalon,lastImg,str(num1)+"image_2newEDITED1_",str(arr1[0]["name"]) +" "+str(round(float(arr1[0]["percentage_probability"]), 3))+" (edited)", arr1[0]["detection_details"])

for a_number in [638,468,161,147,143,140,470,320]:
    number_str = str(a_number)
    zero_filled_number = number_str.zfill(5)
    print(zero_filled_number)
    img1 = r"D:\FLIR_ADAS_1_3.zip\FLIR_ADAS_1_3.zip\FLIR_ADAS_1_3\train\RGB\FLIR_"+str(zero_filled_number)+".jpg"
    img2 = r"D:\FLIR_ADAS_1_3.zip\FLIR_ADAS_1_3.zip\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_"+str(zero_filled_number)+".jpeg"

    if not os.path.isfile(img1):
        continue
    if not os.path.isfile(img2):
        continue

    image = Image.open(img1)
    new_image = image.resize((640, 512))
    img1 = "FLIR_tmp_.jpg"
    new_image.save(img1)
 
    # указываем необходимые для распознования классы объектов на тепловом и обычном изображениях
    # custom = detector.CustomObjects(car=True, motorcycle=True,  bus=True,   truck=True)
    rgb_array = []
    detections = checkImg(probaility,img1,str(zero_filled_number)+"image_1new_")

    print("RGB image")
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"])
        b = eachObject["detection_details"]
        print(("x1:"+str(b[0]), "y1:"+str(b[1])), ("x2:"+str(b[2]), "y2:"+str(b[3])))
        rgb_array.append(b)

    print("--------------------------------")
    ifn_array = []
    detections2 = checkImg(30,img2,str(zero_filled_number)+"image_2new_")

    print("IFN image")
    for eachObject in detections2:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"])
        b = eachObject["detection_details"]
        print(("x1:"+str(b[0]), "y1:"+str(b[1])), ("x2:"+str(b[2]), "y2:"+str(b[3])))
        ifn_array.append(b)

    print("--------------------------------")

    multyspectr(detections,detections2,probaility,str(zero_filled_number))
    detections_RGB = ''
    detections_IFN = ''


 