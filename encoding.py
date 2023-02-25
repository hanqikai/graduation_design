import os

from facenet_retinaface.retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface(1)

list_dir = os.listdir(r"facenet_retinaface\face_dataset")

image_paths = []
names = []
for name in list_dir:
    image_paths.append(r"facenet_retinaface\face_dataset/"+name)
    names.append(name.split("_")[0])

retinaface.encode_face_dataset(image_paths,names)
