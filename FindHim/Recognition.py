from PIL import Image
import face_recognition
import pandas as pd
import numpy as np
from glob import iglob
import glob
import os

known_faces = []

path_photo = '/home/ujjwal/Desktop/FindHim/V_raje/'
path_photos = '/home/ujjwal/Desktop/FindHim/V_raje/Photos/'
path_csv = '/home/ujjwal/Desktop/FindHim/V_raje/Vasundhra Raje/'

os.chdir(path_photo)
unknown_image = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


os.chdir(path_csv)
for f in glob.glob('*.csv'):
    df = pd.read_csv(f)
        
    df = df.values 
    df = df[:,1]
    known_faces.append(df)
    
#print(len(known_faces)) 
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
#print("Is the unknown face a picture of Vasundhara Raje? {}".format(results[1]))
#print(results)

if(results[1] == True):
    print(os.path.basename(os.path.dirname(path_csv)))
