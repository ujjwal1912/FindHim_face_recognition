from PIL import Image
import face_recognition
import pandas as pd
import numpy as np
import os
from glob import iglob
image = face_recognition.load_image_file("detection_image.jpg")

name=[]
path_csv = '/home/ujjwal/Desktop/FindHim/csv/'

face_locations = face_recognition.face_locations(image)
for face_location in face_locations:

    top, right, bottom, left = face_location
    
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

    n = str(input("Enter the name of person: "))
    name.append(n)
length = len(face_locations)
print(length)
face_encoding = face_recognition.face_encodings(image)
for i in range(0, length):
    os.chdir(path_csv)
    df = pd.DataFrame(face_encoding[i], columns = [name[i]])
    df.to_csv(name[i] + ".csv")




    
