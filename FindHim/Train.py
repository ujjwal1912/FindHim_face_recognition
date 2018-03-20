from PIL import Image
import face_recognition
import pandas as pd
import numpy as np
import os
from glob import iglob
import glob


path_photo = '/home/ujjwal/Desktop/FindHim/V_raje/'
path_photos = '/home/ujjwal/Desktop/FindHim/V_raje/Photos/'
path_csv = '/home/ujjwal/Desktop/FindHim/V_raje/Vasundhra Raje/'

os.chdir(path_photos)
for filename in glob.glob('*.jpg'):
    image = face_recognition.load_image_file(filename, mode='RGB')
    name=[]
    face_locations = face_recognition.face_locations(image)
    
    for face_location in face_locations:

        top, right, bottom, left = face_location
    
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
        
        n = str(input("Enter the name of person: "))
        name.append(n)
    os.chdir(path_csv)   
    face_encoding = face_recognition.face_encodings(image)
    df = pd.DataFrame(face_encoding[0], columns = [name[0]])
    df.to_csv(name[0] + ".csv")
    os.chdir(path_photos)
