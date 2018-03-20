# FindHim_face_recognition
  We have used Machine learning to train model by which we can recognize one's face in real time through any camera and get relevant information about the person. This all can be done in real time, so it could be very useful for Government and Security Agencies etc. It is built with dlib library with deep learning, it can give accuracy of 99%. It can easily be installed in web servers and we could monitor public areas and machine can track criminals by recognising his/her face through the cctv cameras without much human intervention. It is very useful for govt, as they have database of UIDAI/Bhamashah so they can train machine very easily using the UIDAI/Bhamashah dataset.   

#### Compatitbilty

It works best on Mac or Linux but it is complex to install deep learning algorithm on Windows.
  
#### Features

#### Find faces in pictures

First of all machine will detect all the faces in photo. The detection is done using HOG(Histograms of Orientation Gradients) algorithm. 
HOG algorithm mainly focuses on cetrtain points:
1) Divide the bigger image into small sub-images, called "cells". Cells can be rectangle(R-HOG) or circular(C-HOG).
2) Accumulate a histogram of edge orientations within that cell.
3) The combined histograms entries are used as the feature vector describing the object.



#### Training machine with the faces found
The training process works by looking at 3 face images at a time:

1)Load a training face image of a known person
2)Load another picture of the same known person
3)Load a picture of a totally different person


#### Find and manipulate facial features in pictures

Get the locations and outlines and landmark's of each person's face. Machine have a default encoding of a image of it's landmark such that whenever a face comes it compares the 68 landmarks of that face with each image inorder to detect basic outline of the face. For this landmark orientation check we use an algorithm called face landmark algorithm.



#### Encode Faces into different datasets
This is mainly done so that we can actually tell faces apart.The image had to be encoded into the 128-numpy array in which the difference between the gradients and every pixel has been calculated and stored for comparasion purposes.

We're here using face_recognition.api.face_encodings(face_image,known_face_locations=None,num_jitters=1) function inorder to generate the 128-dimensional encoding of the face.

Example of 128-dimensional numpy array:-
[-0.08309804  0.02348933  0.0607229   0.01153047 -0.02424719 -0.01670654
 -0.04477307 -0.0837856   0.19990423 -0.10318562  0.12239923 -0.01717109
 -0.12340292 -0.12571339 -0.03782629  0.03846696 -0.07939373 -0.21964854
 -0.00602084 -0.08481363 -0.03722906 -0.03338053  0.01902606  0.00179857
 -0.17573309 -0.38762856 -0.10837193 -0.14738029  0.06291072 -0.07739996
 -0.04934741  0.07634208 -0.15094358 -0.05313366 -0.01934127  0.12964743
  0.02487953  0.03426585  0.21913105  0.02249532 -0.15549859  0.0279968
  0.02143793  0.27614984  0.17839703  0.01842476  0.00402032 -0.00486307
  0.16943865 -0.18406537  0.12460586  0.14067864  0.1407124   0.05181955
  0.15462242 -0.0956174  -0.00831797  0.0252948  -0.14733472  0.08397162
 -0.00377735 -0.01776687 -0.01567785  0.00576194  0.16053182  0.1095928
 -0.00811611 -0.17199373  0.15258603 -0.18346339 -0.01501807  0.0807692
 -0.06326292 -0.09178369 -0.24002279  0.07721616  0.40855816  0.14531739
 -0.17034766  0.10293117 -0.07942596 -0.10028696  0.08073535 -0.00604103
 -0.09483838  0.05998469 -0.1329644  -0.02711085  0.17603001  0.05768422
 -0.07939439  0.06972981 -0.03196303  0.12235357  0.09665334  0.00545109
 -0.16192424  0.02866262 -0.1637383  -0.07017025  0.05850981 -0.01876711
  0.00065649  0.05198951 -0.12503907  0.16642889 -0.0027846  -0.01822686
 -0.02717516  0.11711621 -0.11585353 -0.01550356  0.13841577 -0.2504577
  0.16712104  0.06578305 -0.00182883  0.15413417  0.10752901  0.02148554
 -0.01913529  0.02641986 -0.09296616 -0.08992043 -0.01213597 -0.00850264
  0.06838152  0.07543926]


#### Recognising faces
First of all machine will detect the face in given unknown picture or video or frame then it will create the list of encoding(128 measurement array). Then the machile will compare the faces throughout our existing database and machine will fetch the relevent information about that unknown person.

face_recognition.compare_faces(known_faces, unknown_faces)   

Now this will return the list of True/False.

All we need to do is train a classifier that can take in the measurements from a new test image and tells which known person is the closest match. Running this classifier takes milliseconds. The result of the classifier is the name of the person!

So let’s try out our system. First, I trained a classifier with the embeddings of about 20 pictures each of Vasundhara Raje.Then I ran the classifier on image as well as on every frame of the video of Vasundhara Raje public speach.

It works! And look how well it works for faces in different poses — even sideways faces!


#### Speeding up Face Recognition

Face recognition can be done in parallel if you have a computer with
multiple CPU cores. For example if your system has 4 CPU cores, you can
process about 4 times as many images in the same amount of time by using
all your CPU cores in parallel.

If you are using Python 3.4 or newer, pass in a `--cpus <number_of_cpu_cores_to_use>` parameter:

```bash
$ face_recognition --cpus 4 ./pictures_of_people_i_know/ ./unknown_pictures/
```

You can also pass in `--cpus -1` to use all CPU cores in your system





