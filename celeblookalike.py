import face_recognition
import os
import numpy as np
from IPython.display import display
from PIL import Image

known_encodings = []
known_images = []

# def load_images(known_images_dir):
    
    
#     for file in os.listdir(known_images_dir):
#         #fsdecode function decode the file into filename
#         filename = os.fsdecode(file)
#         image = face_recognition.load_image_file(os.path.join(known_images_dir, filename))

#         enc = face_recognition.face_encodings(image)
#         if len(enc) > 0:
#             known_encodings.append(enc[0])
#             known_images.append(filename)

#     return (known_encodings, known_images)

def calculate_face_distance(known_encodings, unknown_img_path, cutoff=0.5, num_result=4):
    image_to_test = face_recognition.load_image_file(unknown_img_path)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
    
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
    return (unknown_img_path, known_images[face_distances.argmin()])

# def generateEncodings():
#     known_encodings, known_images = load_images("./images")

#     with open("encodings.txt", 'w') as fh:
#         for encoding in known_encodings:
#             for enc in encoding:
#                 fh.write(str(enc))
#                 fh.write(' ')
#             fh.write('\n')

#     with open("actors.txt", 'w') as fh:
#         for name in known_images:
#             fh.write(str(name))
#             fh.write('\n')

def loadEncodings():
    encs = []
    actors = []
    with open("encodings.txt",'r') as fh:
        lines = fh.readlines()
        for line in lines:
            encs.append(np.array([float(num) for num in line.split()]))
    
    with open("actors.txt", 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            actors.append(line[0:-1])
        
    return (encs, actors)



# if os.path.exists("encodings.txt") and os.path.exists("actors.txt"):
#     pass
# else:
#     generateEncodings()

known_encodings , known_images = loadEncodings()


myimage = "myimage.jpg"
matching_image = calculate_face_distance(known_encodings, myimage)[1]
print(matching_image)
img1 = Image.open("./images/" + matching_image)
img2 = Image.open(myimage)
display(img1,img2)