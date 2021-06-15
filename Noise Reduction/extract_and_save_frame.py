
# Importing all necessary libraries
import cv2
import os
from PIL import Image

# try:
#     os.makedirs('training')
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise

# Read the video from specified path
vid_name = input("Enter video path: ")
cam = cv2.VideoCapture(vid_name)

try:
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

# frame
currentframe = 0

while(True):

    # reading from frame
    ret,frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './add_noise/src/image/frames' + str(currentframe) + '.jpg'
        print ('Create and resizing...' + name)

        # writing the extracted images
        # img = Image.open("frame24.jpg")
        # basewidth = 700
        # wpercent = (basewidth / float(frame.size[0]))
        # hsize = int((float(frame.size[1]) * float(wpercent)))
        # frame_new = frame.resize((basewidth, hsize), Image.ANTIALIAS)
        #
        # frame_new.save(name)

        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1



    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
