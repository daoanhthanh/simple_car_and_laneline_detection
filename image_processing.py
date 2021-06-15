import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

#Region of Interest
def ROI(image):
    height, width = image.shape
    # width = image.shape[1]
    polygon = np.array([
        [(0, height), (800, height), (380, 290)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    # return mask

def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #print(line) # detect the array (matrix) that define the line. in this case is 1x4 [[x1, y1, x2, y2]]
            # x1, y1,  x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image


def extract_and_save_frame(video):

    cam = cv2.VideoCapture(video)

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
            name = './data/frame' + str(currentframe) + '.jpg'
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

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # print(left_fit_average, "left")
    # print(right_fit_average, "right")
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*6/9)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])






image = cv2.imread('frame85.jpg')
lane_image = np.copy(image)
# grayscale  = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

canny_image = canny(lane_image)
cropped_image = ROI(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
lines_image = display_line(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.5, lines_image, 1, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)
#
# plt.imshow(grayscale)
# plt.show()

# cap = cv2.VideoCapture("test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped_image = ROI(canny_image)
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)
#     lines_image = display_line(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.5, lines_image, 1, 1)
#     cv2.imshow("result", combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
