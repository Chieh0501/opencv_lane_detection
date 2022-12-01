import cv2 as cv2
import numpy as np
#from vesc import VESC


def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)

    return img_canny


def region_of_interest(image):
    """Extract region of intereset from image.

    Parameters
    ----------
    image : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    bounds = np.array([[[0, 500], [0, 350], [650, 350], [650, 500]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def draw_lines(img, lines):
    """Superimpose lines on black image mask.

    Parameters
    ----------
    img : array-like
    lines : iterable of iterable of float

    Returns
    -------
    np.array
        Line mask
    """
    mask_lines = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)

    return mask_lines


def get_coordinates(line_parameters):
    """Compute image coordinates of line given slope and intercept.

    Parameters
    ----------
    line_parameters : iterable of float
        [slope, intercept] parameters of the line to locate.

    Returns
    -------
    list of int
        Coordinates of line given as [x1, y1, x2, y2]
    """

    slope, intercept = line_parameters
    y1 = 300
    y2 = 120

    x1 = (y1 - intercept) // slope
    x2 = (y2 - intercept) // slope
    return [x1, y1, x2, y2]


def compute_average_lines(lines):
    """Calculate average location over positive and negative slopes.

    Parameters
    ----------
    lines : iterable of 4-tuple of float
        Collection of line coordinates given of the form [x1, y1, x2, y2]

    Returns
    -------
    list of list of points
        List containing [[x1, y1, x2, y2], [x1, y1, x2, y2]] for negative
        and positive slope average lines, respectively.
    """
    left_lane_lines = []
    right_lane_lines = []
    left_weights = []
    right_weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Slope is 0
        if x2 == x1:
            continue

        slope, intercept = np.polyfit(
            (x1, x2), (y1, y2), 1
        )  # implementing polyfit to identify slope and intercept

        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < 0:
            left_lane_lines.append([slope, intercept])
            left_weights.append(length)
        else:
            right_lane_lines.append([slope, intercept])
            right_weights.append(length)

    # Computing average slope and intercept
    left_fit_points = get_coordinates(np.average(left_lane_lines, axis=0))
    right_fit_points = get_coordinates(np.average(right_lane_lines, axis=0))
    return [[left_fit_points], [right_fit_points]]


if __name__ == "__main__":
    # Set bounds for yellow line detection
    lower_yellow = np.array([20, 43, 100])
    upper_yellow = np.array([42, 255, 255])
    # Instantiate vesc
    #vesc_object = VESC("/dev/ttyACM0")

    cap = cv2.VideoCapture("output26.mp4")
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # cv2.imshow('Frame',frame)
            image = frame  # Reading the image file
            result = np.copy(image)
            cropped = cv2.resize(result, (650, 500))
            result = cv2.GaussianBlur(cropped, (9, 9), 0)
            result = region_of_interest(result)
            show_image("original", result)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            result = cv2.bitwise_and(result, result, mask=mask)
            lane_canny = find_canny(result, 100, 200)
            lane_lines = cv2.HoughLinesP(lane_canny, 1, np.pi / 180, 30, 40, 5)

            if lane_lines is not None:
              slope = 0
              count = 0
              for line in lane_lines:
                count += 1
                x1, y1, x2, y2 = line[0]
                try:
                  slope += (y1 - y2) / (x2 - x1)
                  if(np.abs((y1 - y2) / (x2 - x1)) < min_value):
                    min_value = np.abs((y1 - y2) / (x2 - x1))
                except ZeroDivisionError:
                  slope += 100
              show_image("lane_lines", draw_lines(result, lane_lines))
              slope /= count
              sign = 1 if slope > 0 else -1
              magnitude = np.clip(np.abs(slope), 0.25, 100)
              print("Slope: " + str(slope))
              print("Turn Dir: " + str(1 / (sign * magnitude * 4)))
              turn_dir = 1 / (sign * magnitude * 4)
            else:
              # Turn right
              turn_dir = 1

            #vesc_object.run(turn_dir, throttle=0.2)
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
