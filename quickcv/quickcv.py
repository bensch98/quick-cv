import sys
import os
import cv2 as cv
import numpy as np
from enum import Enum


class Mode(Enum):
  PREVIEW = 0
  CUSTOM = 1
  CANNY = 2
  BLUR = 3
  COLOR_SPACE = 4
  TRANSFORM = 5
  THRESHOLD = 6
  FILTERING = 7
  WATERSHED = 8
  FEATURE = 9
  CURVED = 10
  CONTRAST = 11


class FunctionHandler:
  def __init__(self, mode=Mode.PREVIEW):
    # attributes for saving screenshot
    self.save = False
    self.dest = os.path.abspath('.')

    # tracks current status
    self.mode = mode
    self.alive = True

  def select_function(self):
    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
      self.alive = False
    elif key == ord('s'):
      self.save = True
    elif key == ord('w'):
      self.mode = Mode.PREVIEW
    elif key == ord('a'):
      self.mode = Mode.CUSTOM
    elif key == ord('c'):
      self.mode = Mode.CANNY
    elif key == ord('e'):
      self.mode = Mode.BLUR
    elif key == ord('r'):
      self.mode = Mode.COLOR_SPACE
    elif key == ord('t'):
      self.mode = Mode.TRANSFORM
    elif key == ord('y'):
      self.mode = Mode.THRESHOLD
    elif key == ord('u'):
      self.mode = Mode.FILTERING
    elif key == ord('i'):
      self.mode = Mode.WATERSHED
    elif key == ord('o'):
      self.mode = Mode.FEATURE
    elif key == ord('p'):
      self.mode = Mode.CURVED
    elif key == ord('d'):
      self.mode = Mode.CONTRAST
    
  def process(self, frame):
    if self.mode == Mode.PREVIEW:
      frame = frame
    elif self.mode == Mode.CUSTOM:
      # kernels
      kernel1 = np.ones((15,15), np.uint8)
      kernel2 = np.ones((5,5), np.uint8)
      
      # convert
      frame = self.color_space(frame)
      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      # thresholding, morphological transformations
      ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
      #frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel1)
      #frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel2)
      #frame = cv.erode(frame, kernel2, iterations=1)

      x, y, w, h = cv.boundingRect(frame)
      
      # contour properties
      try:
        left = (x, np.argmax(frame[:,x]))
        right = (x+w-1, np.argmax(frame[:,x+w-1]))
        top = (np.argmax(frame[y,:]), y)
        bottom = (np.argmax(frame[y+hj-1,:]), y+h-1)

        cv.circle(frame, left, 4, (0, 50, 255), 2)
        cv.circle(frame, right, 4, (0, 255, 255), -1)
        cv.circle(frame, top, 4, (255, 50, 0), -1)
        cv.circle(frame, bottom, 4, (255, 255, 0), -1)
      except Exception as e:
        pass


    elif self.mode == Mode.CANNY:
      frame = cv.Canny(frame, 100, 150)
    elif self.mode == Mode.BLUR:
      frame = cv.blur(frame, (13, 13))
    elif self.mode == Mode.COLOR_SPACE:
      frame = self.color_space(frame)
    elif self.mode == Mode.TRANSFORM:
      frame = cv.resize(frame, None, fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    elif self.mode == Mode.THRESHOLD:
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      #ret, frame = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
      #frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
      #frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
      ret, frame = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif self.mode == Mode.FILTERING:
      #frame = cv.medianBlur(frame, 5)
      frame = cv.bilateralFilter(frame, 9, 75, 75)
    elif self.mode == Mode.WATERSHED:
      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      ret, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    elif self.mode == Mode.FEATURE:
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      sift = cv.SIFT_create()
      kp = sift.detect(gray, None)
      frame = cv.drawKeypoints(gray, kp, frame)
    elif self.mode == Mode.CURVED:
      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      edges = cv.Canny(frame, 150, 200, apertureSize=3)
      min_line_len = 1
      max_line_gap = 1
      lines = cv.HoughLinesP(edges, cv.HOUGH_PROBABILISTIC, np.pi/180, 90, min_line_len, min_line_len, max_line_gap)
      for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
          pts = np.array([[x1, y1], [x2, y2]], np.int32)
          cv.polylines(frame, [pts], True, (0, 255, 0))

    elif self.mode == Mode.CONTRAST:
      frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
      contrast = 2
      brightness = 1
      frame[:,:,2] = np.clip(contrast*frame[:,:,2]+brightness, 0, 255)
      frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
      cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)


    # log current mode to terminal
    print(self.mode.name)

    if self.save:
      self.save = False
      cv.imwrite(f'{self.dest}/frame.jpg', frame)
    return frame

  def color_space(self, frame):
    # convert frame from BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])

    # threshold the HSV frame to get only blue colors
    mask = cv.inRange(hsv, lower, upper)
    res = cv.bitwise_and(frame, frame, mask=mask)
    return res


def main():
  s = 0
  if len(sys.argv) > 1:
    s = sys.argv[1]

  source = cv.VideoCapture(s)

  win_name = 'Camera'
  cv.namedWindow(win_name, cv.WINDOW_NORMAL)

  fh = FunctionHandler()
  alive = True
  mode = Mode.PREVIEW

  while fh.alive:
    has_frame, frame = source.read()
    if not has_frame:
      break

    # image manipulation
    mode = fh.select_function()
    frame = fh.process(frame)
    
    # display image
    cv.imshow(win_name, frame)

  source.release()
  cv.destroyWindow(win_name)
  

if __name__ == '__main__':
  main()
