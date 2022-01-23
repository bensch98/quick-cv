import sys
import os
import cv2 as cv
import numpy as np
from enum import Enum


class Mode(Enum):
  PREVIEW = 0
  CANNY = 1
  BLUR = 2
  COLOR_SPACE = 3
  CUSTOM = 4


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
    elif key == ord('p'):
      self.mode = Mode.PREVIEW
    elif key == ord('c'):
      self.mode = Mode.CANNY
    elif key == ord('b'):
      self.mode = Mode.BLUR
    elif key == ord('w'):
      self.mode = Mode.COLOR_SPACE
    elif key == ord('a'):
      self.mode = Mode.CUSTOM
    
  def process(self, frame):
    if self.mode == Mode.PREVIEW:
      frame = frame
    elif self.mode == Mode.CANNY:
      frame = cv.Canny(frame, 100, 150)
    elif self.mode == Mode.BLUR:
      frame = cv.blur(frame, (13, 13))
    elif self.mode == Mode.COLOR_SPACE:
      frame = self.color_space(frame)
    elif self.mode == Mode.CUSTOM:
      frame = self.color_space(frame)
      frame = cv.Canny(frame, 200, 210)
    if self.save:
      self.save = False
      cv.imwrite(f'{self.dest}/frame.jpg', frame)
    return frame

  def color_space(self, frame):
    # convert frame from BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower = np.array([160, 50, 50])
    upper = np.array([255, 255, 255])

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
