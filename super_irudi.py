#!/usr/bin/python3

"""
'Super Irudi' command line image processing software.

Copyright (c) 2022, Joxean Koret

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import time

import cv2
import PIL
import requests

import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from skimage import data, exposure
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.interpolate import UnivariateSpline

Image.MAX_IMAGE_PIXELS = None

from histogram_matching import (match_histograms, match_histograms_with,
                                load_histogram_from_file)

#-------------------------------------------------------------------------------
DEBUG = False
VERSION = "1.1"
USER_AGENT = 'SuperIrudi/%s (https://github.com/joxeankoret/super-irudi)' % VERSION

#-------------------------------------------------------------------------------
def log(msg):
  print("[%s] %s" % (time.asctime(), msg))

#-------------------------------------------------------------------------------
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

#-------------------------------------------------------------------------------
def open_file(image_file):
  global USER_AGENT
  tmp = image_file.lower()
  if tmp.startswith("http://") or tmp.startswith("https://"):
    log("Downloading URL %s" % image_file)
    headers = {'User-Agent': USER_AGENT}
    response = requests.get(image_file, headers=headers)
    if response.status_code == 200:
      img = Image.open(BytesIO(response.content))
    else:
      print(response)
      log("Error opening remote image: %s" % response.reason)
      sys.exit(2)
  else:
    img = Image.open(image_file)
  return img

#-------------------------------------------------------------------------------
class CSuperIrudiTool:
  def __init__(self):
    self.img = None
    self.original = None
    self.interactive = False
    self.quality = 95
    self.stitch_list = []
    self.stitch_type = 2

    self.nrows = 1
    self.ncols = 2

  def save_image(self, out_file):
    self.img.save(out_file, quality=self.quality)

  def show(self, prev=None, cmap="viridis"):
    fig, (ax1, ax2) = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(8, 2),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2):
        aa.set_axis_off()

    if prev is not None:
      ax1.imshow(prev, cmap=cmap)
      ax1.set_title('Previous')
    else:
      ax1.imshow(self.original, cmap=cmap)
      ax1.set_title('Original')

    ax2.imshow(self.img, cmap=cmap)
    ax2.set_title('Current')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    if 'window' in dir(mng): mng.resize(*mng.window.maxsize())
    plt.show()

  def image_apply_histogram(self, image, histogram_file, out_file):
    log("Applying histogram file %s" % histogram_file)
    histogram = load_histogram_from_file(histogram_file)
    matched = match_histograms_with(np.asarray(image), histogram, multichannel=True)
    self.img = Image.fromarray(matched)
    if out_file is not None:
      self.img.save(out_file)

    if self.interactive:
      log("Displaying final image")
      fig, (ax1, ax2) = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(8, 3),
                                     sharex=True, sharey=True)
      for aa in (ax1, ax2):
          aa.set_axis_off()

      ax1.imshow(image)
      ax1.set_title('Source')
      ax2.imshow(matched)
      ax2.set_title('Matched')

      plt.tight_layout()
      mng = plt.get_current_fig_manager()
      if 'window' in dir(mng): mng.resize(*mng.window.maxsize())
      plt.show()

  def image_match_histograms(self, image, reference, out_file):
    log("Matching histograms")
    rgbimg1 = Image.new("RGB", image.size)
    rgbimg1.paste(image)
    log("Source image's size %s" % str(image.size))

    rgbimg2 = Image.new("RGB", reference.size)
    rgbimg2.paste(reference)
    log("Reference's image size %s" % str(reference.size))

    matched = match_histograms(np.asarray(rgbimg1), np.asarray(rgbimg2), multichannel=True)
    self.img = Image.fromarray(matched)
    if out_file is not None:
      self.img.save(out_file)

    if self.interactive:
      log("Displaying final image")

      if self.ncols == 2 and self.nrows == 1:
        nrows = 3
        ncols = 1
      else:
        nrows = self.nrows
        ncols = self.ncols

      fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 3),
                                          sharex=True, sharey=True)
      for aa in (ax1, ax2, ax3):
          aa.set_axis_off()

      ax1.imshow(image)
      ax1.set_title('Source')
      ax2.imshow(reference.resize([image.width, image.height], Image.Resampling.LANCZOS))
      ax2.set_title('Reference')
      ax3.imshow(matched)
      ax3.set_title('Matched histogram')

      plt.tight_layout()
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())
      plt.show()

  def apply_filter(self, the_filter):
    prev = self.img
    self.img = self.img.filter(the_filter)
    if self.interactive:
      self.show(prev)
  
  def box_blur(self):
    prev = self.img
    self.img = self.img.filter(the_filter)
    if self.interactive:
      self.show(prev)
  
  def sharpness(self, value):
    prev = self.img
    self.img = ImageEnhance.Sharpness(self.img).enhance(value)
    if self.interactive:
      self.show(prev)
  
  def color(self, value):
    prev = self.img
    self.img = ImageEnhance.Color(self.img).enhance(value)
    if self.interactive:
      self.show(prev)

  def contrast(self, value):
    prev = self.img
    self.img = ImageEnhance.Contrast(self.img).enhance(value)
    if self.interactive:
      self.show(prev)

  def brightness(self, value):
    prev = self.img
    self.img = ImageEnhance.Brightness(self.img).enhance(value)
    if self.interactive:
      self.show(prev = self.img)
  
  def autocontrast(self):
    prev = self.img
    self.img = ImageOps.autocontrast(self.img)
    if self.interactive:
      self.show(prev)

  def equalize(self):
    prev = self.img
    self.img = ImageOps.equalize(self.img)
    if self.interactive:
      self.show(prev)
  
  def flip(self):
    prev = self.img
    self.img = ImageOps.flip(self.img)
    if self.interactive:
      self.show(prev)

  def invert(self):
    prev = self.img
    self.img = ImageOps.invert(self.img)
    if self.interactive:
      self.show(prev)

  def mirror(self):
    prev = self.img
    self.img = ImageOps.mirror(self.img)
    if self.interactive:
      self.show(prev)

  def posterize(self, val):
    prev = self.img
    self.img = ImageOps.posterize(self.img, int(val))
    if self.interactive:
      self.show(prev)

  def solarize(self, val):
    prev = self.img
    self.img = ImageOps.solarize(self.img, int(val))
    if self.interactive:
      self.show(prev)
  
  def pencil(self):
    prev = self.img
    img = np.asarray(self.img)
    _, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    self.img = Image.fromarray(sk_color)
    if self.interactive:
      self.show(prev)

  def hdr(self):
    prev = self.img
    img = np.asarray(self.img)
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    self.img = Image.fromarray(hdr)
    if self.interactive:
      self.show(prev)

  ##############################################################################
  # The following filters were published by Prateek Majumder on this blog:
  #
  # https://www.analyticsvidhya.com/blog/2021/07/an-interesting-opencv-application-creating-filters-like-instagram-and-picsart/
  def get_seasons_filter_data(self, img):
    inc_lookup_table = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    dec_lookup_table = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue, green, red  = cv2.split(img)
    return blue, green, red, inc_lookup_table, dec_lookup_table

  def apply_summer_winter(self, is_winter):
    prev = self.img
    img = np.asarray(self.img)
    vals = self.get_seasons_filter_data(img)
    blue, green, red, inc_lookup_table, dec_lookup_table = vals
    if is_winter:
      red = cv2.LUT(red, inc_lookup_table).astype(np.uint8)
      blue = cv2.LUT(blue, dec_lookup_table).astype(np.uint8)
    else:
      red = cv2.LUT(red, dec_lookup_table).astype(np.uint8)
      blue = cv2.LUT(blue, inc_lookup_table).astype(np.uint8)

    img = cv2.merge((blue, green, red ))
    self.img = Image.fromarray(img)
    if self.interactive:
      self.show(prev)
  #
  ##############################################################################

  def deoldify(self, url):
    """
    Connect to a DeOldify instance to colorize the current image.
    """
    log("Uploading image to DeOldify instance")
    byte_io = BytesIO()
    self.img.save(byte_io, 'png')
    byte_io.seek(0)
    d = {'file': ('1.png', byte_io, 'image/png')}
    r = requests.post(url, files=d)
    if r.status_code != 200:
      log("Remote error: %s" % r.reason)
      return

    prev = self.img
    self.img = Image.open(BytesIO(r.content))
    if self.interactive:
      self.show(prev)
    
  def auto_balance(self):
    """
    Automatically adjust white balance.
    """
    img = np.asarray(self.img)
    balanced_img = np.zeros_like(img)
    for i in range(3):
      hist, bins = np.histogram(img[..., i].ravel(), 256, (0, 256))
      bmin = np.min(np.where(hist>(hist.sum()*0.0005)))
      bmax = np.max(np.where(hist>(hist.sum()*0.0005)))
      balanced_img[...,i] = np.clip(img[...,i], bmin, bmax)
      balanced_img[...,i] = (balanced_img[...,i]-bmin) / (bmax - bmin) * 255
    
    prev = self.img
    self.img = Image.fromarray(balanced_img)
    if self.interactive:
      self.show(prev)

  def rotate(self, angle):
    prev = self.img
    self.img = self.img.rotate(angle, Image.Resampling.NEAREST, expand = 1)
    if self.interactive:
      self.show(prev)

  def set_quality(self, value):
    self.quality = value

  def begin_stitching(self):
    self.stitch_list = [np.asarray(self.img)]

  def maximize_stitching(self):
    tmp = self.stitch_list[0]
    for i, img in enumerate(self.stitch_list):
      print(i)
      if i == 11:
        break

      if i == 0:
        continue

      pano1 = pano2 = None
      try:
        st = cv2.Stitcher.create(0)
        _, tmp_pano = st.stitch([tmp, img])
        if tmp_pano is not None:
          tmp = tmp_pano

        st = cv2.Stitcher.create(1)
        _, tmp_pano = st.stitch([tmp, img])
        if tmp_pano is not None:
          tmp = tmp_pano
      except:
        print("Error at maximize_stitching(): %s" % str(sys.exc_info()[1]))

    return tmp

  def end_stitching(self):
    if self.stitch_type != 2:
      stitcher = cv2.Stitcher.create(self.stitch_type)
    else:
      stitcher = cv2.Stitcher.create()

    status, pano = stitcher.stitch(self.stitch_list)
    self.stitch_list.clear()

    if pano is None:
      log("No images stitched, cv2 status %s" % status)
      return

    prev = self.img
    self.img = Image.fromarray(pano)
    if self.interactive:
      self.show(prev)

  def stitch(self, filename):
    img = Image.open(filename)
    if img is None:
      log("Cannot add image to stitch")
    else:
      self.stitch_list.append(np.asarray(img))
  
  def set_stitch_type(self, val):
    self.stitch_type = val

  def autocrop(self):
    box = self.img.getbbox()
    cropped = self.img.crop(box)
    prev = self.img
    self.img = cropped
    if self.interactive:
      self.show(prev)
  
  def draw_contours(self):
    im = np.asarray(self.img)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 255, 255, 255)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0,255,0), 3)

    prev = self.img
    self.img = Image.fromarray(im)
    if self.interactive:
      self.show(prev)

  def maxcrop(self):
    import imutils

    stitched = np.asarray(self.img)
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))

    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # create two copies of the mask: one to serve as our actual
    # minimum rectangular region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region
    minRect = mask.copy()
    sub = mask.copy()

    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    # use the bounding box coordinates to extract the our final
    # stitched image
    stitched = stitched[y:y + h, x:x + w]

    prev = self.img
    self.img = Image.fromarray(stitched)
    prev.resize([self.img.width, self.img.height], Image.Resampling.LANCZOS)
    if self.interactive:
      self.show(prev)
  
  def add_sub_image(self, image_file, add):
    tmp = open_file(image_file)
    tmp = np.asarray(tmp)
    current = np.asarray(self.img)
    if add:
      ret = cv2.addWeighted(current, 1, tmp, 1, 0)
    else:
      ret = cv2.subtract(current, tmp)

    prev = self.img
    self.img = Image.fromarray(ret)
    if self.interactive:
      self.show(prev)
  
  def denoise(self, val):
    image = np.asarray(self.img)
    if image.ndim > 2:
      log("Denoising color image")
      dst = cv2.fastNlMeansDenoisingColored(image, None, val, val, 7, 21)
    else:
      log("Denoising black & white image")
      dst = cv2.fastNlMeansDenoising(image, None, val, 7, 21)

    prev = self.img
    self.img = Image.fromarray(dst)
    if self.interactive:
      self.show(prev)
  
  def gray(self):
    gray = np.asarray(self.img.convert("L"))
    prev = self.img
    self.img = Image.fromarray(gray)
    if self.interactive:
      self.show(prev)
  
  def find_contours(self, only_count):
    img = np.asarray(self.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img, 100, 255, cv2.CV_8UC1)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    stars = []
    log("Found %d contour(s)" % len(contours))
    if only_count:
      return

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area < 2:
        continue
      x, y, w, h = cv2.boundingRect(cnt)
      # 1. X-coordinate
      x_coord = x + w/2.0
      # 2. Y-coordinate
      y_coord = y + h/2.0
      # 3. brightness
      star_mask = np.zeros(img.shape,np.uint8)
      cv2.drawContours(star_mask, [cnt], 0, 255, -1)
      mean_val = cv2.mean(img, mask=star_mask)[0]
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask=star_mask)
      # 4. radius
      radius = np.sqrt(area/(2*np.pi))
      curr_star = {'x': x_coord,
                    'y': y_coord,
                    'mean_brightness': mean_val,
                    'max_brightness': max_val,
                    'radius': radius}
      stars.append(curr_star)
    prev = self.img
    self.img = Image.fromarray(img)
    if self.interactive:
      self.show(prev)
  
#-------------------------------------------------------------------------------
def main(args):
  out_file = None
  image_file = args[0]
  tool = CSuperIrudiTool()
  tool.img = open_file(image_file)
  tool.original = tool.img
  l = args[1:]

  stitching = False
  ignore_next = False
  for i, arg in enumerate(l):
    if ignore_next:
      ignore_next = False
      continue

    if arg in ["-mh", "--match-histograms"]:
      if i+1 >= len(l):
        print("No image given to %s" % repr(arg))
        sys.exit(1)

      ignore_next = True
      ref_file = l[i+1]
      reference = open_file(ref_file)
      tool.image_match_histograms(tool.img, reference, out_file)
    elif arg in ["-ah", "--apply-histogram"]:
      if i+1 >= len(l):
        print("No histogram file given to %s" % repr(arg))
        sys.exit(1)

      ignore_next = True
      histogram_file = l[i+1]
      tool.image_apply_histogram(tool.img, histogram_file, out_file)
    elif arg in ["-o", "--output"]:
      if i+1 >= len(l):
        print("No image given to %s" % repr(arg))
        sys.exit(1)

      ignore_next = True
      out_file = l[i+1]
      tool.save_image(out_file)
    elif arg in ["-s", "--show"] or arg.startswith("-s=") or arg.startswith("--show="):
      pos = arg.find("=")
      if pos == -1:
        cmap = "viridis"
      else:
        cmap = arg[pos+1:]
      tool.show(cmap=cmap)
    elif arg in ["-i", "--interactive"]:
      tool.interactive = not tool.interactive
    elif arg in ["-b", "--blur"]:
      tool.apply_filter(ImageFilter.BLUR)
    elif arg in ["-c", "--contour"]:
      tool.apply_filter(ImageFilter.CONTOUR)
    elif arg in ["-d", "--detail"]:
      tool.apply_filter(ImageFilter.DETAIL)
    elif arg in ["-ee", "--edge-enhace"]:
      tool.apply_filter(ImageFilter.EDGE_ENHANCE)
    elif arg in ["-eem", "--edge-enhace-more"]:
      tool.apply_filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif arg in ["-eb", "--emboss"]:
      tool.apply_filter(ImageFilter.EMBOSS)
    elif arg in ["-fe", "--find-edges"]:
      tool.apply_filter(ImageFilter.FIND_EDGES)
    elif arg in ["-sh", "--sharpen"]:
      tool.apply_filter(ImageFilter.SHARPEN)
    elif arg in ["-sm", "--smooth"]:
      tool.apply_filter(ImageFilter.SMOOTH)
    elif arg in ["-smm", "--smooth-more"]:
      tool.apply_filter(ImageFilter.SMOOTH_MORE)
    elif arg in ["-minf", "--min-filter"]:
      tool.apply_filter(ImageFilter.MinFilter())
    elif arg in ["-maxf", "--max-filter"]:
      tool.apply_filter(ImageFilter.MaxFilter())
    elif arg in ["-medf", "--median-filter"]:
      tool.apply_filter(ImageFilter.MedianFilter())
    elif arg.startswith("-bb=") or arg.startswith("--box-blur="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.apply_filter(ImageFilter.BoxBlur(val))
    elif arg.startswith("-gb=") or arg.startswith("--gaussian-blur="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.apply_filter(ImageFilter.GaussianBlur(val))
    elif arg.startswith("-sh=") or arg.startswith("--sharpness="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.sharpness(val)
    elif arg.startswith("-c=") or arg.startswith("--color="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.color(val)
    elif arg.startswith("-con=") or arg.startswith("--contrast="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.contrast(val)
    elif arg.startswith("-b=") or arg.startswith("--brightness="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.brightness(val)
    elif arg in ["-ac", "--auto-contrast"]:
      tool.autocontrast()
    elif arg in ["-eq", "--equalize"]:
      tool.equalize()
    elif arg in ["-f", "--flip"]:
      tool.flip()
    elif arg in ["-iv", "--invert"]:
      tool.invert()
    elif arg in ["-m", "--mirror"]:
      tool.mirror()
    elif arg.startswith("-p=") or arg.startswith("--posterize="):
      pos = arg.find("=")
      val = int(arg[pos+1:])
      tool.posterize(val)
    elif arg.startswith("-s=") or arg.startswith("--solarize="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.solarize(val)
    elif arg in ["-pen", "--pencil"]:
      tool.pencil()
    elif arg == "-hdr":
      tool.hdr()
    elif arg == "--summer":
      tool.apply_summer_winter(False)
    elif arg == "--winter":
      tool.apply_summer_winter(True)
    elif arg.startswith("-do") or arg.startswith("--deoldify"):
      pos = arg.find("=")
      val = arg[pos+1:]
      tool.deoldify(val)
    elif arg in ["-ab", "--auto-white-balance"]:
      tool.auto_balance()
    elif arg.startswith("-r=") or arg.startswith("--rotate="):
      pos = arg.find("=")
      val = float(arg[pos+1:])
      tool.rotate(val)
    elif arg.startswith("-q=") or arg.startswith("--quality="):
      pos = arg.find("=")
      val = int(arg[pos+1:])
      tool.set_quality(val)
    elif arg in ["-bs", "--begin-stitching"]:
      stitching = True
      tool.begin_stitching()
    elif arg in ["-es", "--end-stitching"]:
      stitching = False
      tool.end_stitching()
    elif stitching and os.path.exists(arg):
      tool.stitch(arg)
    elif arg.startswith("-st=") or arg.startswith("--stitch-type="):
      pos = arg.find("=")
      val = int(arg[pos+1:])
      tool.set_stitch_type(val)
    elif arg == "--auto-crop":
      tool.autocrop()
    elif arg in ["-mc", "--max-crop"]:
      tool.maxcrop()
    elif arg in ["-dc", "--draw-contours"]:
      tool.draw_contours()
    elif arg in ["-a", "--add", "-su", "--subtract"]:
      if i+1 >= len(l):
        print("No image file given to %s" % repr(arg))
        sys.exit(1)

      ignore_next = True
      image_file = l[i+1]

      add = arg in ["-a", "--add"]
      tool.add_sub_image(image_file, add=add)
    elif arg.startswith("-dn=") or arg.startswith("--denoise"):
      pos = arg.find("=")
      if pos == -1:
        val = 10
      else:
        val = float(arg[pos+1:])
      tool.denoise(val)
    elif arg in ["-g", "--gray"]:
      tool.gray()
    elif arg in ["-fc", "--find-contours"]:
      tool.find_contours(False)
    elif arg in ["-cc", "--count-contours"]:
      tool.find_contours(True)
    elif arg.startswith("--ncols="):
      pos = arg.find("=")
      val = int(arg[pos+1:])
      tool.ncols = val
    elif arg.startswith("--nrows="):
      pos = arg.find("=")
      val = int(arg[pos+1:])
      tool.nrows = val
    elif arg in ["-p", "-print", "--print"]:
      if i+1 >= len(l):
        print("No message given to %s" % repr(arg))
        sys.exit(1)

      ignore_next = True
      message = l[i+1]
      log(message)
    elif arg == "--debug":
      global DEBUG
      DEBUG = True
    elif arg.startswith("-ua=") or arg.startswith("--user-agent="):
      global USER_AGENT
      pos = arg.find("=")
      USER_AGENT = arg[pos+1:]
    else:
      print("Unknown command line option %s" % repr(arg))
      sys.exit(2)

#-------------------------------------------------------------------------------
def usage():
  # TODO: Document all the command line options (ETOOMANY & ETOOLAZY)
  print("Usage: <source image> [options]")
  print()
  print("Options:")
  print("-o/--output                      Output image filename.")
  print("-s/--show                        Show the current in-memory image.")
  print("-sg/--show-gray                  Show the current in-memory image as gray.")
  print("-i/--interactive                 Toggle interactive mode.")
  print("-ah/--apply-histogram <file>     Apply histogram file.")
  print("-mh/--match-histograms <image>   Use the color histogram from the given image.")
  print("-gb/--gausian-blur               Apply a Gaussian blur filter.")
  print("-sh/--sharpen                    Apply a sharpen filter.")
  print("-sh=/--sharpness=<value>         Set a sharpness factor.")
  print("-c=/--color=<value>              Adjust image color balance, value 0 to 1.")
  print("-con=/--contrast=<value>         Adjust image contrast, value 0 to 1.")
  print("-ac/--auto-contrast              Normalize image contrast.")
  print("-do=/--deoldify=<url>            Colorize using DeOldify server at given URL.")
  print("-ab/--auto-white-balance         Automatic white balance.")
  print("-r=/--rotate=<angle>             Rotate the image the given angle.")
  print("-q=/--quality=<value>            Image quality, from 0 to 100.")
  print("-bs/--begin-stitching            Initialize the picture stitching mode.")
  print("-es/--end-stitching              End the picture stitching mode.")
  print("-st=/--stich-type=<type>         Set the stitching type, either 0 (panorama) or 1 (scan).")
  print("--auto-crop                      Automatically crop the image.")
  print("-fc/--find-contours              Find and draw a broder around all contours in the image.")
  print("-cc/--count-contours             Count all contours in the image.")
  print("-mc/--max-crop                   Find the maximum crop for stitched images (buggy).")
  print("-a/--add <image>                 Add the given image file.")
  print("-su/--subtract <image>           Subtract the given image file.")
  print("--nrows=<value>                  Set the number of rows for displaying images.")
  print("--ncols=<value>                  Set the number of columns for displaying images.")
  print("-p/--print <message>             Show the given message.")
  print("-ua=/--user-agent=<agent>        Set the given User-Agent.")
  print()

if __name__ == "__main__":
  if len(sys.argv) <= 2:
    usage()
  else:
    try:
      main(sys.argv[1:])
    except SystemExit as e:
      sys.exit(e)
    except KeyboardInterrupt:
      print("Aborted")
      if DEBUG:
        raise
    except:
      err = str(sys.exc_info()[1])
      print("ERROR:", err)
      if DEBUG:
        raise

