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

import sys
import time

import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, exposure
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.interpolate import UnivariateSpline

Image.MAX_IMAGE_PIXELS = None

from histogram_matching import (match_histograms, match_histograms_with,
                                load_histogram_from_file)

#-------------------------------------------------------------------------------
DEBUG = False

#-------------------------------------------------------------------------------
def log(msg):
  print("[%s] %s" % (time.asctime(), msg))

#-------------------------------------------------------------------------------
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

#-------------------------------------------------------------------------------
class CSuperIrudiTool:
  def __init__(self):
    self.img = None
    self.original = None
    self.interactive = False

  def save_image(self, out_file):
    self.img.save(out_file)

  def show(self, prev=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2):
        aa.set_axis_off()

    if prev is not None:
      ax1.imshow(prev)
      ax1.set_title('Previous')
    else:
      ax1.imshow(self.original)
      ax1.set_title('Original')

    ax2.imshow(self.img)
    ax2.set_title('Current')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
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
      fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                          sharex=True, sharey=True)
      for aa in (ax1, ax2, ax3):
          aa.set_axis_off()

      ax1.imshow(image)
      ax1.set_title('Source')
      ax3.imshow(matched)
      ax3.set_title('Matched')

      plt.tight_layout()
      mng = plt.get_current_fig_manager()
      mng.resize(*mng.window.maxsize())
      plt.show()

  def image_match_histograms(self, image, reference, out_file):
    log("Matching histograms...")
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

      fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
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

#-------------------------------------------------------------------------------
def main(args):
  out_file = None
  image_file = args[0]
  tool = CSuperIrudiTool()
  tool.img = Image.open(image_file)
  tool.original = tool.img
  l = args[1:]

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
      reference = Image.open(ref_file)
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
    elif arg in ["-s", "--show"]:
      tool.show()
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
    elif arg == "--debug":
      global DEBUG
      DEBUG = True
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
  print("-i/--interactive                 Toggle interactive mode.")
  print("-ah/--apply-histogram <file>     Apply histogram file.")
  print("-mh/--match-histograms <img>     Use the color histogram from the given image.")
  print("-gb/--gausian-blur               Apply a Gaussian blur filter.")
  print("-sh/--sharpen                    Apply a sharpen filter.")
  print("-sh=/--sharpness=<value>         Set a sharpness factor.")
  print("-c=/--color=<value>              Adjust image color balance, value 0 to 1.")
  print("-con=/--contrast=<value>         Adjust image contrast, value 0 to 1.")
  print("-ac/--auto-contrast              Normalize image contrast.")
  print()

if __name__ == "__main__":
  if len(sys.argv) <= 2:
    usage()
  else:
    try:
      main(sys.argv[1:])
    except SystemExit as e:
      sys.exit(e)
    except:
      err = str(sys.exc_info()[1])
      print("ERROR:", err)
      if DEBUG:
        raise

