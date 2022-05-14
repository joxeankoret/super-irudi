#!/usr/bin/python3

"""
Based on the original histogram_matching.py from skimage, and with the same
license. 

Copyright (c) 2022, Joxean Koret

Original copyright notice bellow:

Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

#-------------------------------------------------------------------------------
DEBUG = False

#-------------------------------------------------------------------------------
def _match_cumulative_cdf_with(source, tmpl_values, tmpl_counts, tmpl_size):
  """
  Return modified source array so that the cumulative density function of
  its values matches the cumulative density function of the template.
  """
  src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                               return_inverse=True,
                               return_counts=True)

  # calculate normalized quantiles for each array
  src_quantiles = np.cumsum(src_counts) / source.size
  tmpl_quantiles = np.cumsum(tmpl_counts) / tmpl_size

  interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
  return interp_a_values[src_unique_indices].reshape(source.shape)

#-------------------------------------------------------------------------------
def match_histograms_with(image, histogram, multichannel=False):
  if multichannel:
    if image.shape[-1] != len(histogram):
      raise ValueError('Number of channels in the input image and '
               'histogram file must match!')

    matched = np.empty(image.shape, dtype=image.dtype)
    for channel in range(image.shape[-1]):
      matched_channel = _match_cumulative_cdf_with(image[..., channel],
                          histogram[channel][0],
                          histogram[channel][1],
                          histogram[channel][2])
      matched[..., channel] = matched_channel
  else:
    matched = _match_cumulative_cdf_with(image, histogram)

  return matched

#-------------------------------------------------------------------------------
def _match_cumulative_cdf(source, template):
  """
  Return modified source array so that the cumulative density function of
  its values matches the cumulative density function of the template.
  """
  src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                               return_inverse=True,
                               return_counts=True)
  tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

  # calculate normalized quantiles for each array
  src_quantiles = np.cumsum(src_counts) / source.size
  tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

  interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
  return interp_a_values[src_unique_indices].reshape(source.shape)

#-------------------------------------------------------------------------------
def match_histograms(image, reference, *, multichannel=False):
  """Adjust an image so that its cumulative histogram matches that of another.

  The adjustment is applied separately for each channel.

  Parameters
  ----------
  image : ndarray
    Input image. Can be gray-scale or in color.
  reference : ndarray
    Image to match histogram of. Must have the same number of channels as
    image.
  multichannel : bool, optional
    Apply the matching separately for each channel.

  Returns
  -------
  matched : ndarray
    Transformed input image.

  Raises
  ------
  ValueError
    Thrown when the number of channels in the input image and the reference
    differ.

  References
  ----------
  .. [1] http://paulbourke.net/miscellaneous/equalisation/

  """
  if image.ndim != reference.ndim:
    raise ValueError('Image and reference must have the same number '
             'of channels.')

  if multichannel:
    if image.shape[-1] != reference.shape[-1]:
      raise ValueError('Number of channels in the input image and '
               'reference image must match!')

    matched = np.empty(image.shape, dtype=image.dtype)
    for channel in range(image.shape[-1]):
      matched_channel = _match_cumulative_cdf(image[..., channel],
                          reference[..., channel])
      matched[..., channel] = matched_channel
  else:
    matched = _match_cumulative_cdf(image, reference)

  return matched

#-------------------------------------------------------------------------------
def get_histogram(image):
  histogram = []
  for channel in range(image.shape[-1]):
    template = image[..., channel]
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    histogram.append([tmpl_values.tolist(), tmpl_counts.tolist(), template.size])
  return histogram

#-------------------------------------------------------------------------------
def get_histogram_from_file(filename):
  image = Image.open(filename)
  return get_histogram(np.asarray(image))

#-------------------------------------------------------------------------------
def save_histogram_to_file(histogram, filename):
  with open(filename, "w") as f:
    f.write(json.dumps(histogram))

#-------------------------------------------------------------------------------
def load_histogram_from_file(filename):
  buf = open(filename, "r").read()
  return json.loads(buf)

#-------------------------------------------------------------------------------
def test_show(img1, img2):
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                      sharex=True, sharey=True)
  for aa in (ax1, ax2, ax3):
      aa.set_axis_off()

  ax1.imshow(img1)
  ax1.set_title('Source')
  ax3.imshow(img2)
  ax3.set_title('Matched histogram')

  plt.tight_layout()
  mng = plt.get_current_fig_manager()
  mng.resize(*mng.window.maxsize())
  plt.show()

#-------------------------------------------------------------------------------
def main(image_filename, histogram_file):
  global DEBUG

  histogram = get_histogram_from_file(image_filename)
  save_histogram_to_file(histogram, histogram_file)
  if DEBUG:
    histo = load_histogram_from_file(histogram_file)

    img = Image.open("eso1205ec.jpg")
    matched = match_histograms_with(np.asarray(img), histo, multichannel=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(img)
    ax1.set_title('Source')
    ax3.imshow(matched)
    ax3.set_title('Matched histogram')

    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: %s <image file> <histogram file>" % sys.argv[0])
  else:
    main(sys.argv[1], sys.argv[2])

