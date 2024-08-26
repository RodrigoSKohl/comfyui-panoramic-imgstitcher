<div align="center">

# Panoramic Image Stitcher for ComfyUI

</div>

## Simple Node to make panoramic images using [OpenCV](https://github.com/opencv) stitch function


This node is available on [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

#

![1](https://github.com/user-attachments/assets/012d3e35-f12c-4ea8-b1df-3561e7fd9eca)
![2](https://github.com/user-attachments/assets/9216004f-bc8e-46a3-8fef-414eed4b0da0)

## PARAMETERS
| Name   | Description                                           |
|--------|-------------------------------------------------------|
| ``device`` | Improved performance only if using it in flow, since OpenCV functions run only on the CPU    |
| ``crop``   | Crops the smallest acceptable area with perspective warp pixels |
| ``mode`` | Switches between panoramic mode (default) or scans mode, optimized for documents and images with details such as letters |
| ``threshold`` | Decreases the precision to create stitch points, but can cause errors, see [FAQ](#faq)|

## FAQ

- ``Error 1``
  
Occurs when the stitcher cannot find the stitch points between the images (the stitcher function algorithm needs it to return an average of equality points to join the image. Increasing the number of images and decreasing the spacing between them avoids this problem

- ``Error 3``

It happens when the threshold is too low. Always use the default value 1.0 and go down to refine. In the tests carried out there was little gain in the reduction of warp perspective by lowering this parameter.

## Next features

- Improvements in code redundancy between some functions.
- Impement the mask as alpha directly in the image.

*Feel free to open PR or issues.*
