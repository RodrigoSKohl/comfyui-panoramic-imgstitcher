<div align="center">

# Panoramic Image Stitcher for ComfyUI

</div>

## Simple Node to make panoramic images using [OpenCV](https://github.com/opencv) stitch function


This node is available on [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

#



## PARAMETERS
| Name           | Description                                                                                                        | Type                     | Min   | Max   | Default       | Step |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------ | ----- | ----- | ------------- | ---- |
| `crop`         | Enable or disable cropping of the smallest acceptable area with perspective warp pixels                            | `"enable" \| "disable"`  | —     | —     | `"enable"`    | —    |
| `mode`         | Switch between stitching modes: `"panoramic"` (default) or `"scans"` (optimized for documents/text)                | `"panoramic" \| "scans"` | —     | —     | `"panoramic"` | —    |
| `conf_thresh`    | Decreases the confidence threshold for creating stitch points. Lowering this value makes the algorithm less strict, potentially causing stitching errors or misalignments. See [FAQ](#faq) | Float (0.0–1.0)          | 0.0   | 1.0   | 1.0           | 0.01 |
| `work_megapix` | Resolution (in megapixels) used for registration step. Higher values = better quality, slower process              | Float (0.001–100.0)      | 0.001 | 100.0 | 0.6           | 0.01 |
| `seam_megapix` | Resolution (in megapixels) used for seam estimation. Lower values speed up the process with some quality loss      | Float (0.001–100.0)      | 0.001 | 100.0 | 0.1           | 0.01 |

[Default values encountered here](https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp)

## FAQ

- ``Error 1``
  
This error occurs when the stitcher cannot find enough stitch points between the images. The stitching algorithm requires a sufficient number of matching points to correctly join the images.

How to avoid or fix this error:

Increase the number of images and reduce the spacing between them to improve overlap and matching points;
Increase the work_megapix and seam_megapix parameters, especially work_megapix, to process images at higher resolution. This is particularly useful for very large (high-resolution) images where default downscaling might remove important details needed for matching;


- ``Error 3``

Controls the confidence threshold for creating stitch points. Lowering this value makes the algorithm less strict but can cause stitching errors, especially if images are not well captured or not in the correct sequential order (batch).

Important: Always use the default value 1.0 unless your images are high quality, well aligned, and captured carefully;
Images are provided in the correct sequence, ensuring smooth visual continuity.
Lowering the threshold offers no real advantage except to avoid false positives images. 

## Next features

- Automatically calculate work_megapix and seam_megapix.

*Feel free to open PR or issues.*
