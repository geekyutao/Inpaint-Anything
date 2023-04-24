# Web UI for Inpaint Anything
We provide a web UI for people who want to run the demo Web locally.

## Usage
  We use [Gradio](https://gradio.app/) to build our Web UI. First, install Gradio through pip.
  ```
  pip install gradio
  ```
  Then use the following command to create a web link.
  ```
  python app.py \
        --lama_config [the path of lama config] \
        --lama_ckpt [the path of lama ckpt] \
        --sam_ckpt [the path of sam ckpt]
  ```

<!-- ## Components
  - `Input Image`: where the uploaded image is displayed for the user to view.
  - `Pointed Image`: where the pointed image is displayed for the user to view.
  - `Control Panel`: where users can control Point Coordinate mannually, and execute `Predict Mask Using SAM`, `Inpaint Image Using LaMA`, and `Reset`
  - `Segmentation Mask`: where the segmentation mask predicted by SAM is displayed for the user to view.
  - `Image with Mask`: where the image with the segmentation mask predicted by SAM is displayed for the user to view.
  - `Image Removed with Mask`: where the image inpainted by LaMA is displayed for the user to view. -->

## Instruction
There are 4 steps for *Remove Anything*:
- Step 1: Upload your image;
- Step 2: Click on the object that you want to remove or input the coordinates to specify the point location, and wait until the pointed image shows;
- Step 3: Hit the "Predict Mask Using SAM" button, and wait until the segmentation results show;
- Step 4: Hit the "Inpaint Image Using Lama" button, and wait until the inpainting results show.

if need, you can hit the "Reset" button to reset the web to the initial state.

### Example

<!-- ### Upload Your Image
To upload an image, click on the `Input Image` component or drop your image to the component. Once uploaded, the `Input Image` component will display the image in the Web UI. 

## Set A Point Prompt for SAM
- Click a point in the `Input Image` to set a point prompt for SAM, and the pointed image will be shown in the `Pointed Image`.
- The example of the operation is shown below. -->
Step 1 & 2:
<p align="center"><img src="./assets/point_prompt.png" width = "1500" height = "400" alt="point_prompt"/></p>

<!-- - Or you can set the point prompt in `Control Panel` by fill the coordinate. -->

<!-- ## Predict Mask Using SAM
 - Click the `Predict Mask Using SAM` button in the `Control Panel` component to get the masks prediceted by SAM.
 - The prediceted masks will be shown in the `Segmentation Mask` component automatically.
 - The images with the segmentation masks will be shown in `Image with Mask` component automatically.
 - The example of the operation is shown below. -->
Step 3:
<p align="center"><img src="./assets/segmentation_mask.png" width = "1500" height = "1200" alt="segmentation_mask"/></p>

<!-- ## 5. Inpaint Image Using LaMA
  - Click the `Inpaint Image Using LaMA` button in the `Control Panel` component to get the image inpainted by LaMA.
  - The inpainted images will be shown in the `Image Removed with Mask` component automatically.
  - The example of the operation is shown below. -->
Step 4:
<p align="center"><img src="./assets/image_removed.png" width = "1500" height = "1600" alt="image_removed"/></p>

<!-- ## 6. Reset
  - By Clicking the `Reset` button in  the `Control Panel`, you can reinitialize Inpaint Anything. -->
