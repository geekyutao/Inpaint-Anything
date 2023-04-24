# WebUI tutorial

## 1. Preparation 
  - We use [**Gradio**](https://gradio.app/quickstart/) to construct our WebUI, please install **Gradio** by
  ```
  pip install gradio
  ```
  - Then, using the following command to create a Web link.
  ```
  python app.py \
        --lama_config [the path of lama config] \
        --lama_ckpt [the path of lama ckpt] \
        --sam_ckpt [the path of sam ckpt]
  ```

## 2. Upload your Image
- To upload a image, click on the `Input Image` component or drop your image to the component. Once uploaded, the `Input Image` component will display the image in the WebUI. 

## 3. Set a Point Prompt for SAM
- Click a point in the `Input Image` to set a point prompt for SAM, and the pointed image will be shown in the `Pointed Image`.
- The example of the operation is shown below.
<p align="center"><img src="./assets/point_prompt.png" width = "1500" height = "400" alt="point_prompt"/></p>

- Or you can set the point prompt in `Control Panel` by fill the coordinate.

## 4. Predict Mask Using SAM
 - Click the `Predict Mask Using SAM` button in the `Control Panel` component to get the masks prediceted by SAM.
 - The prediceted masks will be shown in the `Segmentation Mask` component automatically.
 - The images with the segmentation masks will be shown in `Image with Mask` component automatically.
 - The example of the operation is shown below.
<p align="center"><img src="./assets/segmentation_mask.png" width = "1500" height = "1200" alt="segmentation_mask"/></p>

## 5. Inpaint Image Using LaMA
  - Click the `Inpaint Image Using LaMA` button in the `Control Panel` component to get the image inpainted by LaMA.
  - The inpainted images will be shown in the `Image Removed with Mask` component automatically.
  - The example of the operation is shown below.
<p align="center"><img src="./assets/image_removed.png" width = "1500" height = "1600" alt="image_removed"/></p>

## 6. Reset
  - By Clicking the `Reset` button in  the `Control Panel`, you can reinitialize Inpaint Anything.