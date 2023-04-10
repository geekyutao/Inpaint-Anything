from segment_anything import SamPredictor, sam_model_registry
# import cv2
import PIL.Image as Image
import numpy as np

'''
    sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    predictor = SamPredictor(sam)
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
'''

def generate_msk(
    p_image,
    input_point,
    input_label,
    sam_checkpoint="./../../pretrain/sam_vit_b_01ec64.pth",
    model_type="vit_b",
    device="cuda",
):
    """An simple example of using segment_anything to generate mask.

    Args:
        p_image (str): the path of the image.
        input_point (np.ndarray or None): A Nx2 array of point prompts to the
          model. Each point is in (X,Y) in pixels.
        point_labels (np.ndarray or None): A length N array of labels for the
          point prompts. 1 indicates a foreground point and 0 indicates a
          background point.
        sam_checkpoint (str, optional): the path of checkpoint. Defaults to "./../../pretrain/sam_vit_b_01ec64.pth".
        model_type (str, optional): the model type. Defaults to "vit_b".
        device (str, optional): device. Defaults to "cuda".

    Returns:
        masks: (numpy.array, bool) the returned mask, shape is (3, H, W). 
        scores: (list, float) the returned score. len(scores) = 3.
        logits: todo.
    """    

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam) 

    image = np.array(Image.open(p_image).convert("RGB"))

    # generate image embedding
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    return masks, scores, logits


if __name__ == '__main__':
    # specify your model
    sam_checkpoint = "./../../pretrain/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # sam_checkpoint = "./../../pretrain/sam_vit_l_0b3195.pth"
    # model_type = "vit_l"
    
    # sam_checkpoint = "./../../pretrain/sam_vit_b_01ec64.pth"
    # model_type = "vit_b"

    # specify the path of your image and the point prompts
    p_image = "./segment_anything/notebooks/images/truck.jpg"
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # now, generate the mask
    masks, scores, logits = generate_msk(
        p_image,
        input_point,
        input_label,
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        device="cuda",
    )

    # save the mask
    for idx in range(len(masks)):
        mask = (masks[idx]*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save("example/example_mask_{}.png".format(idx))