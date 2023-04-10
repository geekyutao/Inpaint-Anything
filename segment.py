from segment_anything import SamPredictor, sam_model_registry
import cv2
from matplotlib import pyplot as plt
import PIL.Image as Image
import numpy as np

'''
    sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    predictor = SamPredictor(sam)
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
'''

def generate_mask(
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
        masks: (numpy.array [bool]) the returned masks, shape is (3, H, W). 
        scores: (list [float]) the returned score. len(scores) = 3.
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


if __name__ == '__main__':
    choose_model = 'h' # 'h' for vit_h, 'l' for vit_l, 'b' for vit_b
    samples = ['baseball', 'boat', 'bridge', 'cat',
                        'dog', 'groceries', 'hippopotamus', 'person', 'person_kite', 'person_umbrella']
    # samples = ['person_umbrella']
    input_points = {
        'baseball': [240, 250],
        'boat': [300, 580],
        'bridge': [100, 300],
        'cat': [600, 1100],
        'dog': [200, 450],
        'groceries': [400, 300],
        'hippopotamus': [400, 300],
        'person': [50, 700],
        'person_kite': [275, 225],
        'person_umbrella': [500, 200],
    }
    for sample in samples:
        dilate_factor = 15

        # specify your model
        assert choose_model in ['h', 'l', 'b']
        if choose_model == 'h':
            sam_checkpoint = "./../../pretrain/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
        elif choose_model == 'l':
            sam_checkpoint = "./../../pretrain/sam_vit_l_0b3195.pth"
            model_type = "vit_l"
        elif choose_model == 'b':
            sam_checkpoint = "./../../pretrain/sam_vit_b_01ec64.pth"
            model_type = "vit_b"
        
        # specify the path of your image and the point prompts
        p_image = "./example/{}.jpg".format(sample)
        input_label = np.array([1])
        input_point =  np.array([input_points[sample]])

        # now, generate the mask
        masks, scores, logits = generate_mask(
            p_image,
            input_point,
            input_label,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            device="cuda",
        )

        masks = (masks*255).astype(np.uint8)

        image = np.array(Image.open(p_image).convert("RGB"))
        for idx in range(len(masks)):
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            show_points(input_point, input_label, plt.gca())
            mask = masks[idx]
            mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)
            mask_image = show_mask(((mask)/255).astype(np.bool_), plt.gca(), random_color=False)
            plt.imshow(mask_image)
            plt.savefig("example/{}_masked_{}.png".format(sample, idx), dpi=300, bbox_inches='tight', pad_inches=0)

            # save the mask
            mask = Image.fromarray(mask)
            mask.save("example/{}_mask_{}.png".format(sample, idx))
            plt.close()