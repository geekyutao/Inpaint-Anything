<p align="center">
  <img src="./example/IAM.png">
</p>

# **Inpaint-Anything**
## Segment Anything Meets Image Inpainting
<p align="center">
  <img src="./example/framework.png" width="100%">
</p>


## Inpaint Anything Features
- [x] Remove Anything
- [ ] Fill Anything (coming soon)
- [ ] Replace Anything (coming soon)
- [ ] Demo Website (coming soon)


## Remove Anything
**Click** on an object in the image (2K image supported!), and Inpainting Anything will **remove** it instantly!
- Click on an object, [Segment Anything Model](https://segment-anything.com/) (SAM) is used to segment the object and obtain an object mask.
- With the mask, inpainting models (e.g., [LaMa](https://advimman.github.io/lama-project/)) fill the "hole".

<table>
  <tr>
    <td><img src="./example/remove-anything/dog_pointed.png" width="100%"></td>
    <td><img src="./example/remove-anything/dog_masked.png" width="100%"></td>
    <td><img src="./example/remove-anything/dog_inpainted.png" width="100%"></td>
  </tr>
</table>


<table>
  <tr>
    <td><img src="./example/remove-anything/person_pointed.png" width="100%"></td>
    <td><img src="./example/remove-anything/person_masked.png" width="100%"></td>
    <td><img src="./example/remove-anything/person_inpainted.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./example/remove-anything/bridge_pointed.png" width="100%"></td>
    <td><img src="./example/remove-anything/bridge_masked.png" width="100%"></td>
    <td><img src="./example/remove-anything/bridge_inpainted.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./example/remove-anything/boat_pointed.png" width="100%"></td>
    <td><img src="./example/remove-anything/boat_masked.png" width="100%"></td>
    <td><img src="./example/remove-anything/boat_inpainted.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./example/remove-anything/baseball_pointed.png" width="100%"></td>
    <td><img src="./example/remove-anything/baseball_masked.png" width="100%"></td>
    <td><img src="./example/remove-anything/baseball_inpainted.png" width="100%"></td>
  </tr>
</table>


## Acknowledgments
 - [SAM](https://github.com/facebookresearch/segment-anything) from Meta AI
 - Inpainting models are from [LaMa](https://github.com/advimman/lama)

 ## Other Interesting Repo
- [Awesome Anything](https://github.com/VainF/Awesome-Anything)
- [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)







