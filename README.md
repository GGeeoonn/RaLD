# RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation

## Training
The training code is still under refactoring, and will update it later.

**Training Details** <br>
The model was trained for 300 epochs using 7 NVIDIA A6000 GPUs, and the total training time was approximately 8 days.

# RaDL GUI
We have combined RaDL and [GLIGEN-GUI](https://github.com/mut-ex/gligen-gui) to generate images that are more convenient for users.

**Generate different styles of images**: RaDL can also serve as a flexible plug-and-play module, enabling the generation of diverse image styles while preserving instance-specific attributes and relationships. By integrating RaDL with different base diffusion models, users can easily adapt the framework to produce images with varying artistic, photorealistic, or domain-specific styles without compromising spatial accuracy or semantic alignment. <br>
(Alternatively, you can visit [civitai](https://civitai.com/) to download other models of your preference)

**For instance:**
- **[RV60B1](https://civitai.com/models/4201/realistic-vision-v60-b1)**: Ideal for those seeking lifelike detail, RV60B1 specializes in generating images with stunning realism. <br>
- **[Cetus-Mix](https://civitai.com/models/6755/cetus-mix)**: These robust base models excel in crafting animated content.

## ⬇ Installation
### 1. Conda environment setup
```
conda create -n radl_env python=3.9.21
conda activate radl_env
```
<br>
Move to the RaDL folder
<br>
```
pip install -r requirements.txt
pip install -e .
```

### 2. Checkpoint
Download the [radl_pretrained.ckpt](https://drive.google.com/file/d/1wD_xokpeKK8MxXhmnQ2DkN-ihkBzlGZz/view?usp=sharing) and put it under the 'pretrained_weights' folder.
```
├── pretrained_weights
│   ├── radl_pretrained.ckpt
├── radl
│   ├── ...
├── radl_gui
│   ├── ...
```

### 3. CLIP Text Encoder
Download the [CLIPTextModel](https://drive.google.com/file/d/1SkXlvXQZxYFNzaAaEdnANhKSuB0p9C1a/view?usp=sharing) and put it under the 'text_encoder' folder. <br>
(place it in `radl_gui_weights/clip/text_encoder/pytorch_model.bin`.)
```
├── clip
│   ├── text_encoder
├── │   ├── pytorch_model.bin
│   ├── tokenizer
├── │   ├── ...
```

### 4. style models 
Download the [RV60B1](https://drive.google.com/file/d/16vJ0dF-NjTx4duL7RooheNCbX7HG_BgU/view?usp=sharing) model and place it in `radl_gui_weights/sd/realisticVisionV60B1_v51HyperVAE.safetensors`. Alternatively, you can visit [civitai](https://civitai.com/) to download other models of your preference and place them in `radl_gui_weights/sd/`. <br>
**※ Even if you add multiple style models, you can choose a style within the GUI later.**

```
├── pretrained_weights
│   ├── radl_pretrained.ckpt
├── radl_gui_weights
│   ├── clip
│   │   ├── text_encoder
│   │   │   ├── pytorch_model.bin
│   │   ├── '''
│   ├── sd
│   │   ├── realisticVisionV60B1_v51HyperVAE.safetensors
```

### 4. Move GUI folder
```
cd radl_gui
```

### 5. Run GUI
Launch the application by running `python app.py --port=22222`. You can now access the RaDL GUI through http://localhost:22222/. <br> 
You can freely change the port at your convenience.

### ● GUI Example Image
![image](https://github.com/user-attachments/assets/b34ecafa-3ca2-4b81-afff-5d1148368e8a)
![image](https://github.com/user-attachments/assets/72e08883-e1a1-4b1d-8693-537eaf42ccd3)
![image](https://github.com/user-attachments/assets/cafe66d7-d4a5-4bf6-be8a-ad0dd6698862)
![image](https://github.com/user-attachments/assets/227dac2c-7027-4f23-8907-6debf28a4ee6)

## Contact us
If you have any questions, feel free to contact me via email geonpark@korea.ac.kr

## References
- Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., ... & Lee, Y. J. (2023). Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 22511-22521).
- Zhou, D., Li, Y., Ma, F., Zhang, X., & Yang, Y. (2024). Migc: Multi-instance generation controller for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 6818-6828).
