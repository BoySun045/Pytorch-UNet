# FrontierNet training code 


### Adapt from https://github.com/milesial/Pytorch-UNet

### Install

Install dependencies
```bash
pip install -r requirements.txt
```


### Training

Train both distance field and info gain segmentation head (recommanded): 

```console
python3 /cluster/project/cvg/boysun/Pytorch-UNet-latest/train.py --amp --epochs 100 -b 16 -s 0.67 -c 7 -v 10 -l 5e-5 -rw 8.0 -ud --head_mode df_seg
```

Notes:

- Best performance comes from "df_seg" mode.
- Change the "dir_path" in train.py to your dataset folder. 
- Check the args setup in train.py for details about the arguments. 
- Change the number of classes (-c in args) to fit your segmentation setup, and provide your corresponding "multi_class_weights_path" in train.py, which gives threshold for each class. 
- Data Preprocessing always does crop and rescale - the input to the model is always rescaled squared image.
- RGBD as input gives much better results (i.e. with -ud), if you need to train with RGB only then remove the arugment, might have some unknown issues. 

### Dataset Generation/Transformation

- Check dataset/data_gen.py as an example. 
- It refine the binary frontier mask with depth discontinuity. Modify the "get_frontier_line_mask()" function and remove this refinment if you won't need it. 

