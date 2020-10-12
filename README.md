# LPRNet Pytorch
Pytorch Implementation For LPRNet, A High Performance And Lightweight License Plate Recognition Framework.(Chinese Number Plates Recognition)

Indian Number Plate Recognition Modification.

## Dependencies

- pytorch >= 1.0.0
- opencv-python 3.x
- python 3.x
- imutils
- Pillow
- numpy

### Tasks

- [x] Dataset preprocessor for csv label format. 
- [x] Added robustness to preprocessor. 
- [ ] Tune hyperparameters.

#### Dataset preprocessing

1. Image name should be its label and separated into test and train. Otherwise:
2. Preprocessor.py will split data into train and test (9:1) and rename labels.
3. Run preprocessor.py and pass input folder, required format:
4. Input folder to contain 2 items, a folder containing all images and a csv/excel file of labels.
5. Csv should look like:

| img name | Label |
| :----: | :----: |
| xyz.png  | KA00XX0000 |


##### Training and Testing

1. Uncomment get_size function calls in train() to train with median size of dataset, default size is 94,24. Edit: Model only works for 94,24 size right now.
2. Based on your dataset path modify the script and its hyperparameters.
3. Adjust other hyperparameters if needed.
4. Run 'python train_LPRNet.py' or 'python test_LPRNet.py'.
5. If want to show testing result, add '--show true' or '--show 1' to run command.


###### References

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)
3. [https://github.com/sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
