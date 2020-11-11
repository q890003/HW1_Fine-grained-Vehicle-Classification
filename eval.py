import csv
import os
import torch
import config
import utils
from PIL import Image
from transforms import get_image_transform
from datasets import get_dataloader 
from models import get_resnet101_32x8d_pretrained_model


# check gpu
device = utils.get_gpu()

# Load ckpt and get state_dict
model_ft = get_resnet101_32x8d_pretrained_model()
loaded_ckpt = config.ckpt_dir + config.model_name
state_dict = torch.load(loaded_ckpt)

# Load weights
model_ft.load_state_dict(state_dict)
model_ft = model_ft.to(device)

model_name = config.model_name

#make result file
csv_name = model_name + '.csv'
with open(csv_name, 'w', newline='') as csvFile:
    field = ['id', 'label']
    writer = csv.DictWriter(csvFile, field)
    writer.writeheader()
    
    # load classification_map
    _, _, dataset_map = get_dataloader()

    # get name list of testing images 
    allFileList = os.listdir(config.test_img_folder)
    
    # Read each testing image
    for file in allFileList:
        if os.path.isfile(config.test_img_folder + file):
            path = config.test_img_folder + file
            img = Image.open(path).convert('RGB')
            dataset_transform = get_image_transform()  
            img = dataset_transform['val'](img)
            img = img.unsqueeze(0)
            
            # prediction
            with torch.no_grad():
                model_ft.eval()
                img = img.to(device)
                outputs = model_ft(img)
                _, preds = torch.max(outputs, 1)
                predict = preds.cpu().numpy()
            
            # record predictions and convert labels from numbers back to string
            writer.writerow({'id': file.split('.jpg')[0],'label':dataset_map[predict[0]]})
