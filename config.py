##################################################
# Training Config
##################################################
workers = 4  # number of Dataloader workers
epochs = 50  # number of epochs
batch_size = 8  # batch size
learning_rate = 1e-3  # initial learning rate

##################################################
# Dataset/Path Config
##################################################
img_folder = "./data/training_data"  # training dataset path
csv_file = "./data/training_labels.csv"  # training label

split = 0.2  # percentage of validation set

# saving directory of .ckpt models
ckpt_dir = "./checkpoints/"
model_name = "resnext101_32x8d_aug_sg_lr_"
log_name = "train.log"  # Beta

##################################################
# Eval Config
##################################################
test_img_folder = "./data/testing_data/"  # testing images
result_pth = "./results/"
csv_name = model_name + ".csv"  # saving prediction result
