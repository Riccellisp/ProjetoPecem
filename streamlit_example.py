import os
import json

import streamlit as st

from PIL import Image
from torchvision import models, transforms 

import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import classification_report
from torchmetrics import ConfusionMatrix


from torchvision import models
from torchvision import models
import torch.nn as nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def create_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        #model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = models.resnet50(Pretreined=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(Pretreined=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
        
    elif model_name == "mobilenet":
        """ MobileNet
        """
        model_ft = models.mobilenet_v3_small(weights=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs,2)
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# Initialize the model for this run
# model_ft, input_size = create_model(model_name, num_classes, feature_extract, use_pretrained=True)

# # Print the model we just instantiated
# print(model_ft)
# 
class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams["num_classes"], False, use_pretrained=True)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        #self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        self.model.eval()
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
#         confmat = ConfusionMatrix(num_classes=2).to(device)
#         matrix = confmat(preds, labels)
#         tp = matrix[0,0]
#         tn = matrix[1,1]
#         fp = matrix[1,0]
#         fn = matrix[0,1]
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        
        self.log("test_acc", acc)
#         self.log("tp", tp)
#         self.log("tn", tn)
#         self.log("fp", tp)
#         self.log("fn", tn)


def get_prediction(image, model, imagenet_class_index):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx][1]


def transform_image(image):
    """ Transform image to fit model

    Args:
        image (image): Input image from the user

    Returns:
        tensor: transformed image 
    """
    transformation = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return transformation(image).unsqueeze(0)



@st.cache
def load_model():

    ckpt_path = f'C:\\Users\\LESC\\Desktop\\ProjetoPecem\\blur-detection-mobilenet-5358.ckpt'
    model_ft = ClassificationModel.load_from_checkpoint(ckpt_path)
    # Since we are using our model only for inference, switch to `eval` mode:
    model_ft.eval()

    imagenet_class_index = json.load(open(f"{os.getcwd()}/data/imagenet_class_index.json"))
    
    return model_ft, imagenet_class_index


def main():
    
    st.title("Predict objects in an image")
    st.write("This application knows the objects in an image , but works best when only one object is in the image")

    
    model, imagenet_class_index = load_model()
    
    image_file  = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file:
       
        left_column, right_column = st.beta_columns(2)
        left_column.image(image_file, caption="Uploaded image", use_column_width=True)
        image = Image.open(image_file)

        pred_button = st.button("Predict")
        
        
        if pred_button:

            prediction = get_prediction(image, model, imagenet_class_index)
            right_column.title("Prediction")
            right_column.write(prediction)

if __name__ == '__main__':
    main()