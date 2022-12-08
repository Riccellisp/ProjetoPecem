import os
import glob
import json
import streamlit as st
import streamlit.components.v1 as components

try: 
    import cv2 
except ImportError: 
    import pip 
    pip.main(['install', '--user', 'opencv-python']) 
    import cv2  

from PIL import Image
from torchvision import models, transforms 
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import classification_report
from torchmetrics import ConfusionMatrix
from torchvision import models  # torchvision
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
        """ MobileNet"""
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


def loadImages(path):
    '''
    function to load all images from a folder, resize images to size 360x240 and return a list
    if necessary, we might merge these function with apply_metric_list to save memory
    @param path: str path for root_dataset_folder
    '''
    # root_dataset_folder
    #   # child_cam_folder
    #       # grandson_img_file
    image_list = []
    for cam_folder in sorted(os.listdir(path)):
        for class_folder in sorted(os.listdir(path+"\\"+cam_folder)):
            for image_paths in sorted(glob.glob(path+"\\"+cam_folder+"\\"+f"{class_folder}"+"\\*")):
                print(image_paths)
                image = cv2.cvtColor(cv2.imread(image_paths), cv2.COLOR_BGR2RGB)
                image_list.append(image)

    return image_list

def main():
    """Função responsável por gerar a pagina web"""
    model, imagenet_class_index = load_model()
    st.title("Sistema de Classificação Manual")

    # st.write("This application knows the objects in an image , but works best when only one object is in the image")
    with st.sidebar:
        imagens_fixas=[cv2.cvtColor(cv2.imread("dataset_pecem\cam_77_3\Bom\Imagem13.jpg"),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread("dataset_pecem\cam_77_3\Excelente\Imagem11.jpg"),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread("dataset_pecem\cam_77_3\Pessimo\Imagem17.jpg"),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread("dataset_pecem\cam_77_3\Ruim\Imagem15.jpg"),cv2.COLOR_BGR2RGB)]
        st.image(imagens_fixas[0])
        st.image(imagens_fixas[1])
        st.image(imagens_fixas[2])
        st.image(imagens_fixas[3])
            
    with st.container():
        images=loadImages("dataset_pecem")
        image_file  = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        # botões das classes:
        # html_string = """
        #                 <div id='menu_classificacao'>
        #                     <button type='submit'>Excelente</button>
        #                     <button type='submit'>Boa</button>
        #                     <button type='submit'>Ruim</button>
        #                     <button type='submit'>Pessima</button>
        #                 </div>"""
        # st.markdown(html_string, unsafe_allow_html=True)

    if image_file:
       
        left_column, right_column = st.columns(2)
        left_column.image(image_file, caption="Uploaded image", use_column_width=True)
        image = Image.open(image_file)
        # pred_button = st.button("Predict")
        
        
        prediction = get_prediction(image, model, imagenet_class_index)
        # botoes resultado e confimação
        c1,c2=st.columns(2)
        with c1:
            resultado=st.button(f"{prediction}", key="previsao", help=None, on_click=None, args=None, kwargs=None, disabled=False)
            
        with c2:
            st.button("Confirmar", key="ok", help=None, on_click=None, args=None, kwargs=None, disabled=False)

        st.markdown("<hr>",unsafe_allow_html=True)
        # botões classificacao via streamlit    
        b1,b2,b3,b4=st.columns(4)
        with b1:
            st.button("Excelente", key="exe", help=None, on_click=None, args=None, kwargs=None, disabled=False)
        with b2:
            st.button("Boa", key="boa", help=None, on_click=None, args=None, kwargs=None, disabled=False)
        with b3:
            st.button("Ruim", key="rum", help=None, on_click=None, args=None, kwargs=None, disabled=False)
        with b4:
            st.button("Pessima", key="pes", help=None, on_click=None, args=None, kwargs=None, disabled=False)

    


if __name__ == '__main__':
    main()
