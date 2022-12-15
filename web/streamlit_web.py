import os
import glob
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import streamlit_authenticator as stauth
import mysql.connector

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

@st.experimental_singleton
def init_connection():
    return mysql.connector.connect(host="sql10.freesqldatabase.com", port=3306, database="sql10584922", user="sql10584922",
                                   password="AJXVfXtTFY", auth_plugin='mysql_native_password')

conn = init_connection()
cur = conn.cursor(buffered=False)

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
        # model_ft = models.resnet50(pretrained=use_pretrained)
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
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(Pretreined=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "mobilenet":
        """ MobileNet"""
        model_ft = models.mobilenet_v3_small(weights=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, 2)
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
        # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

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


def get_prediction(image, model, imagenet_class_index):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx][1]


@st.cache(suppress_st_warning=True)
def load_model():
    ckpt_path = f'web/blur-detection-mobilenet-5358.ckpt'
    model_ft = ClassificationModel.load_from_checkpoint(ckpt_path)
    # Since we are using our model only for inference, switch to `eval` mode:
    model_ft.eval()
    imagenet_class_index = json.load(open(f"{os.getcwd()}/web/data/imagenet_class_index.json"))

    return model_ft, imagenet_class_index


# ________________________________________________________________________________________________________
# callbacks para botões:
def confirma_callback():
    ev_string = f"ev_label_{st.session_state['name'][3]}"
    sql = f'UPDATE db_pecem SET {ev_string}=%s WHERE image_name=%s;'
    cur.execute(sql, (st.session_state.prediction, st.session_state.image_infos.iloc[st.session_state.count][0]))
    conn.commit()
    st.session_state.count += 1

def b1_callback():
    ev_label = 'Excelente'
    ev_string = f"ev_label_{st.session_state['name'][3]}"
    sql = f'UPDATE db_pecem SET {ev_string}=%s WHERE image_name=%s;'
    cur.execute(sql, (ev_label, st.session_state.image_infos.iloc[st.session_state.count][0]))
    conn.commit()
    st.session_state.count += 1

def b2_callback():
    ev_label = 'Bom'
    ev_string = f"ev_label_{st.session_state['name'][3]}"
    sql = f'UPDATE db_pecem SET {ev_string}=%s WHERE image_name=%s;'
    cur.execute(sql, (ev_label, st.session_state.image_infos.iloc[st.session_state.count][0]))
    conn.commit()
    st.session_state.count += 1

def b3_callback():
    ev_label = 'Ruim'
    ev_string = f"ev_label_{st.session_state['name'][3]}"
    sql = f'UPDATE db_pecem SET {ev_string}=%s WHERE image_name=%s;'
    cur.execute(sql, (ev_label, st.session_state.image_infos.iloc[st.session_state.count][0]))
    conn.commit()
    st.session_state.count += 1

def b4_callback():
    ev_label = 'Pessimo'
    ev_string = f"ev_label_{st.session_state['name'][3]}"
    sql = f'UPDATE db_pecem SET {ev_string}=%s WHERE image_name=%s;'
    cur.execute(sql, (ev_label, st.session_state.image_infos.iloc[st.session_state.count][0]))
    conn.commit()
    st.session_state.count += 1


def read_html():
    with open("web/index.html") as f:
        return f.read()


# ________________________________________________________________________________________________________
def main():
    """Funcao responsavel pela autenticacao do login"""

    senha_global = '123'
    names = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
    usernames = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
    passwords = [senha_global, senha_global, senha_global, senha_global, senha_global,
                 senha_global, senha_global, senha_global]

    hashed_passwords = stauth.Hasher(passwords).generate()
    credentials = {
        "usernames": {
            usernames[0]: {
                "name": names[0],
                "password": hashed_passwords[0]
            },
            usernames[1]: {
                "name": names[1],
                "password": hashed_passwords[1]
            },
            usernames[2]: {
                "name": names[2],
                "password": hashed_passwords[2]
            },
            usernames[3]: {
                "name": names[3],
                "password": hashed_passwords[3]
            },
            usernames[4]: {
                "name": names[4],
                "password": hashed_passwords[4]
            },
            usernames[5]: {
                "name": names[5],
                "password": hashed_passwords[5]
            },
            usernames[6]: {
                "name": names[6],
                "password": hashed_passwords[6]
            },
            usernames[7]: {
                "name": names[7],
                "password": hashed_passwords[7]
            }
        }
    }

    authenticator = stauth.Authenticate(credentials, 'some_cookie_name', 'some_signature_key',
                                        cookie_expiry_days=1)

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        col1, col2 = st.columns([1, 4])
        with col1:
            authenticator.logout('Logout', 'main')
        with col2:
            st.write('ID: ', st.session_state['name'])
        pagina_web()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


def pagina_web():
    """Função responsável por gerar a pagina web"""

    model, imagenet_class_index = load_model()
    # Descrição
    # st.write("This application knows the objects in an image , but works best when only one object is in the image")

    # Variaveis de session_state
    if 'image_infos' not in st.session_state:
        infos = pd.read_csv("web/db_pecem.csv")
        st.session_state.image_infos = infos
    if 'count' not in st.session_state:
        st.session_state.count = 0
    csv_infos = st.session_state.image_infos

    # pagina web
    if st.session_state.count < len(csv_infos) - 1:  # Verificar se a avaliação foi completa
        with st.sidebar:
            st.image(Image.open(csv_infos.iloc[st.session_state.count][4][1::]), "Excelente")
            st.image(Image.open(csv_infos.iloc[st.session_state.count][5][1::]), "Boa")
            st.image(Image.open(csv_infos.iloc[st.session_state.count][6][1::]), "Ruim")
            st.image(Image.open(csv_infos.iloc[st.session_state.count][7][1::]), "Pessima")

        img = Image.open('web/' + csv_infos['image_path'][st.session_state.count][2::])
        st.image(img)

        # botoes resultado e confimação
        c1, c2 = st.columns(2)
        with c1:
            prediction = get_prediction(img, model, imagenet_class_index)
            st.session_state.prediction = f'{prediction}'
            sql = 'UPDATE db_pecem SET pred_label=%s WHERE image_name=%s;'
            cur.execute(sql, (f'{prediction}', csv_infos.iloc[st.session_state.count][0]))
            conn.commit()
            resultado = st.button(f"Classificação: {prediction}", key="previsao")
        with c2:
            confirma_button = st.button("Confirmar", key="ok", on_click=confirma_callback)
        st.markdown("<hr>", unsafe_allow_html=True)

        # botões classificacao via streamlit
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            b1 = st.button("Excelente", key="exe", on_click=b1_callback)
        with c2:
            b2 = st.button("Boa", key="boa", on_click=b2_callback)
        with c3:
            b3 = st.button("Ruim", key="rum", on_click=b3_callback)
        with c4:
            b4 = st.button("Pessima", key="pes", on_click=b4_callback)

        # Estilos
        components.html(
            read_html(),
            height=0,
            width=0,
        )

    else:
        st.markdown("## A valiação concluida! ✅")


if __name__ == '__main__':
    main()

