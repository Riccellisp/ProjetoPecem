import os
import glob
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import streamlit_authenticator as stauth
#import mysql.connector
import pygsheets
import yaml
from yaml import SafeLoader
from PIL import Image

st.set_page_config(
    page_title="Avaliação"
)


# from torchvision import models, transforms
# import pytorch_lightning as pl
# import torch.optim as optim
# from sklearn.metrics import classification_report
# from torchmetrics import ConfusionMatrix
# from torchvision import models  # torchvision
# from torchvision import models
# import torch.nn as nn

# @st.experimental_singleton
# def init_connection():
#     return mysql.connector.connect(host="sql10.freesqldatabase.com", port=3306, database="sql10584922", user="sql10584922",
#                                    password="AJXVfXtTFY", auth_plugin='mysql_native_password')

# conn = init_connection()
# conn.reconnect()
# cur = conn.cursor(buffered=True)

# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
#
#
# def create_model(model_name, num_classes, feature_extract, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0
#
#     if model_name == "resnet":
#         """ Resnet50
#         """
#         # model_ft = models.resnet50(pretrained=use_pretrained)
#         model_ft = models.resnet50(Pretreined=True)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 224
#
#     elif model_name == "vgg":
#         """ VGG16_bn
#         """
#         model_ft = models.vgg16_bn(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
#         input_size = 224
#
#     elif model_name == "squeezenet":
#         """ Squeezenet
#         """
#         model_ft = models.squeezenet1_0(Pretreined=True)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#         model_ft.num_classes = num_classes
#         input_size = 224
#
#     elif model_name == "mobilenet":
#         """ MobileNet"""
#         model_ft = models.mobilenet_v3_small(weights=None)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[3].in_features
#         model_ft.classifier[3] = nn.Linear(num_ftrs, 2)
#         model_ft.num_classes = num_classes
#         input_size = 224
#
#     elif model_name == "densenet":
#         """ Densenet
#         """
#         model_ft = models.densenet201(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
#         input_size = 224
#
#     else:
#         print("Invalid model name, exiting...")
#         exit()
#
#     return model_ft
#
#
# # Initialize the model for this run
# # model_ft, input_size = create_model(model_name, num_classes, feature_extract, use_pretrained=True)
#
# # # Print the model we just instantiated
# # print(model_ft)
# #
# class ClassificationModel(pl.LightningModule):
#     def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
#         """
#         Inputs:
#             model_name - Name of the model/CNN to run. Used for creating the model (see function below)
#             model_hparams - Hyperparameters for the model, as dictionary.
#             optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
#             optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
#         """
#         super().__init__()
#         # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
#         self.save_hyperparameters()
#         # Create model
#         self.model = create_model(model_name, model_hparams["num_classes"], False, use_pretrained=True)
#         # Create loss module
#         self.loss_module = nn.CrossEntropyLoss()
#         # Example input for visualizing the graph in Tensorboard
#         # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
#
#     def forward(self, imgs):
#         # Forward function that is run when visualizing the graph
#         return self.model(imgs)
#
#     def configure_optimizers(self):
#         # We will support Adam or SGD as optimizers.
#         if self.hparams.optimizer_name == "Adam":
#             # AdamW is Adam with a correct implementation of weight decay (see here
#             # for details: https://arxiv.org/pdf/1711.05101.pdf)
#             optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
#         elif self.hparams.optimizer_name == "SGD":
#             optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
#         else:
#             assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
#
#         # We will reduce the learning rate by 0.1 after 100 and 150 epochs
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
#         return [optimizer], [scheduler]
#
#     def training_step(self, batch, batch_idx):
#         # "batch" is the output of the training data loader.
#         imgs, labels = batch
#         preds = self.model(imgs)
#         loss = self.loss_module(preds, labels)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()
#
#         # Logs the accuracy per epoch to tensorboard (weighted average over batches)
#         self.log("train_acc", acc, on_step=False, on_epoch=True)
#         self.log("train_loss", loss)
#         return loss  # Return tensor to call ".backward" on
#
#     def validation_step(self, batch, batch_idx):
#         imgs, labels = batch
#         preds = self.model(imgs)
#         loss = self.loss_module(preds, labels)
#         preds = preds.argmax(dim=-1)
#         acc = (labels == preds).float().mean()
#         # By default logs it per epoch (weighted average over batches)
#         self.log("val_acc", acc)
#         self.log("val_loss", loss)
#
#     def test_step(self, batch, batch_idx):
#         imgs, labels = batch
#         self.model.eval()
#         preds = self.model(imgs).argmax(dim=-1)
#         acc = (labels == preds).float().mean()
#         #         confmat = ConfusionMatrix(num_classes=2).to(device)
#         #         matrix = confmat(preds, labels)
#         #         tp = matrix[0,0]
#         #         tn = matrix[1,1]
#         #         fp = matrix[1,0]
#         #         fn = matrix[0,1]
#         # By default logs it per epoch (weighted average over batches), and returns it afterwards
#
#         self.log("test_acc", acc)
#
#
# #         self.log("tp", tp)
# #         self.log("tn", tn)
# #         self.log("fp", tp)
# #         self.log("fn", tn)
#
# def transform_image(image):
#     """ Transform image to fit model
#     Args:
#         image (image): Input image from the user
#     Returns:
#         tensor: transformed image
#     """
#     transformation = transforms.Compose([transforms.Resize(255),
#                                          transforms.CenterCrop(224),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(
#                                              [0.485, 0.456, 0.406],
#                                              [0.229, 0.224, 0.225])])
#     return transformation(image).unsqueeze(0)
#
#
# def get_prediction(image, model, imagenet_class_index):
#     tensor = transform_image(image=image)
#     outputs = model.forward(tensor)
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx][1]
#
#
# @st.cache(suppress_st_warning=True)
# def load_model():
#     ckpt_path = f'web/blur-detection-mobilenet-5358.ckpt'
#     #ckpt_path = f'blur-detection-mobilenet-5358.ckpt'
#     model_ft = ClassificationModel.load_from_checkpoint(ckpt_path)
#     # Since we are using our model only for inference, switch to `eval` mode:
#     model_ft.eval()
#     imagenet_class_index = json.load(open(f"{os.getcwd()}/web/data/imagenet_class_index.json"))
#     #imagenet_class_index = json.load(open(f"{os.getcwd()}/data/imagenet_class_index.json"))
#
#     return model_ft, imagenet_class_index


# ________________________________________________________________________________________________________
# callbacks para botões:
def voltar_callback():
    if st.session_state.count == 0:
        st.write("Realize ao menos uma avaliação!")
    else:
        st.session_state.count -= 1

        ev_label = ''
        ev_string = f"ev_label_{st.session_state['name']}"

        # gc = pygsheets.authorize(service_file='dbpecem-cf62256085c7.json')
        gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')
        sh = gc.open('teste_pecem')
        wks = sh[0]
        csv_infos = wks.get_as_df()
        # csv_infos = st.session_state.image_infos
        # csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
        csv_infos.loc[csv_infos['image_path'] == csv_infos['image_path'][st.session_state.count], ev_string] = ev_label
        wks.set_dataframe(csv_infos, (0, 0))


def b1_callback():
    ev_label = 'Excelente'
    ev_string = f"ev_label_{st.session_state['name']}"
    
    #gc = pygsheets.authorize(service_file='dbpecem-cf62256085c7.json')
    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')
    sh = gc.open('teste_pecem')
    wks = sh[0]
    csv_infos = wks.get_as_df()
    #csv_infos = st.session_state.image_infos
    # csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
    csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], ev_string] = ev_label
    wks.set_dataframe(csv_infos,(0,0))

    st.session_state.count += 1

def b2_callback():
    ev_label = 'Bom'
    ev_string = f"ev_label_{st.session_state['name']}"
    
    #gc = pygsheets.authorize(service_file='dbpecem-cf62256085c7.json')
    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')
    sh = gc.open('teste_pecem')
    wks = sh[0]
    csv_infos = wks.get_as_df()
    #csv_infos = st.session_state.image_infos
    # csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
    csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], ev_string] = ev_label
    wks.set_dataframe(csv_infos,(0,0))

    st.session_state.count += 1

def b3_callback():
    ev_label = 'Ruim'
    ev_string = f"ev_label_{st.session_state['name']}"
    
    #gc = pygsheets.authorize(service_file='dbpecem-cf62256085c7.json')
    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')
    sh = gc.open('teste_pecem')
    wks = sh[0]
    csv_infos = wks.get_as_df()
    #csv_infos = st.session_state.image_infos
    # csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
    csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], ev_string] = ev_label
    wks.set_dataframe(csv_infos,(0,0))

    st.session_state.count += 1

def b4_callback():
    ev_label = 'Pessimo'
    ev_string = f"ev_label_{st.session_state['name']}"
    
    #gc = pygsheets.authorize(service_file='dbpecem-cf62256085c7.json')
    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')
    sh = gc.open('teste_pecem')
    wks = sh[0]
    csv_infos = wks.get_as_df()
    #csv_infos = st.session_state.image_infos
    # csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
    csv_infos.loc[csv_infos['image_path']==csv_infos['image_path'][st.session_state.count], ev_string] = ev_label
    wks.set_dataframe(csv_infos,(0,0))

    st.session_state.count += 1

def read_html():
    #with open("web/index.html") as f:
    with open("web/index.html") as f:
        return f.read()
# ________________________________________________________________________________________________________

# Conta de serviço:
# paginaweb-teste@sheets-lesc.iam.gserviceaccount.com
# id: paginaweb-teste

def main():
    """Funcao responsavel por autenticacao do login"""

    with open('web/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')  # se conecta com a planilha
    sh = gc.open('pecem_credenciais')  #
    # wks = sh[0]                                                           # worksheet 1
    logins = sh[1]
    pw_df = logins.get_as_df()

    names = pw_df.loc[0:13, 'name'].values
    usernames = pw_df.loc[0:13, 'username'].values
    hashed_passwords = pw_df.loc[0:13, 'password'].values

    # A senha padrao e '123'

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
            },
            usernames[8]: {
                "name": names[8],
                "password": hashed_passwords[8]
            },
            usernames[9]: {
                "name": names[9],
                "password": hashed_passwords[9]
            },
            usernames[10]: {
                "name": names[10],
                "password": hashed_passwords[10]
            },
            usernames[11]: {
                "name": names[11],
                "password": hashed_passwords[11]
            },
            usernames[12]: {
                "name": names[12],
                "password": hashed_passwords[12]
            },
            usernames[13]: {
                "name": names[13],
                "password": hashed_passwords[13]
            }
        }
    }

    authenticator = stauth.Authenticate(
        credentials,
        'cookie_name', 'signature_key',
        cookie_expiry_days=1
        )

    name, authentication_status, username = authenticator.login('Login', 'main')
    # Verificar se a avaliação foi completa:

    st.session_state.authentication = authenticator

    if name == None:
        st.session_state.count = 0

    if 'reset_flag' not in st.session_state:
        st.session_state.reset_flag = False

    if authentication_status:
        if st.button('Trocar Senha') or st.session_state.reset_flag:
            st.session_state.new_password = st.text_input("Escreva a nova senha e aperte Enter")
            if st.session_state.reset_flag and not st.session_state.new_password == '':
                gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')  # se conecta com a planilha
                sh = gc.open('pecem_credenciais')  #
                # wks = sh[0]                                                           # worksheet 1
                logins = sh[1]
                pw_df = logins.get_as_df()
                hashed_new_password = stauth.Hasher([st.session_state.new_password]).generate()
                pw_df.loc[pw_df['name'] == st.session_state['name'], 'password'] = hashed_new_password
                logins.set_dataframe(pw_df, (0, 0))  # atualiza a planilha

                st.write('Senha alterada!')

                st.session_state.reset_flag = False
            else:
                st.session_state.reset_flag = True

    if authentication_status:
        pagina_web()
    elif authentication_status == False:
        st.error('Email/senha está incorreto')
    elif authentication_status == None:
        st.warning('Favor inserir seu email e senha')


def pagina_web():
    """Função responsável por gerar a pagina web"""
    # model, imagenet_class_index = load_model()

    gc = pygsheets.authorize(service_file='web/dbpecem-cf62256085c7.json')

    # Variaveis de session_state
    sh = gc.open('teste_pecem')
    wks, passwords= sh[0], sh[1]
    infos = wks.get_as_df()  # create the dataframe
    st.session_state.image_infos = infos
    if 'count' not in st.session_state:
        st.session_state.count = 0

    # pagina web
    if st.session_state.count < len(st.session_state.image_infos):  # Verificar se a avaliação foi completa
        while not (st.session_state.image_infos.loc[st.session_state.count, f"ev_label_{st.session_state.name}"]) == "":
            st.session_state.count += 1
        # header acima da imagem principal
        col1, col2, col3,col4=st.columns([3,4,1,1])
        with col1:
            st.session_state.authentication.logout('Logout', 'main')
        print("Contator no inicio: ",st.session_state.count)
        with col2:
            total=len(st.session_state.image_infos.loc[st.session_state.image_infos['cam_num']==st.session_state.image_infos.iloc[st.session_state.count][2]])-1
            # concluido=st.session_state.count 
            # if st.session_state.image_infos.iloc[0][2].split('_')[0]=='cam' and st.session_state.image_infos.iloc[st.session_state.count][2]!=st.session_state.image_infos.iloc[st.session_state.count-1][2]:
            #     concluido=0
            # st.button(f"{st.session_state.image_infos.iloc[st.session_state.count][2]} | Concluídas: {concluido}/{total}")  
            st.button(f"{st.session_state.image_infos.iloc[st.session_state.count][2]} | Total de Imagens: {total}")  
        with col3:
            logoPecem=Image.open("web/logos/logoPecem.jpg")
            st.image(logoPecem,width=60)
        with col4:
            logoLesc=Image.open("web/logos/logoLesc.png")
            st.image(logoLesc,width=40)

        with st.sidebar:
            st.image(Image.open(st.session_state.image_infos.iloc[st.session_state.count][4][1::]), "Excelente")
            st.image(Image.open(st.session_state.image_infos.iloc[st.session_state.count][5][1::]), "Boa")
            st.image(Image.open(st.session_state.image_infos.iloc[st.session_state.count][6][1::]), "Ruim")
            st.image(Image.open(st.session_state.image_infos.iloc[st.session_state.count][7][1::]), "Pessima")

        img = Image.open('web/' + st.session_state.image_infos['image_path'][st.session_state.count][2::])
        # prediction = get_prediction(img, model, imagenet_class_index)
        # st.session_state.prediction = f'{prediction}'
        st.image(img)

        # botões classificacao via streamlit
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("Excelente", key="exe", on_click=b1_callback)
        with c2:
            st.button("Boa", key="boa", on_click=b2_callback)
        with c3:
            st.button("Ruim", key="rum", on_click=b3_callback)
        with c4:
            st.button("Pessima", key="pes", on_click=b4_callback)

        st.markdown("<hr>", unsafe_allow_html=True)

        # botao de voltar
        c1, c2, c3, c4=st.columns(4)
        with c4:
            st.button("Voltar", key="back", on_click=voltar_callback)
        #sql = 'UPDATE db_pecem SET pred_label=%s WHERE image_name=%s;'
        #cur.execute(sql, (f'{prediction}', st.session_state.image_infos.iloc[st.session_state.count][0]))
        #conn.commit()
        #print(st.session_state.image_infos['image_path'][st.session_state.count])
        #st.session_state.image_infos.loc[st.session_state.image_infos['image_path']==st.session_state.image_infos['image_path'][st.session_state.count], 'pred_label'] = st.session_state.prediction
        #wks.set_dataframe(st.session_state.image_infos,(0,0))

        # Estilos
        components.html(
            read_html(),
            height=0,
            width=0,
        )

        print("Contador no fim:",st.session_state.count)

    else:
        st.markdown("## A valiação foi concluida! ✅")


if __name__ == '__main__':
    main()

