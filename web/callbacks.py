import streamlit as st
import pygsheets

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

reset_password_flag = True
