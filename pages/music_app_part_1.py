#!/usr/bin/env python
# coding: utf-8

# In[152]:


import json
import pandas as pd
import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import dcc, html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from dash import dash_table
from dash.dash_table import DataTable, FormatTemplate
import plotly.express as px
from dash.dash_table.Format import Group
import seaborn as sns
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from sklearn import preprocessing
import wordcloud
from wordcloud import WordCloud as wc, ImageColorGenerator
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from os import path, getcwd
from nltk.corpus import stopwords
import nltk
import io
import base64
import pathlib

dash.register_page(__name__, name='Get your song')

#get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
dataframe = pd.read_csv(DATA_PATH.joinpath("rock.csv"))
# dataframe = pd.read_csv('D://MSBA//Extra Projects//Data//New folder//rock.csv')

# creating a categorical variable "rock_era"
bins = [dataframe['release_date'].min() - 1, 1979, 1999, dataframe['release_date'].max()]
labels = ['Old Rock', 'Sweet Rock', 'Modern Rock']
dataframe['rock_era'] = pd.cut(dataframe['release_date'], bins, labels=labels)

# Scaling the numerical columns

min_max_scaler = preprocessing.MinMaxScaler()
numerical_columns = dataframe.select_dtypes(include=['float', 'int']).columns.tolist()
remove_columns = ['index', 'release_date', 'danceability.1', 'time_signature']
numerical_columns = [col for col in numerical_columns if col not in remove_columns]
for col in numerical_columns:
    dataframe[col] = min_max_scaler.fit_transform(dataframe[col].values.reshape(-1, 1))


#  rounding values of dataframe

numerical_columns = dataframe.select_dtypes(include=[float]).columns.tolist()
dataframe[numerical_columns] = dataframe[numerical_columns].applymap(lambda x: round(x, 2))


dataframe = dataframe.drop(['danceability.1'], axis=1)


# Instantiating App


# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE],
#
#                 meta_tags=[{'name': 'viewport',
#                             'content': 'width=device-width, initial-scale = 0.9, maximum-scale = 1.9, minimum-scale = 0.5'}])

# dataframe = pd.read_csv('D://MSBA//Extra Projects//Data//New folder//rock.csv')
# bins = [dataframe['release_date'].min()-1, 1981, 2001, dataframe['release_date'].max()]
# labels =['Old Rock', 'Sweet Rock', 'Modern Rock']
# dataframe['rock_era'] = pd.cut(dataframe['release_date'], bins, labels = labels)


layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1('Get your desired rock songs',
                        style={"textAlign": 'center',
                               "background-color": 'rgb(40, 140, 170)',
                               "margin": '40px 5px 15px 5px', "color": 'white',
                               'font-size': '25px'}),
                html.Hr(style={'textAlign': 'center',
                               "height": '10px', 'margin': '0px 0px 0px 0px'}),
                html.Br(),
                html.Br()
            ])

            #             html.Div(
            #             style ={'borderLeft': '5px solid rgb(40, 140, 170)',
            #                    "height": '500%',
            #                    'transform': 'translateX(0%)',
            #                    'top': '-5'})

            , width={'size': 12, 'order': 1})
    ]),

    dbc.Row([
        dbc.Col(children = [
            html.H4('Selection',
                    style={"textAlign": "left", "margin": "0px 0px 0px 15px"}),
            html.Br(),
            html.H6('Select rock era',
                    style={'textAlign': 'left', "margin": "0px 0px 10px 29px", 'font-size': "15px"}),
            dcc.Dropdown(id='rock_era_id',
                         options=[{"label": i, 'value': i} for i in dataframe['rock_era'].unique()],
                         value='Sweet Rock',
                         style={"margin": "0px 0px 10px 15px", "width": "130px", 'font-size': "13px"}
                         ),
            html.Img(src='assets/pinkfloyed.jpg',
                     style={'height': '270px', 'width': '90%'})

            #                     dcc.Slider(
            #                         min = dataframe['popularity'].min(),
            #                         max = dataframe['popularity'].max(),
            #                         value = 70,
            #                         step = 1,
            #                         vertical = False)

        ],xs = 12, sm = 12, md = 6, lg=5, xl=5, xxl=5),

        dbc.Col([
            html.Br(),
            html.H5('Set value of song attributes',
                    style={'textAlign': 'left', "margin": "40px -10px 10px 25px", 'font-size': '14px'}
                    ),
            #                   html.H6('Popularity index',
            #                      style ={"font-size": "12px",'textAlign':'left', "margin": "0px -10px 0px 10px"}
            #                      )]
            html.H6('Popularity index',
                    style={"font-size": "12px", 'textAlign': 'left', "margin": "0px 0px 0px 25px"}
                    ),
            dcc.RangeSlider(
                id='popularity_slider',
                min=dataframe['popularity'].min(),
                max=dataframe['popularity'].max(),
                value=[0.6, 0.7],
                step=0.1,
                vertical=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.H6(
                'Danceability index',
                style={"font-size": "12px", 'textAlign': 'left', "margin": "0px 0px 0px 25px"}
            ),
            dcc.RangeSlider(
                id='danceability_slider',
                min=dataframe['danceability'].min(),
                max=dataframe['danceability'].max(),
                value=[0.1, 0.4],
                step=0.1,
                vertical=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.H6('Acousticness index',
                    style={"font-size": "12px", 'textAlign': 'left', "margin": "0px 0px 0px 25px"}
                    ),
            dcc.RangeSlider(
                id='acoustic_slider',
                min=dataframe['acousticness'].min(),
                max=dataframe['acousticness'].max(),
                value=[0.4, 0.6],
                step=0.1,
                vertical=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.H6(
                'Liveness index',
                style={"font-size": "12px", 'textAlign': 'left', "margin": "0px 0px 0px 25px"}
            ),
            dcc.RangeSlider(
                id='liveness index',
                min=dataframe['liveness'].min(),
                max=dataframe['liveness'].max(),
                value=[0.1, 0.3],
                step=0.1,
                vertical=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

        ], xs = 12, sm = 12, md = 6, lg=5, xl=5, xxl=5)
# width={'size': 7, 'order': 3}


    ]),
    dbc.Row([
        dbc.Col(children=[
            html.Br(),
            html.Br(),
            html.H5(
                'List of songs based on selected parameters',
                style={"font-size": "15px", "textAlign": 'left', "color": "blue", "margin": '5px 0px 10px 5px'}),
            html.Div(id='table_container', style={'margin': "5px", 'overflowX': "scroll"})

        ])
    ])

],fluid=True)

@callback(
    Output(component_id='table_container',
           component_property='children'),

    Input(component_id='rock_era_id',
          component_property='value'),

    Input(component_id='popularity_slider',
          component_property='value'),

    Input(component_id='danceability_slider',
          component_property='value'),

    Input(component_id='acoustic_slider',
          component_property='value'),

    Input(component_id='liveness index',
          component_property='value')
)
def get_table(option, popularity_slider, danceability_slider, acousticness_slider, liveness_slider):
    if (option) and (popularity_slider) and (danceability_slider) and (acousticness_slider) and (liveness_slider):
        selection = option
        global dataframe

        dataframe = dataframe.copy(deep=True)
        dataframe1 = dataframe[
            ['name', 'artist', 'release_date', 'popularity', 'danceability', 'acousticness', 'liveness', 'rock_era']]
        dataframe2 = dataframe1[dataframe1['rock_era'] == selection].sort_values(by='popularity', ascending=False)
        selection = popularity_slider
        dataframe3 = dataframe2[(dataframe2['popularity'] >= selection[0]) & (dataframe2['popularity'] <= selection[1])]
        selection = danceability_slider
        dataframe4 = dataframe3[
            (dataframe3['danceability'] >= selection[0]) & (dataframe3['danceability'] <= selection[1])]
        selection = acousticness_slider
        dataframe5 = dataframe4[
            (dataframe4['acousticness'] >= selection[0]) & (dataframe4['acousticness'] <= selection[1])]

        selection = liveness_slider
        dataframe6 = dataframe5[(dataframe5['liveness'] >= selection[0]) & (dataframe5['liveness'] <= selection[1])]
        dataframe6 = dataframe6.drop(['rock_era'], axis=1)

        df_dash = DataTable(data=dataframe6.to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in dataframe6.columns],
                            cell_selectable=False,
                            sort_action='native',
                            page_action='native',
                            page_current=0,
                            page_size=8,
                            style_cell={"textAlign": 'left'})

        #         dataframe_wc = dataframe1['artist']=='option'
        #         text = ' '.join([artist for artist in dataframe_wc])
        #         mask_image_path ="D://MSBA//extra//rocka.png"
        #         mask_image = np.array(Image.open(mask_image_path))
        #         wordcloud = wc(mask = mask_image, background_color ='black', min_font_size = 10,
        #                collocations = True, colormap ='BuPu',
        #                max_words =150).generate(text)
        #         output_image_path = "D://MSBA//extra//rockb.png"
        #         wordcloud.to_file(output_image_path)

        #         plt.figure(figsize=(8,7))
        #         plt.imshow(wordcloud, interpolation = 'bilinear')
        #         plt.axis ('off')
        #         plt.title ("Tob Rock Artist")
        #         plt.savefig("rock3.png", format="png")
        #         plt.show()
        #         buf = io.BytesIO()
        #         plt.savefig(buf, format = "png")
        #         plt.close(fig)
        #         buf.seek(0)

        #         pic = base64.b64encode(buf.read()).decode('utf-8')

        return df_dash,



# if __name__ == '__main__':
#     app.run(debug=True, port=8053, jupyter_mode='external')

#