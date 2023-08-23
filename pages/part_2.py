#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import pandas as pd
import numpy as np

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
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go
import joypy
import pathlib


# #### Instantiate the app

# In[2]:
dash.register_page(__name__, path='/',name ='Analytics-1')
#get relative path of datasets
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
dataframe = pd.read_csv(DATA_PATH.joinpath("rock.csv"))

# dataframe = pd.read_csv('D://MSBA//Extra Projects//Data//New folder//rock.csv')


# In[3]:


bins = [dataframe['release_date'].min()-1, 1981, 2001, dataframe['release_date'].max()]
labels =['Old Rock', 'Sweet Rock', 'Modern Rock']
dataframe['rock_era'] = pd.cut(dataframe['release_date'], bins, labels = labels)


# #### Transform the variables to make their valies ranging from 0-1

# In[4]:


min_max_scaler = preprocessing.MinMaxScaler()


# In[5]:


dataframe.columns


# In[6]:


columns = ['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 
           'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'length']


# In[7]:


for col in columns:
    dataframe[col] = min_max_scaler.fit_transform(dataframe[col].values.reshape(-1, 1))


# In[8]:


dataframe = dataframe.drop(['danceability.1'], axis=1)


# In[9]:


dataframe.columns


# In[10]:


# columns = ['danceability', 'acousticness', 'liveness', 'energy', 'valence', "popularity", 'length', 'loudness', 'tempo', 'speechiness']




# In[11]:


columns1 = ['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'length', 'rock_era']


# In[12]:


data = dataframe[columns1]



# In[13]:


### rounding numerical values 

numerical_columns = data.select_dtypes(include =[float, int]).columns.tolist()
print(numerical_columns)
data[numerical_columns]= data[numerical_columns].applymap(lambda x:round(x, 3))


# In[14]:


data.head()


# In[15]:


# app = dash.Dash(__name__, external_stylesheets =[dbc.themes.PULSE],
#                meta_tags = [{'name':'viewport',
#                             'content':'width = device-width, initial-scale = 0.8, '
#                                       'maximum-scale = 1.0, minimum-scale=0.5'}])



layout = dbc.Container([
    dbc.Row([
        dbc.Col(children =[
        html.Br(),
        html.H1('Analytics-1',
                style={'textAlign': 'center', 'font-size': '21px', "margin": '0px 0px 0px 0px'}
        
        ),
        html.Br(),
      
        ], xs=12, sm=12, md=11, lg=11, xl=11, xxl=11)
        
    ], justify='center'),
    
    dbc.Row([

        dbc.Col(children=[
            html.H3('Attribute Selection Controls', style = {'textAlign':'center', 'font-size':'20px', "color":'SeaGreen', "margin":'0px 0px 0px 0px'}),

            html.Hr(style ={"color":'red', 'borderWidth':'2px', 'margin': '0px 0px 0px 0px'})
              ], xs = 12, sm = 12, md = 12, lg=12, xl=12, xxl= 12),
            
        dbc.Col([
            dcc.Checklist(id ='checklist', options=[col for col in data.columns if col !='rock_era'], inline = False,
                          value = ['acousticness', 'popularity', 'energy', 'danceability'],
#                                    ,'popularity', 'energy', 'danceability'],
                          labelStyle = {"display":'inline-block', 'align-items': 'middle', "width": '30%','margin': '0px 0px 0px 0px'},
                         labelClassName ='mr-4, text-info', style = {'transform':'scale(0.7)'}),
            
        ], width ={'size':'8px', 'offset':2})
            

            
         

        
    ], justify ='center'),
        
        
        
    dbc.Row([
        dbc.Col(children=[
            html.Br(),
            html.Br(),
            html.H4('Distributions Plot',
                   style = {'font-size': "20px", 'color':'#1214A1', 'textAlign': 'left' }),
            
            html.Br(),
            html.Br(), 
            html.Img(id = 'matplot_graph', style = {'max-width':'100%', "margin":'0px 0px 0px -30px'})
            
            
            
            
            ],xs = 10, sm= 10, md = 6, lg=5, xl=5, xxl=5),
        

    

    
    
        dbc.Col( children=[
            html.Br(),
            
            
            html.H4('Correlation Heatmap', style = {'font-size': "18px", 'color':'#1214A1', 'margin':'23px 0px 0px 70px'}),
            # html.H6('Click on cell to get the corresponding scatterplot',
            #         style = {'font-size': "10px", 'color':'red','textAlign': 'center'}),

            dcc.Graph(id = 'heatmap',  config={'responsive': True}),
            html.H6('Click on cell to get the corresponding scatterplot',
                    style = {'font-size': "10px", 'color':'red','textAlign': 'center'}),
            
            html.H4('Scatter Plot', style = {'font-size': "18px", 'color':'#1214A1', 'margin':'23px 0px 0px 70px'}),
            
#             html.P(id = 'click-text'),
            dcc.Graph(id= 'scatter_plot',  config={'responsive': True})
        
        ] ,xs = 12, sm = 12, md = 6, lg=5, xl=5, xxl=5)
    ],justify = 'around'),
    
    
    

    
   ])



@callback(
    Output(component_id ='matplot_graph', component_property ='src'),
    Output(component_id='heatmap', component_property = 'figure'),
    Input(component_id = 'checklist', component_property = 'value'), 
    
)
    


    
def update_ridgeplot(checkbox):

    if checkbox:
        data1 = data.copy(deep= True)
#         column = data1.columns
        columns = [col for col in checkbox]
       
        columns.append('rock_era')
        
        

        dataframe= data1[columns]
        
      
        
        
        fig, ax = joypy.joyplot (dataframe, by = 'rock_era',
                        figsize = (5.5,7.5), alpha = 0.8, labels =labels)
        legend = ax[0].legend()
        legend.set_title('')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_visible(False)
        legend.get_title().set_fontsize(6)
        for label in legend.get_texts():
            label.set_fontsize(6)
        
        
        plt.rc('font', size = 10)
        
  
        
        
        buf = io.BytesIO()
        plt.savefig(buf, format = "png")
        plt.close(fig)
        buf.seek(0)

        pic = base64.b64encode(buf.read()).decode('utf-8')
        df_corr= round(dataframe.drop(['rock_era'], axis =1).corr(numeric_only = False, method = 'pearson'), 2)
        
        heatmap = px.imshow(df_corr, text_auto = True, color_continuous_scale = 
                            'rdbu_r').update_layout(height = 500).update_coloraxes(showscale = False)

        
        
        return "data:image/png;base64,{}".format(pic), heatmap
        
    
    
@callback(
#     Output('click-text', 'children'),
    Output('scatter_plot', 'figure'),
    Input('heatmap', 'clickData')

)


    
    
    
    
def capture_click_data(clickData):
    data2 = data.copy(deep = True)
    if clickData is None:
        return px.scatter(data_frame = data2, x = 'acousticness',
                              y='energy', 
                             color = "rock_era").update_layout(legend=dict(yanchor="top",
                                                                           y=-0.15,
                                                                           xanchor="right",
                                                                           x=0.5),
                                                               legend_title_text="")
    x_value= clickData["points"][0]["x"]
    y_value = clickData["points"][0]["y"]
    
    if x_value == y_value:
        return dash.no_update
     
    else:
        x_value= clickData["points"][0]["x"]
        y_value = clickData["points"][0]["y"]
        scatter_data = data2[[x_value, y_value, 'rock_era']]
        fig2 = px.scatter(data_frame = scatter_data, 
                         x= x_value,
                         y= y_value,
                         color ='rock_era', opacity = 0.9).update_layout(legend = {'yanchor':'top',
                                                                                  'y':-0.15,
                                                                                  'xanchor':'right',
                                                                                  'x':0.5},
        legend_title_text ="").update_traces(marker = dict(size=6, line={'color':'DarkSlateGrey'}))
        
        
        return fig2  
    
    
    
    
    
    
    
    

# if __name__ == '__main__':
#     app.run(debug = True, port = 8055, jupyter_mode = 'external')







