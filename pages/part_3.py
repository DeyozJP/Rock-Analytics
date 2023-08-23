#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import dash
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from sklearn import preprocessing
import plotly.graph_objects as go

# #### Instantiate the app

# In[2]:

dash.register_page(__name__,  name='Analytics-2')
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
dataframe = pd.read_csv(DATA_PATH.joinpath("rock.csv"))


# #### Transform the variables to make their valies ranging from 0-1

min_max_scaler = preprocessing.MinMaxScaler()

columns = ['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'length']

for  col in columns:
    dataframe[col] = min_max_scaler.fit_transform(dataframe[col].values.reshape(-1, 1))

dataframe = dataframe.drop(['danceability.1'], axis=1)

dataframe1 = dataframe.copy(deep=True)
columns = ['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'length', 'time_signature']
data = dataframe1.copy(deep=True)

data = data[columns]

# app = dash.Dash(__name__, external_stylesheets =[dbc.themes.PULSE],
#                meta_tags = [{'name':'viewport',
#                             'content':'width = device-width, initial-scale = 0.8, maximum-scale = 1.0, minimum-scale=0.5'}])

layout = dbc.Container([
    dbc.Row([
        dbc.Col(children=[
        html.Br(),
        html.H1('Rock Analytics', 
                style={'textAlign': 'center', 'font-size': '21px', "margin": '0px 0px 0px 0px',
                       'font family': 'sans-serif'}),
        html.Br(),
      
        ], xs=12, sm=12, md=11, lg=11, xl=11, xxl=11)
        
    ], justify='center'),
    
    dbc.Row([
        
        dbc.Col(children=[
            html.Hr(style ={"color":'red', 'borderWidth':'2px', 'margin': '0px 0px 0px 0px'}),
              
            html.H3('What makes some rock songs more popular?', style = {'textAlign':'center', 'font-size':'15px', "color":'SeaGreen', "margin":'0px 0px 0px 0px'}),

              ], xs=12, sm=12, md=12, lg=12, xl=12, xxl=12)
    ]),
    
    dbc.Row([
            
        dbc.Col([
            html.Br(),
            html.H5('Filter most popular song', 
                    style ={'textAlign': 'center', 'color': 'darkBlue',
                            'font-size': '18px', 'margin': '0px 0px -5px 0px'},),
            html.P('Scale (0.5 - 1)', style={'textAlign': 'center', 'color': 'darkBlue', 'font-size': '12px'}),
            
            dcc.RangeSlider(id='most_popular_slider',
                            min=0.5,
                            max=1,
                            value=[0.7, 1],
                            step=0.1,
                            vertical=False,
                            marks=None,
                            tooltip={'placement': 'bottom', 'always_visible': True}),
            
            html.Br(),

           
            
        ], xs=12, sm=12, md=6, lg=6, xl=6, xxl=5),
        
        
        dbc.Col([
            html.Br(),
            html.H5('Filter least popular song ', 
                    style={'textAlign': 'center', 'color': 'darkBlue',
                           'font-size': '18px', 'margin': '0px 0px -5px 0px'}),
            html.P('Scale (0 - 0.5)',style={'textAlign': 'center', 'color': 'darkBlue', 'font-size': '12px'}),
            dcc.RangeSlider(id='least_popular_slider',
                       min=0.0,
                       max=0.5,
                    value=[0.0, 0.3],
                      step=0.1,
                      vertical=False,
                      marks=None,
                      tooltip={'placement': 'bottom', 'always_visible': True},),
            html.Br(),

          
        ],xs=12, sm=12, md=6, lg=6, xl=6, xxl=5)
        
        
        
    ], justify='around'),
    
    dbc.Row([
        dbc.Col([

            html.H6('Comparison of attributes of most and least popular rock songs',
                    style={'textAlign': 'center', 'color': 'Blue', 'font-size': '15px', 'margin':'0px 0px -2px 0px'}),
            html.P('(Average)',style={'textAlign': 'center', 'color': 'darkBlue', 'font-size': '12px'}),
            
            dcc.Graph(id='barchart_1', style={'margin': '-18px 0px'}),
            html.Hr()
            
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
         html.H5("Attributes by time signature", 
                style={'textAlign': 'center', 'font-size': '15px', "color": 'SeaGreen',
                       "margin": '-5px 0px 55px 0px'}),
        html.P('(Average)',style={'textAlign': 'center', 'color': 'darkBlue',
                                  'font-size': '12px', "margin": '-60px 0px 40px 0px'}),
            
        dcc.Graph(id = 'time_signature_comparison', style ={'margin': '40px 0px 0px 0px'})
        ], xs=12, sm=12, md=6, lg=6, xl=6, xxl=6),
        
        
        
        dbc.Col([
        html.H5("Attributes over time", 
                style={'textAlign': 'center', 'font-size': '15px', "color": 'SeaGreen',
                         "margin": '-5px 0px -10px 0px'}),
        html.P('(Average)', style={'textAlign': 'center', 'color': 'darkBlue',
                                   'font-size': '12px', "margin": '2px 0px -20px 0px'}),
        html.Br(),
            
        dcc.Checklist(id = 'checklist2', labelStyle={"display": 'inline-block', 'width': "35%",
                                                     'align-items': 'center', 'margin': '-5px 0px -10px 5px'},
                          labelClassName ='mr-3, text-info',
                          value =['popularity', 'acousticness'],
                          options =[col for col in data.columns if col not in ['rock_era', 'pop_cat', 'time_signature',
                                                       'liveness', 'speechiness', 'loudness', 'tempo',
                                                                               'key', 'length', 'valence']], style={"transform": 'scale(0.5)'}),
        dcc.Graph(id="line_chart", style = {'margin': '-10px 0px -20px 0px'})
        ], xs=12, sm=12, md=6, lg=6, xl=6, xxl=6)
        
    ], justify='around'),
], fluid=True)

    

@callback(
    Output(component_id='barchart_1', component_property="figure"),
    Output(component_id='time_signature_comparison', component_property="figure"),
    Input(component_id='most_popular_slider', component_property='value'),
    Input(component_id='least_popular_slider', component_property='value'))


def barchart_1(most_popular_slider, least_popular_slider):
    if most_popular_slider is None and least_popular_slider is None:
        selection = "no selection"
    else:
        global data
        data1 = data.copy(deep=True)

        data1['pop_cat'] = ['pop' if i >= 0.5 else 'unpop' for i in data1['popularity']]
        selection = most_popular_slider
        dataframeA = data1[(data1['popularity'] >= selection[0]) & (data1['popularity'] <= selection[1])]
        selection = least_popular_slider
        dataframeB = data1[(data1['popularity'] >= selection[0]) & (data1['popularity'] <= selection[1])]
        dataframe3 = pd.concat([dataframeA, dataframeB], ignore_index=True)

        dataframe4 = dataframe3.copy(deep=True)
        dataframe4 = dataframe4.drop(['popularity', 'time_signature'], axis=1)

        numerical_columns = dataframe4.select_dtypes(include=[float, int]).columns.tolist()
        data_agg = (dataframe4.groupby('pop_cat')[numerical_columns].mean().T.reset_index().
                    sort_values(by='pop', ascending=False))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=-data_agg['pop'].values,
                y=data_agg['index'],
                orientation='h',
                name='Most Popular Songs',
                textposition='outside',
                text=np.round(data_agg['pop'].values, 2),
                marker={'color': '#000000'}),
         )

        fig.add_trace(go.Bar(x=data_agg['unpop'].values,
                y=data_agg['index'],
                orientation='h',
                name='Least Popular Songs',
                 text=np.round(data_agg['unpop'].values, 2),
                marker={'color': '#05637f'})
         )
        fig.update_layout(barmode='relative', 
 
              yaxis_autorange='reversed',
              bargap=0.00,
              legend_orientation ='h',
              legend_x=-0.02, legend_y=1.2,
              xaxis ={'visible': False}
             ).update_traces(textposition='inside')

# for fig2

        dataframe5 = data1.copy(deep=True)
        dataframe5['time_signature'] = [str(i) for i in dataframe5['time_signature']]

        #         dataframe5['time_signature'].astype('object')
        dataframe5 = dataframe5[(dataframe5['time_signature'] == '4') | (dataframe5['time_signature'] == '3')
                                | (dataframe5['time_signature'] == '5')]
        agg_col = [col for col in dataframe5.columns if col != 'time_signature']
        # agg_col.append('popularity')
        df_ts = round(dataframe5.groupby(['time_signature'])[agg_col].mean(), 2).reset_index()
        df_ts = pd.melt(df_ts, id_vars=('time_signature'))
        fig2 = px.bar(df_ts,
                      x='variable',
                      y='value',
                      color='time_signature',
                      text_auto=True,
                      barmode='stack',
                      color_discrete_map={'3': 'black', '4': '#05637f', '5': '#4e1403'},
                      labels={'variable': ''})
        fig2.update_layout(xaxis={"visible": True}, yaxis={"visible": False},
                           legend=dict(orientation='h', yanchor="top", y=1.2, xanchor="left", x=0),
                           legend_title_text="Time Signature")
        return fig, fig2

# @callback(

    # Output(component_id='time_signature_comparison', component_property="figure"),
    # Input(component_id='most_popular_slider', component_property='value'),
    # Input(component_id='least_popular_slider', component_property='value'))
# def time_line(slider1, slider2):
#     global data
#     data1 = data.copy(deep=True)
#     dataframe5 = data1.copy(deep=True)
#     dataframe5['time_signature'] = [str(i) for i in dataframe5['time_signature']]
#
# #         dataframe5['time_signature'].astype('object')
#     dataframe5 = dataframe5[(dataframe5['time_signature'] == '4') | (dataframe5['time_signature'] == '3')
#                             | (dataframe5['time_signature'] == '5')]
#     agg_col = [col for col in dataframe5.columns if col != 'time_signature']
#     agg_col.append('popularity')
#     df_ts =round(dataframe5.groupby(['time_signature'])[agg_col].mean(), 2).reset_index()
#     df_ts = pd.melt(df_ts, id_vars=('time_signature'))
#     fig2 = px.bar(df_ts,
#             x='variable',
#             y='value',
#             color='time_signature',
#             text_auto=True,
#             barmode='stack',
#             color_discrete_map={'3': 'black', '4': '#05637f', '5': '#4e1403'},
#             labels={'variable': ''})
#     fig2.update_layout(xaxis={"visible": True}, yaxis={"visible": False},
#                           legend=dict(orientation='h', yanchor="top", y=1.2, xanchor="left", x=0),
#                                                                legend_title_text="Time Signature")
#
#     return fig2

@callback(
    Output('line_chart', 'figure'),
    Input('checklist2', 'value')
)

def line_chart(checklist2):
    
    if checklist2:
#         global dataframe1
        data_line = dataframe1.copy(deep=True)
#         column = data_line.columns
        columns = [col for col in checklist2]
        columns.append('release_date')
        data_subset = data_line[columns]
        col_agg = [col for col in columns if col !='release_date' and col != 'time_signature']
        data_agg = data_subset.groupby('release_date')[col_agg].mean()
        line = px.line(data_agg,
                      y=col_agg,
                     labels ={'release_date': 'year', 'value': 'index'})
        line.update_layout(legend=dict(yanchor="top",
                                       y=-0.042,
                                       xanchor="right",x=0.2),
                           legend_title_text="").update_traces(textfont_size=8)
        return line
    

    
    

# if __name__ == '__main__':
#     app.run(debug = True)

