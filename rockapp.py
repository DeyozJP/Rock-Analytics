import dash
from dash import html, dcc
import dash_bootstrap_components as dbc



app = dash.Dash(__name__,suppress_callback_exceptions = True, use_pages = True,
                external_stylesheets=[dbc.themes.SPACELAB])
sidebar = dbc.Nav([
    dbc.NavLink([
        html.Div(page['name'], className='ms-2', style = {"margin": "-10px 0px 0px -20px"}),
    ],
        href = page['path'],
        active ='exact',
    )
    for page in dash.page_registry.values()
],
    vertical = False, pills = True, className='Success',style={'transform': 'scale(0.7)'},
)



app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Img(src='assets/Logo.jpg',
                     style={'height': '45px', 'width': '80%', 'margin': '10px 0px 0px 0px'},)

        ],xs=2, sm=2, md=2, lg=1, xl=1, xxl=1),
        dbc.Col([html.Div("Rock Analytics",

                         style={'font-size': 30, 'textAlign': 'center', 'color': 'purple',
                                'margin': '10px 0px 0px -60px'})],
                xs= 9, sm = 9, md = 9, lg=9, xl=9, xxl=9)
    ],justify='around'),

    html.Hr(),
    dbc.Row([

        dbc.Col([
            sidebar
        ], xs=12, sm=12, md=12, lg=12, xl=12, xxl=12),
    ]),

    dbc.Row([
        dbc.Col([
            dash.page_container
        ],xs=12, sm=12, md=11, lg=11, xl=11, xxl=11)

    ])


])





if __name__=='__main__':
    app.run(debug=True)
