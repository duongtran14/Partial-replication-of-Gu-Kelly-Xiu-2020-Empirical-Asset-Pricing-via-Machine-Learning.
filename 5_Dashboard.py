import dash
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=2.0, minimum-scale=1.0',
                            }]
)
server = app.server

df = pd.read_csv('monthly_performance.csv')
dd_range = [max(df[['Return drawdown', 'Alpha drawdown']].max()), 0]
to_range = [df['Turnover'].min().item(), df['Turnover'].max().item()]

overall_df = pd.read_csv('overall_performance.csv')
betas_df = pd.read_csv('market_betas.csv')
factors = betas_df['Factors'].unique().tolist()
factors.insert(0,'Factors')

font = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif'

popovers = dmc.Group([
    dmc.HoverCard(
        [
            dmc.HoverCardTarget(dmc.Button('Methodology',
                                         variant='outline',color='#1098ad')),
            dmc.HoverCardDropdown([
                dmc.Anchor('3-hidden-layer neural networks following Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning", Review of Financial Studies, Vol. 33, Issue 5, (2020), 2223-2273.',
                           href='https://doi.org/10.1093/rfs/hhaa009', target="_blank")
            ]),
        ],
        width=200
    ),
    dmc.HoverCard(
        [
            dmc.HoverCardTarget(dmc.Button('Monthly Stocks Returns Data',
                                         variant='outline',color='#1098ad')),
            dmc.HoverCardDropdown(dmc.Text('Downloaded from WRDS')),
        ],
    ),
    dmc.HoverCard(
        [
            dmc.HoverCardTarget(dmc.Button('Monthly Stocks Characteristics Data',
                                         variant='outline',color='#1098ad')),
            dmc.HoverCardDropdown([
                dmc.Anchor('Dacheng Xiu\'s webpage',
                           href='https://dachxiu.chicagobooth.edu/', target="_blank")
            ]),
        ],
    ),
    dmc.HoverCard(
        [
            dmc.HoverCardTarget(dmc.Button('Monthly Macro-economic Predictors Data',
                                         variant='outline',color='#1098ad')),
            dmc.HoverCardDropdown([
                dmc.Anchor('Amit Goyal\'s webpage, under the paper Goyal, Amit, Ivo Welch, and Athanasse Zafirov, November 2024, A Comprehensive 2022 Look at the Empirical Performance of Equity Premium Prediction, Review of Financial Studies 37(11), 3490–3557.',
                           href='https://sites.google.com/view/agoyal145', target="_blank")
            ]),
        ],
        width=300
    ),
    dmc.HoverCard(
        [
            dmc.HoverCardTarget(dmc.Button('Scripts',
                                         variant='outline',color='#1098ad')),
            dmc.HoverCardDropdown([
                dmc.Anchor('Full scripts for model training, validation, and out-of-sample testing',
                           href='https://github.com/duongtran14/Partial-replication-of-Gu-Kelly-Xiu-2020-Empirical-Asset-Pricing-via-Machine-Learning.?tab=readme-ov-file', target="_blank")
            ]),
        ],
        width=200
    ),
    ],
    style={'marginLeft':49,'marginTop':9,'marginBottom':9},
)

HEADER = dbc.Navbar(
    children=[
        html.A([
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand('Trading Performance using Neural Networks',
                                        className='ml-2',style={'font-size':29})
                    ),
                ],
                align='center',
                style={'marginLeft': 49},
                className='g-0',
            ),
            dbc.Row(
                [
                    'Interactive dashboard displaying backtest results of different strategies in parallel for easier comparison'
                ],
                align='center',
                style={'marginLeft': 49, 'font-size': 18, 'color': 'white'},
                className='g-0',
            ),
        ]),
    ],
    color='#3b3b3b',
    dark=True,
    sticky='top',
)

LEFT_SELECTOR = dbc.Container(
    [
        html.H4(children='Select criteria', className='lead', style={'font-weight': '619', 'font-size':18}),
        html.Hr(className='my-2'),
        html.Label('Return', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='ret-radio-left',
            options=df['Return_type'].unique(),
            value=df['Return_type'].unique()[0],
            inline=True,
            style={'font-size':13},
        ),
        html.Label('Strategy', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='strat-radio-left',
            options=df['Strategy_type'].unique(),
            value=df['Strategy_type'].unique()[0],
            inline=True,
            style={'font-size':13},
        ),
        html.Label('Portfolio', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='portf-radio-left',
            options=df['Portfolio_type'].unique(),
            value=df['Portfolio_type'].unique()[0],
            inline=True,
            style={'font-size':13},
        ),
    ],
    fluid=True,
)

LEFT_OVR_TABLE = html.Div(
    id='ovr-tbl-left-block',
    children=[
            dash_table.DataTable(
                id='ovr-tbl-left',
                style_as_list_view=True,
                style_cell_conditional=[
                    {'if': {'column_id': 'Statistics'},
                    'textAlign': 'left','minWidth': '99px'},
                ],
                style_data_conditional=[
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': '#f1f3f5',
                    },
                    {
                        'if': {'filter_query': '{Annualized} is nil', 'column_id': 'Annualized'},
                        'backgroundColor':'#868e96'
                    },
                ],
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'max-width': '0',
                },
                style_header={
                    'backgroundColor': '#424242',
                    'fontWeight': '619', 'font-family': font, 'fontSize': '15px',
                    'color': 'white',
                },
                style_data={
                    'whiteSpace': 'normal', 'height': 'auto', 'font-family': font, 'fontSize': '13px',
                    'width': '51px',
                    'maxWidth': '51px',
                    'minWidth': '51px',
                },
                style_table={
                    'overflowX': 'auto'
                },
                columns=[{'id': c, 'name': c} for c in overall_df.columns[2:]],
                data=[],
            )
    ]
)

LEFT_BETA_TABLE = html.Div(
    children=[
        dash_table.DataTable(
            id='beta-tbl-left',
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Factors'},
                    'textAlign': 'left',
                }
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f1f3f5',
                }
            ],
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
                'max-width': '0',
                'textAlign': 'center',
            },
            style_header={
                'backgroundColor': '#424242',
                'fontWeight': '619', 'font-family': font, 'fontSize': '14px',
                'color': 'white',
            },
            style_data={'whiteSpace': 'normal', 'height': 'auto', 'font-family': font, 'fontSize': '13px'},
            columns=[{'id': c, 'name': c} for c in factors],
            data=[],
        )
    ],
)

LEFT_PLOTS = [
    dcc.Graph(id='ret-graph-left',),
    dcc.Graph(id='dd-graph-left',),
    dcc.Graph(id='to-graph-left',),
]

LEFT_UPPER_BODY = dbc.Col(
    [
        dbc.Row(
        [
            dbc.Col(LEFT_SELECTOR,width=4),
            dbc.Col(dbc.Card(LEFT_OVR_TABLE),width=8),
        ],
        style={'marginTop': 19, 'marginBottom': 19},
        ),
        dbc.Row([
            dbc.Col(LEFT_BETA_TABLE,width=12,style={'marginBottom':9}),
            ])
    ],
    style={'marginLeft':9}
)

LEFT_BODY = dbc.Col(
    [
        dbc.Card([
            dbc.CardHeader(LEFT_UPPER_BODY),
            dbc.CardBody(
                [
                    dbc.Col(LEFT_PLOTS,style={'marginTop':1}),
                ]
            )
        ],
        color='dark',outline=True,
        )
    ],width=12
)

RIGHT_SELECTOR = dbc.Container(
    [
        html.H4(children='Select criteria', className='lead', style={'font-weight': '619', 'font-size':18}),
        html.Hr(className='my-2'),
        html.Label('Return', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='ret-radio-right',
            options=df['Return_type'].unique(),
            value=df['Return_type'].unique()[0],
            inline=True,
            style={'font-size':13},
        ),
        html.Label('Strategy', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='strat-radio-right',
            options=df['Strategy_type'].unique(),
            value=df['Strategy_type'].unique()[0],
            inline=True,
            style={'font-size':13},
        ),
        html.Label('Portfolio', className='lead',style={'font-size':15}),
        dcc.RadioItems(
            id='portf-radio-right',
            options=df['Portfolio_type'].unique(),
            value=df['Portfolio_type'].unique()[1],
            inline=True,
            style={'font-size':13},
        ),
    ],
    fluid=True,
)

RIGHT_OVR_TABLE = html.Div(
    id='ovr-tbl-right-block',
    children=[
            dash_table.DataTable(
                id='ovr-tbl-right',
                style_as_list_view=True,
                style_cell_conditional=[
                    {'if': {'column_id': 'Statistics'},
                    'textAlign': 'left','minWidth': '99px'},
                ],
                style_data_conditional=[
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': '#f1f3f5',
                    },
                    {
                        'if': {'filter_query': '{Annualized} is nil', 'column_id': 'Annualized'},
                        'backgroundColor': '#868e96'
                    },
                ],
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'max-width': '0',
                },
                style_header={
                    'backgroundColor': '#424242',
                    'fontWeight': '619', 'font-family': font, 'fontSize': '15px',
                    'color': 'white',
                },
                style_data={
                    'whiteSpace': 'normal', 'height': 'auto', 'font-family': font, 'fontSize': '13px',
                    'width': '51px',
                    'maxWidth': '51px',
                    'minWidth': '51px',
                },
                style_table={
                    'overflowX': 'auto'
                },
                columns=[{'id': c, 'name': c} for c in overall_df.columns[2:]],
                data=[],
            )
    ]
)

RIGHT_BETA_TABLE = html.Div(
    children=[
        dash_table.DataTable(
            id='beta-tbl-right',
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Factors'},
                    'textAlign': 'left',
                }
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f1f3f5',
                }
            ],
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
                'max-width': '0',
                'textAlign': 'center',
            },
            style_header={
                'backgroundColor': '#424242',
                'fontWeight': '619', 'font-family': font, 'fontSize': '14px',
                'color': 'white',
            },
            style_data={'whiteSpace': 'normal', 'height': 'auto', 'font-family': font, 'fontSize': '13px'},
            columns=[{'id': c, 'name': c} for c in factors],
            data=[],
        )
    ],
)

RIGHT_PLOTS = [
    dcc.Graph(id='ret-graph-right',),
    dcc.Graph(id='dd-graph-right',),
    dcc.Graph(id='to-graph-right',),
]

RIGHT_UPPER_BODY = dbc.Col(
    [
        dbc.Row(
        [
            dbc.Col(RIGHT_SELECTOR,width=4),
            dbc.Col(dbc.Card(RIGHT_OVR_TABLE),width=8),
        ],
        style={'marginTop': 19, 'marginBottom': 19,},
        ),
        dbc.Row([
            dbc.Col(RIGHT_BETA_TABLE,width=12,style={'marginLeft':2,'marginBottom':9}),
            ])
    ],
    style={'marginLeft':9}
)

RIGHT_BODY = dbc.Col(
    [
        dbc.Card([
            dbc.CardHeader(RIGHT_UPPER_BODY),
            dbc.CardBody(
                [
                    dbc.Col(RIGHT_PLOTS,style={'marginTop':1}),
                ]
            )
        ],
        color='dark',outline=True,
        )
    ],width=12
)


BODY = dbc.Stack([
    dbc.Col(LEFT_BODY,width={'xs':6,'sm':6,'md':6,'lg':6,'order':1},
            style={'marginLeft':9,'marginRight':3},),
    dbc.Col(RIGHT_BODY,width={'xs':6,'sm':6,'md':6,'lg':6,'order':'last'},
            style={'marginLeft':9,'marginRight':3},),
    ],
    direction='horizontal',
)

app.layout = dmc.MantineProvider([
    dmc.Stack(
    [
        HEADER,
        popovers,
        BODY
    ],
    gap=3,
    )
])

@app.callback(
    Output('ret-graph-left', 'figure'),
    Output('dd-graph-left', 'figure'),
    Output('to-graph-left', 'figure'),
    Input('ret-radio-left', 'value'),
    Input('strat-radio-left', 'value'),
    Input('portf-radio-left', 'value'),
)
def display_graph_left(ret_type, strat_type, portf_type):
    df_filtered = df[
        (df['Return_type'] == ret_type) & (df['Strategy_type'] == strat_type) & (df['Portfolio_type'] == portf_type)]
    ret_fig = px.line(df_filtered, x='Date', y=['Return', 'Alpha', 'SP500'], title='Cummulative Return',
                      template='none',
                      height=319,
                      color_discrete_sequence=['#0c8599','#ff8787','#96f2d7'])
    dd_fig = px.line(df_filtered, x='Date', y=['Return drawdown', 'Alpha drawdown'], title='Drawdown',
                     line_shape='spline', template='none',
                     height=239,
                     color_discrete_sequence=['#0c8599','#ff8787','#96f2d7'])
    to_fig = px.line(df_filtered, x='Date', y=['Turnover'], title='Turnover',
                     template='none',
                     height=239,
                     color_discrete_sequence=['#1098ad'])

    ret_fig.update_xaxes(dtick='M12')
    dd_fig.update_xaxes(dtick='M12', visible=False)
    to_fig.update_xaxes(dtick='M12')

    dd_fig.update_yaxes(range=dd_range, fixedrange=True)
    to_fig.update_yaxes(range=to_range, fixedrange=True)

    ret_fig.update_layout(xaxis_title=None, yaxis_title=None,
                          title={'y':0.99},
                          font_family=font,
                          legend=dict(orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01),
                          legend_title_text=None,
                          margin=dict(t=19,b=59,r=3,l=21),
                          )
    dd_fig.update_layout(xaxis_title=None, yaxis_title=None,
                         font_family=font,
                         legend=dict(orientation='h', yanchor='top', y=1.19, xanchor='right', x=1.0),
                         legend_title_text=None,
                         margin=dict(t=19,b=19,r=3,l=21)
                         )
    to_fig.update_layout(xaxis_title=None, yaxis_title=None,
                         font_family=font,
                         legend=dict(orientation='h', yanchor='top', y=1.09, xanchor='left', x=0.01),
                         legend_title_text=None,
                         margin=dict(t=27,b=39,r=3,l=21)
                         )

    dd_fig.update_traces(fill='tozeroy')

    return ret_fig, dd_fig, to_fig

@app.callback(
    Output('ovr-tbl-left', 'data'),
    Output('beta-tbl-left', 'data'),
    Input('strat-radio-left', 'value'),
    Input('portf-radio-left', 'value'),
)
def display_table_left(strat_type, portf_type):
    overall_df_filtered = overall_df[(overall_df['Strategy_type'] == strat_type) & (overall_df['Portfolio_type'] == portf_type)]
    overall_df_filtered = overall_df_filtered.round(3)
    betas_df_filtered = betas_df[(betas_df['Strategy_type'] == strat_type) & (betas_df['Portfolio_type'] == portf_type)]
    betas_df_filtered = betas_df_filtered.iloc[:,2:].set_index('Factors').T
    betas_df_filtered = betas_df_filtered.rename_axis('Factors').reset_index().rename_axis(None, axis=1)
    betas_df_filtered = betas_df_filtered.round(3)
    return overall_df_filtered.to_dict('records'), betas_df_filtered.to_dict('records')

@app.callback(
    Output('ret-graph-right', 'figure'),
    Output('dd-graph-right', 'figure'),
    Output('to-graph-right', 'figure'),
    Input('ret-radio-right', 'value'),
    Input('strat-radio-right', 'value'),
    Input('portf-radio-right', 'value'),
)
def display_graph_right(ret_type, strat_type, portf_type):
    df_filtered = df[
        (df['Return_type'] == ret_type) & (df['Strategy_type'] == strat_type) & (df['Portfolio_type'] == portf_type)]
    ret_fig = px.line(df_filtered, x='Date', y=['Return', 'Alpha', 'SP500'], title='Cummulative Return',
                      template='none',
                      height=319,
                      color_discrete_sequence=['#0c8599','#ff8787','#96f2d7'])
    dd_fig = px.line(df_filtered, x='Date', y=['Return drawdown', 'Alpha drawdown'], title='Drawdown',
                     line_shape='spline', template='none',
                     height=239,
                     color_discrete_sequence=['#0c8599','#ff8787','#96f2d7'])
    to_fig = px.line(df_filtered, x='Date', y=['Turnover'], title='Turnover',
                     template='none',
                     height=239,
                     color_discrete_sequence=['#1098ad'])

    ret_fig.update_xaxes(dtick='M12')
    dd_fig.update_xaxes(dtick='M12', visible=False)
    to_fig.update_xaxes(dtick='M12')

    dd_fig.update_yaxes(range=dd_range, fixedrange=True)
    to_fig.update_yaxes(range=to_range, fixedrange=True)

    ret_fig.update_layout(xaxis_title=None, yaxis_title=None,
                          title={'y':0.99},
                          font_family=font,
                          legend=dict(orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01),
                          legend_title_text=None,
                          margin=dict(t=19,b=59,r=3,l=21),
                          )
    dd_fig.update_layout(xaxis_title=None, yaxis_title=None,
                         font_family=font,
                         legend=dict(orientation='h', yanchor='top', y=1.19, xanchor='right', x=1.0),
                         legend_title_text=None,
                         margin=dict(t=19,b=19,r=3,l=21)
                         )
    to_fig.update_layout(xaxis_title=None, yaxis_title=None,
                         font_family=font,
                         legend=dict(orientation='h', yanchor='top', y=1.09, xanchor='left', x=0.01),
                         legend_title_text=None,
                         margin=dict(t=27,b=39,r=3,l=21)
                         )

    dd_fig.update_traces(fill='tozeroy')

    return ret_fig, dd_fig, to_fig


@app.callback(
    Output('ovr-tbl-right', 'data'),
    Output('beta-tbl-right', 'data'),
    Input('strat-radio-right', 'value'),
    Input('portf-radio-right', 'value'),
)
def display_table_right(strat_type, portf_type):
    overall_df_filtered = overall_df[(overall_df['Strategy_type'] == strat_type) & (overall_df['Portfolio_type'] == portf_type)]
    overall_df_filtered = overall_df_filtered.round(3)
    betas_df_filtered = betas_df[(betas_df['Strategy_type'] == strat_type) & (betas_df['Portfolio_type'] == portf_type)]
    betas_df_filtered = betas_df_filtered.iloc[:,2:].set_index('Factors').T
    betas_df_filtered = betas_df_filtered.rename_axis('Factors').reset_index().rename_axis(None, axis=1)
    betas_df_filtered = betas_df_filtered.round(3)
    return overall_df_filtered.to_dict('records'), betas_df_filtered.to_dict('records')

app.run(debug=True)
