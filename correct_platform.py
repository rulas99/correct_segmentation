import plotly.express as px
import dash
from dash import Dash, callback_context
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from skimage import io
from glob import glob
from numba import njit

from warnings import filterwarnings
filterwarnings("ignore")

ref = sorted(glob('classify_sentinel2_trainingdataset/*+dtn.png'))
eva = sorted(glob('classify_sentinel2_trainingdataset/*_dtn.png'))

corr = glob('mask_correction/*_dtnC.png')

evaN = []
refN = []
for i,j in zip(eva,ref):
    imgN = f'mask_correction/{i.split("/")[1]}'
    if not f"{imgN[:imgN.rindex('.')]}C.png" in corr:
        evaN.append(i)
        refN.append(j)

counter = 1
totalLen = len(refN)

ix = 0
imgR = io.imread(refN[ix]) 
imgE = io.imread(evaN[ix])
imgE_c = imgE.copy()

ref_img = px.imshow(imgR, binary_string=True, labels=dict(x=refN[ix].split('/')[1]))
ref_img.update_xaxes(showticklabels=False,side='top')
ref_img.update_yaxes(showticklabels=False)

eva_img = px.imshow(imgE, binary_string=True, labels=dict(x=evaN[ix].split('/')[1]))
eva_img.update_layout(dragmode="drawrect")
eva_img.update_xaxes(side='top',showticklabels=False)
eva_img.update_yaxes(showticklabels=False)

external_stylesheets = ['https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css',
						 dbc.themes.BOOTSTRAP]
app = Dash(__name__,  external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(html.H3("Thermal anomalies correction"), style = {'margin': '15px 2px 2px 15px'}),
        html.Div(html.H6(f"{counter} de {totalLen}",id="counter"), style = {'text-align':'center'}),
        html.Div(
            [dcc.Graph(id="reference", figure=ref_img,style={'width': '750px', 'height': '750px'}),],
            style={"width": "50%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div(
            [dcc.Graph(id="evaluated", figure=eva_img,style={'width': '750px', 'height': '750px'}),],
            style={"width": "50%", "display": "inline-block", "padding": "0 0"},
        ),
    	dbc.Row([
    		dbc.Col(html.Div([dbc.Button('Reset', id='reset', n_clicks=0, className='mr-1',
                                size='md', style={'width': '130px','height': '40px','backgroundColor': '#CF7541',
                                'border-color': '#C60D0D'})],style={'padding': '23px 1px 15px 1px', 'text-align': 'center',
                                       'margin': '2px 2px 2px 2px'}),sm=4),

    		dbc.Col(html.Div([dbc.Button('Guardar y continuar', id='saveycont', n_clicks=0, className='mr-1',
                                size='md', style={'width': '200px','height': '40px','backgroundColor': '#0DC675',
                                'border-color': '#099211'})],style={'padding': '23px 1px 15px 1px', 'text-align': 'center',
                                       'margin': '2px 2px 2px 2px'}),sm=4),

    		dbc.Col(html.Div([dbc.Button('Regresar', id='back', n_clicks=0, className='mr-1',
                                size='md', style={'width': '130px','height': '40px','backgroundColor': '#7BA3AE',
                                'border-color': '#376470'})],style={'padding': '23px 1px 15px 1px', 'text-align': 'center',
                                       'margin': '2px 2px 2px 2px'}),sm=4),
    		], no_gutters=True, justify='center')
    ]
)

@njit
def changePixels(imgE_c,x0,x1,y0,y1):
	for i in range(y0,y1+1):
		for j in range(x0,x1+1):
			imgE_c[i,j] = 0

	return imgE_c

@app.callback(
    Output("evaluated", "figure"),
    Output("reference", "figure"),
    Output("counter", "children"),
    Input("evaluated", "relayoutData"),
    Input("reset", "n_clicks"),
    Input("saveycont", "n_clicks"),
    Input("back", "n_clicks"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data, reset, saveycont, back):
	global imgE_c, ix, eva_img, ref_img, imgE_c, imgE, counter

	changed_id = [p['prop_id'] for p in callback_context.triggered][0]

	if 'reset' in changed_id:
		imgE_c = imgE.copy()
		return eva_img, ref_img, f"{counter} de {len(refN)}"

	elif 'saveycont' in changed_id:
		ix += 1

		rut = evaN[ix].split('/')[1]
		rut = f"mask_correction/{rut[:rut.rindex('.')]}C.png"
		io.imsave(rut,imgE_c)

		imgR = io.imread(refN[ix]) 
		imgE = io.imread(evaN[ix])
		imgE_c = imgE.copy()

		ref_img = px.imshow(imgR, binary_string=True, labels=dict(x=refN[ix].split('/')[1]))
		ref_img.update_xaxes(showticklabels=False,side='top')
		ref_img.update_yaxes(showticklabels=False)

		eva_img = px.imshow(imgE, binary_string=True, labels=dict(x=evaN[ix].split('/')[1]))
		eva_img.update_layout(dragmode="drawrect")
		eva_img.update_xaxes(side='top',showticklabels=False)
		eva_img.update_yaxes(showticklabels=False)



		counter += 1
		return eva_img, ref_img, f"{counter} de {totalLen}"

	elif 'back' in changed_id:
		ix -= 1
		imgR = io.imread(refN[ix]) 
		imgE = io.imread(evaN[ix])
		imgE_c = imgE.copy()

		ref_img = px.imshow(imgR, binary_string=True, labels=dict(x=refN[ix].split('/')[1]))
		ref_img.update_xaxes(showticklabels=False,side='top')
		ref_img.update_yaxes(showticklabels=False)

		eva_img = px.imshow(imgE, binary_string=True, labels=dict(x=evaN[ix].split('/')[1]))
		eva_img.update_layout(dragmode="drawrect")
		eva_img.update_xaxes(side='top',showticklabels=False)
		eva_img.update_yaxes(showticklabels=False)

		counter -= 1

		return eva_img, ref_img, f"{counter} de {totalLen}"

	elif "shapes" in relayout_data:

		last_shape = relayout_data["shapes"][-1]
		x0, y0 = round(last_shape["x0"]), round(last_shape["y0"])
		x1, y1 = round(last_shape["x1"]), round(last_shape["y1"])

		imgE_c = changePixels(imgE_c,x0,x1,y0,y1)

		eva_imgC = px.imshow(imgE_c, binary_string=True, labels=dict(x=evaN[ix].split('/')[1]))
		eva_imgC.update_layout(dragmode="drawrect")
		eva_imgC.update_xaxes(side='top',showticklabels=False)
		eva_imgC.update_yaxes(showticklabels=False)

		return eva_imgC, ref_img, f"{counter} de {totalLen}"

	else:
		return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)