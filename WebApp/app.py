import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d
from bokeh.transform import factor_cmap
import pandas as pd


@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def infty_proj(x):
     return (256 if ~np.isfinite(x) else x)

@st.cache
def GetPds(img):
    pds = pd0 = pd1 = None
    image_val = np.array(img)
    images_gudhi = np.resize(image_val, [128, 128])
    images_gudhi = images_gudhi.reshape(128*128,1)
    #st.image(image_val)
    cub_filtration = gd.CubicalComplex(dimensions = [128,128], top_dimensional_cells=images_gudhi)
    pds = cub_filtration.persistence()
    pd0 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==0])
    pd1 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==1])
    return pds, pd0, pd1

def PlotPersistantDiagram(pd0, pd1):
    tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]

    fig = figure(height=400, tools = tools)
    fig.xaxis.axis_label = 'Birth'
    fig.yaxis.axis_label = 'Death'

    col1 = np.append(np.full(len(pd0[:,0]), 'PH0'), np.full(len(pd1[:,0]), 'PH1'))
    col2 = np.append(pd0[:,0], pd1[:,0])
    col3 = np.append(pd0[:,1], pd1[:,1])
    data = pd.DataFrame(data={'dim': col1, 'x': col2, 'y': col3})

    index_cmap = factor_cmap('dim', palette=['red', 'blue'], factors=sorted(data.dim.unique()))
    fig.scatter("x", "y", source=data,
    legend_group="dim", fill_alpha=0.2, size=4,
    marker="circle",line_color=None,
    fill_color=index_cmap)

    fig.legend.location = "top_left"
    fig.legend.title = "Persistent Diagram"


    maxDim0 = np.max(pd0) if len(pd0) > 0 else 0
    maxDim1 = np.max(pd1) if len(pd1) > 0 else 0
    lineY = max(maxDim0, maxDim1) + 2
    fig.line([-1,lineY], [-1, lineY], line_width=1, color='black', alpha=0.5)
    fig.x_range = Range1d(-1, lineY)
    fig.y_range = Range1d(-1, lineY)
    # chart.set(xlim=(-1, lineY), ylim=(-1, lineY))
    return fig

def main():
    st.title("Featurized Persistent Barcode")
    
    tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
    
    menu = ["Cifar10","Fashion MNIST","Outex68", "Custom"]
    choice = st.sidebar.selectbox("Select a Dataset",menu)

    if choice == "Cifar10":
        image_file = r"data\cifar10.png"
    elif choice == "Fashion MNIST":
        image_file = r"data\FashionMNIST.jpg"
    elif choice == "Outex68":
        image_file = r"data\Outex1.bmp"
    else:
        image_file = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg','bmp'])

    if image_file is not None:
        
        isShowImageChecked = st.checkbox('Input File', value=True)
        img = load_image(image_file)

        if isShowImageChecked:
            st.image(img,width=250)

        pds, pd0, pd1 = GetPds(img)

        isPersBarChecked = st.checkbox('Persistence Barcodes')

        if isPersBarChecked:
            st.subheader("Persistence Barcode")
            # col1, col2 = st.columns(2)
            # fig, ax = plt.subplots()
            # gd.plot_persistence_barcode(pd0, axes=ax)
            # ax.set_title("Persistence Barcode [dim = 0]")
            # col1.pyplot(fig)

            source = ColumnDataSource(data={'x1': pd0[:,0], 'x2': pd0[:,1] - pd0[:,0], 'y': range(len(pd0[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
            fig.hbar(y='y', left='x1', right='x2', height=0.1, alpha=0.5, source=source)
            st.bokeh_chart(fig, use_container_width=True)
            
            source = ColumnDataSource(data={'x1': pd1[:,0], 'x2': pd1[:,1] - pd1[:,0], 'y': range(len(pd1[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
            fig.hbar(y='y', left='x1', right='x2', height=0.1, alpha=0.5, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            # fig, ax = plt.subplots()
            # gd.plot_persistence_barcode(pd1, axes=ax)
            # ax.set_title("Persistence Barcode [dim = 1]")
            # st.pyplot(fig)

        isPersDiagChecked = st.checkbox('Persistence Diagram')

        if isPersDiagChecked:
            st.subheader("Persistence Diagram")
            # fig, ax = plt.subplots()
            # gd.plot_persistence_diagram(pds, axes=ax)
            # ax.set_title("Persistence Diagram")
            # st.pyplot(fig)
            fig = PlotPersistantDiagram(pd0, pd1)
            st.bokeh_chart(fig, use_container_width=True)
        
        isBettiCurveChecked = st.checkbox('Betti Curve')

        if isBettiCurveChecked:
            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.subheader("Betti Curve")

            st.slider("Resolution", 0, 100, value=60, step=1, key='BettiCurveRes')

            Btt_0 = vec.GetBettiCurveFeature(pd0, st.session_state.BettiCurveRes)
            source = ColumnDataSource(data={'x': range(0, len(Btt_0)), 'y': Btt_0})
            fig = figure(title='Betti Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

            Btt_1 = vec.GetBettiCurveFeature(pd1, st.session_state.BettiCurveRes)
            fig = figure(title='Betti Curve [dim = 1]', height=250, tools = tools)
            fig.line(range(0, len(Btt_1)), Btt_1, color='blue', alpha=0.5)
            fig.circle(range(0, len(Btt_1)), Btt_1, fill_color="darkblue", alpha=0.4, size=4, hover_color="red")
            st.bokeh_chart(fig, use_container_width=True)

        isPersStatsChecked = st.checkbox('Persistent Statistics')

        if isPersStatsChecked:
            st.subheader("Persistent Statistics")
            stat_0 = vec.GetPersStats(pd0)
            stat_1 = vec.GetPersStats(pd1)
            df = pd.DataFrame(np.array((stat_0, stat_1)), index=['PH(0)', 'PH(1)'])
            df.columns =['stat. 1', 'stat. 2', 'stat. 3', 'stat. 4', 'stat. 5', 'stat. 6', 'stat. 7', 'stat. 8', 'stat. 9', 'stat. 10', 'stat. 11']
            st.dataframe(df)

        isPersImgChecked = st.checkbox('Persistent Image')

        if isPersImgChecked:
            st.subheader("Persistent Image")
            col1, col2 = st.columns(2)
            res = [100,100]
            PI_0 = vec.GetPersImageFeature(pd0, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_0, res), 0))
            ax.set_title("Persistent Image [dim = 0]")
            col1.pyplot(fig)
        
            PI_1 = vec.GetPersImageFeature(pd1, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_1, res), 0))
            ax.set_title("Persistent Image [dim = 1]")
            col2.pyplot(fig)

        isPersLandChecked = st.checkbox('Persistent Landscape')

        if isPersLandChecked:
            st.subheader("Persistent Landscape")
            col1, col2 = st.columns(2)
            PL_0 = vec.GetPersLandscapeFeature(pd0, num=100)
            fig, ax = plt.subplots()
            ax.plot(PL_0[:100])
            ax.plot(PL_0[100:200])
            ax.plot(PL_0[200:300])
            ax.set_title("Persistent Landscape [dim = 0]")
            col1.pyplot(fig)

            PL_1 = vec.GetPersLandscapeFeature(pd1, num=100)
            fig, ax = plt.subplots()
            ax.plot(PL_1[:100])
            ax.plot(PL_1[100:200])
            ax.plot(PL_1[200:300])
            ax.set_title("Persistent Landscape [dim = 1]")
            col2.pyplot(fig)

        isPersEntropyChecked = st.checkbox('Persistent Entropy')

        if isPersEntropyChecked:
            st.subheader("Persistent Entropy")
            PersEntropy_0 = vec.GetPersEntropyFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent entropy [dim = 0]")
            st.line_chart(PersEntropy_0)

            PersEntropy_1 = vec.GetPersEntropyFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent entropy [dim = 1]")
            st.line_chart(PersEntropy_1)

        isPersSilChecked = st.checkbox('Persistent Silhouette')

        if isPersSilChecked:
            st.subheader("Persistent silhouette")
            PersSil_0 = vec.GetPersSilhouetteFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent silhouette [dim = 0]")
            st.line_chart(PersSil_0)

            PersSil_1 = vec.GetPersSilhouetteFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent silhouette [dim = 1]")
            st.line_chart(PersSil_1)

        isAtolChecked = st.checkbox('Atol')

        if isAtolChecked:
            st.subheader("Atol")
            atol_0 = vec.GetAtolFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Atol [dim = 0]")
            st.line_chart(atol_0)

            atol_1 = vec.GetAtolFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Atol [dim = 1]")
            st.line_chart(atol_1)

        isCarlsCoordsChecked = st.checkbox('Carlsson Coordinates')

        if isCarlsCoordsChecked:
            st.subheader("Carlsson Coordinates")
            carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Carlsson Coordinates [dim = 0]")
            st.line_chart(carlsCoords_0)

            carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Carlsson Coordinates [dim = 1]")
            st.line_chart(carlsCoords_1)

        isPersLifeSpanChecked = st.checkbox('Persistent Life Span')

        if isPersLifeSpanChecked:
            st.subheader("Persistent Life Span")
            persLifeSpan_0 = vec.GetPersLifespanFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Life Span [dim = 0]")
            st.line_chart(persLifeSpan_0)

            persLifeSpan_1 = vec.GetPersLifespanFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Life Span [dim = 1]")
            st.line_chart(persLifeSpan_1)

        isComplexPolynomialChecked = st.checkbox('Complex Polynomial')

        if isComplexPolynomialChecked:
            st.subheader("Complex Polynomial")

            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.selectbox("Polynomial Type",["R", "S", "T"], index=0, key='CPType')

            CP_pd0 = vec.GetComplexPolynomialFeature(pd0, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': range(0, len(CP_pd0)), 'y': CP_pd0})
            fig = figure(title='Complex Polynomial [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CP_pd1 = vec.GetComplexPolynomialFeature(pd1, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': range(0, len(CP_pd1)), 'y': CP_pd1})
            fig = figure(title='Complex Polynomial [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

        isTopologicalVectorChecked = st.checkbox('Topological Vector')

        if isTopologicalVectorChecked:
            st.subheader("Persistent Topological Vector")
            topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Topological Vector [dim = 0]")
            st.line_chart(topologicalVector_0)

            topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Topological Vector [dim = 1]")
            st.line_chart(topologicalVector_1)

        isPersTropCoordsChecked = st.checkbox('Persistent Tropical Coordinates')

        if isPersTropCoordsChecked:
            st.subheader("Persistent Tropical Coordinates")
            persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Tropical Coordinates [dim = 0]")
            st.line_chart(persTropCoords_0)

            persTropCoords_1 = vec.GetPersTropicalCoordinatesFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Tropical Coordinates [dim = 1]")
            st.line_chart(persTropCoords_1)


if __name__ == '__main__':
    main()
