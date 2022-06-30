import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from io import BytesIO
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def infty_proj(x):
     return (256 if ~np.isfinite(x) else x)

def main():
    st.title("Featurized Persistent Barcode Visualization")

    menu = ["Custom","Cifar10","Fashion MNIST","Outex68"]
    choice = st.sidebar.selectbox("Select a Dataset",menu)

    if choice == "Custom":
        st.text("Select visualization options:")

        col1, col2, col3 = st.columns(3)
        isShowImageChecked = col1.checkbox('Input File', value=True)
        isPersBarChecked = col2.checkbox('Persistence Barcodes')
        isPersDiagChecked = col3.checkbox('Persistence Diagram')

        col1, col2, col3 = st.columns(3)
        isBettiCurveChecked = col1.checkbox('Betti Curve')
        isPersStatsChecked = col2.checkbox('Persistent Statistics')
        isPersImgChecked = col3.checkbox('Persistent Image')

        col1, col2, col3 = st.columns(3)
        isPersLandChecked = col1.checkbox('Persistent Landscape')
        isPersEntropyChecked = col2.checkbox('Persistent Entropy')
        isPersSilChecked = col3.checkbox('Persistent Silhouette')

        col1, col2, col3 = st.columns(3)
        isAtolChecked = col1.checkbox('Atol')
        isCarlsCoordsChecked = col2.checkbox('Carlsson Coordinates')
        isPersLifeSpanChecked = col3.checkbox('Persistent Life Span')

        col1, col2, col3 = st.columns(3)
        isComplexPolynomialTypeRChecked = col1.checkbox('Complex Polynomial [R]')
        isComplexPolynomialTypeSChecked = col2.checkbox('Complex Polynomial [S]')
        isComplexPolynomialTypeTChecked = col3.checkbox('Complex Polynomial [T]')

        col1, col2, col3 = st.columns(3)
        isTopologicalVectorChecked = col1.checkbox('Topological Vector')
        isPersTropCoordsChecked = col2.checkbox('Persistent Tropical Coordinates')

        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        pds = pd0 = pd1 = None
        
        if image_file is not None:
            
            col1, col2 = st.columns(2)
            file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            col1.write(file_details)

            img = load_image(image_file)

            if isShowImageChecked:
                col2.image(img,width=250)

            #image_val = io.imread(img)
            image_val = np.array(img)
            images_gudhi = np.resize(image_val, [128, 128])
            images_gudhi = images_gudhi.reshape(128*128,1)
            #st.image(image_val)
            cub_filtration = gd.CubicalComplex(dimensions = [128,128], top_dimensional_cells=images_gudhi)
            pds = cub_filtration.persistence()
            pd0 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==0])
            pd1 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==1])

            buf = BytesIO()

            if isPersBarChecked:
                st.subheader("Persistence Barcode")
                col1, col2 = st.columns(2)
                fig, ax = plt.subplots()
                gd.plot_persistence_barcode(pd0, axes=ax)
                ax.set_title("Persistence Barcode [dim = 0]")
                col1.pyplot(fig)

                fig, ax = plt.subplots()
                gd.plot_persistence_barcode(pd1, axes=ax)
                ax.set_title("Persistence Barcode [dim = 1]")
                col2.pyplot(fig)

            if isPersDiagChecked:
                st.subheader("Persistence Diagram")
                fig, ax = plt.subplots()
                gd.plot_persistence_diagram(pds, axes=ax)
                ax.set_title("Persistence Diagram")
                st.pyplot(fig)
            
            if isBettiCurveChecked:
                tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
                st.subheader("Betti Curve")
                col1, col2 = st.columns(2)
                res = col1.slider("Resolution", min_value=0, max_value=100, value=60, step=1)
                Btt_0 = vec.GetBettiCurveFeature(pd0, res)
                source = ColumnDataSource(data={'x': range(0, len(Btt_0)), 'y': Btt_0})
                fig = figure(title='Betti Curve [dim = 0]', height=250, tools = tools)
                fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
                fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
                # slider = Slider(start=0, end=100, value=res, step=1, title="Resolution")
                # def updateBettiCurve(attr, old, new):
                #     res = slider.value
                #     Btt_0 = vec.GetBettiCurveFeature(pd0, res)
                #     source.data= {'x': range(0, len(Btt_0)), 'y': Btt_0}

                # slider.on_change('value', updateBettiCurve)
                st.bokeh_chart(fig, use_container_width=True)

                Btt_1 = vec.GetBettiCurveFeature(pd1, res)
                fig = figure(title='Betti Curve [dim = 1]', height=250, tools = tools)
                fig.line(range(0, len(Btt_1)), Btt_1, color='blue', alpha=0.5)
                fig.circle(range(0, len(Btt_1)), Btt_1, fill_color="darkblue", alpha=0.4, size=4, hover_color="red")
                st.bokeh_chart(fig, use_container_width=True)

            if isPersStatsChecked:
                st.subheader("Persistent Statistics")
                col1, col2 = st.columns(2)
                stat_0 = vec.GetPersStats(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Statistics [dim = 0]")
                col1.bar_chart(stat_0)

                stat_1 = vec.GetPersStats(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Statistics [dim = 1]")
                col2.bar_chart(stat_1)

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

            if isPersEntropyChecked:
                st.subheader("Persistent Entropy")
                col1, col2 = st.columns(2)
                PersEntropy_0 = vec.GetPersEntropyFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent entropy [dim = 0]")
                col1.line_chart(PersEntropy_0)

                PersEntropy_1 = vec.GetPersEntropyFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent entropy [dim = 1]")
                col2.line_chart(PersEntropy_1)
                
            if isPersSilChecked:
                st.subheader("Persistent silhouette")
                col1, col2 = st.columns(2)
                PersSil_0 = vec.GetPersSilhouetteFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent silhouette [dim = 0]")
                col1.line_chart(PersSil_0)

                PersSil_1 = vec.GetPersSilhouetteFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent silhouette [dim = 1]")
                col2.line_chart(PersSil_1)

            if isAtolChecked:
                st.subheader("Atol")
                col1, col2 = st.columns(2)
                atol_0 = vec.GetAtolFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Atol [dim = 0]")
                col1.line_chart(atol_0)

                atol_1 = vec.GetAtolFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Atol [dim = 1]")
                col2.line_chart(atol_1)

            if isCarlsCoordsChecked:
                st.subheader("Carlsson Coordinates")
                col1, col2 = st.columns(2)
                carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Carlsson Coordinates [dim = 0]")
                col1.line_chart(carlsCoords_0)

                carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Carlsson Coordinates [dim = 1]")
                col2.line_chart(carlsCoords_1)

            if isPersLifeSpanChecked:
                st.subheader("Persistent Life Span")
                col1, col2 = st.columns(2)
                persLifeSpan_0 = vec.GetPersLifespanFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Life Span [dim = 0]")
                col1.line_chart(persLifeSpan_0)

                persLifeSpan_1 = vec.GetPersLifespanFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Life Span [dim = 1]")
                col2.line_chart(persLifeSpan_1)

            if isComplexPolynomialTypeRChecked:
                st.subheader("Complex Polynomial [R]")
                col1, col2 = st.columns(2)
                complexPolynomialTypeR_0 = vec.GetComplexPolynomialFeature(pd0, pol_type='R')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 0]")
                col1.line_chart(complexPolynomialTypeR_0)

                complexPolynomialTypeR_1 = vec.GetComplexPolynomialFeature(pd1, pol_type='R')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 1]")
                col2.line_chart(complexPolynomialTypeR_1)

            if isComplexPolynomialTypeSChecked:
                st.subheader("Complex Polynomial [S]")
                col1, col2 = st.columns(2)
                complexPolynomialTypeS_0 = vec.GetComplexPolynomialFeature(pd0, pol_type='S')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 0]")
                col1.line_chart(complexPolynomialTypeS_0)

                complexPolynomialTypeS_1 = vec.GetComplexPolynomialFeature(pd1, pol_type='S')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 1]")
                col2.line_chart(complexPolynomialTypeS_1)

            if isComplexPolynomialTypeTChecked:
                st.subheader("Complex Polynomial [T]")
                col1, col2 = st.columns(2)
                complexPolynomialTypeT_0 = vec.GetComplexPolynomialFeature(pd0, pol_type='T')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 0]")
                col1.line_chart(complexPolynomialTypeT_0)

                complexPolynomialTypeT_1 = vec.GetComplexPolynomialFeature(pd1, pol_type='T')
                fig, ax = plt.subplots()
                ax.set_title("Complex Polynomial [dim = 1]")
                col2.line_chart(complexPolynomialTypeT_1)

            if isTopologicalVectorChecked:
                st.subheader("Persistent Topological Vector")
                col1, col2 = st.columns(2)
                topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Topological Vector [dim = 0]")
                col1.line_chart(topologicalVector_0)

                topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Topological Vector [dim = 1]")
                col2.line_chart(topologicalVector_1)

            if isPersTropCoordsChecked:
                st.subheader("Persistent Tropical Coordinates")
                col1, col2 = st.columns(2)
                persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Tropical Coordinates [dim = 0]")
                col1.line_chart(persTropCoords_0)

                persTropCoords_1 = vec.GetPersTropicalCoordinatesFeature(pd1)
                fig, ax = plt.subplots()
                ax.set_title("Persistent Tropical Coordinates [dim = 1]")
                col2.line_chart(persTropCoords_1)

    elif choice == "Cifar10":
        st.subheader("Example from Cifar10 database")
        st.write("Choose an example")

    elif choice == "Fashion MNIST":
        st.subheader("Example from Fashion MNIST database")
        st.write("Choose an example")

    elif choice == "Outex68":
        st.subheader("Example from Outex68 database")
        st.write("Choose an example")


if __name__ == '__main__':
    main()
