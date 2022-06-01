import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from PIL import Image
from skimage import io
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from io import BytesIO
import extract_featurized_barcodes as ex


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
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        pds = pd0 = pd1 = None
        
        if image_file is not None:

            file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
            st.write(file_details)

            img = load_image(image_file)
            st.image(img,width=250)

            image_val = io.imread(image_file)
            images_gudhi = np.resize(image_val, [128, 128])
            images_gudhi = images_gudhi.reshape(128*128,1)
            #st.image(image_val)
            cub_filtration = gd.CubicalComplex(dimensions = [128,128], top_dimensional_cells=images_gudhi)
            pds = cub_filtration.persistence()
            pd0 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==0])
            pd1 = np.array([[x[1][0], infty_proj(x[1][1])]  for x in pds if x[0]==1])

            buf = BytesIO()

            st.subheader("Persistence barcode")
            fig, ax = plt.subplots()
            gd.plot_persistence_barcode(pd0, axes=ax)
            ax.set_title("Persistence barcode [dim = 0]")
            fig.savefig(buf, format="png")
            st.image(buf)

            fig, ax = plt.subplots()
            gd.plot_persistence_barcode(pd1, axes=ax)
            ax.set_title("Persistence barcode [dim = 1]")
            fig.savefig(buf, format="png")
            st.image(buf)

            st.subheader("Persistence diagram")
            fig, ax = plt.subplots()
            gd.plot_persistence_diagram(pds, axes=ax)
            ax.set_title("Persistence diagram")
            fig.savefig(buf, format="png")
            st.image(buf)
            
            st.subheader("BettiCurve")
            res = 100
            Btt_0 = ex.GetBettiCurveFeature(pd0, res)
            ax.set_title("BettiCurve [dim = 0]")
            st.line_chart(Btt_0)

            res = 100
            Btt_1 = ex.GetBettiCurveFeature(pd1, res)
            ax.set_title("BettiCurve [dim = 1]")
            st.line_chart(Btt_0)

            st.subheader("Persistent Statistics")
            stat_0 = ex.GetPersStats(pd0)
            ax.set_title("Persistent Statistics [dim = 0]")
            st.bar_chart(stat_0)

            stat_1 = ex.GetPersStats(pd1)
            ax.set_title("Persistent Statistics [dim = 1]")
            st.bar_chart(stat_1)

            st.subheader("Persistent Image")
            res = [6,6]
            PI_0 = ex.GetPersImageFeature(pd0, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_0, res), 0))
            ax.set_title("Persistent Image [dim = 0]")
            fig.savefig(buf, format="png")
            st.image(buf)
            
            PI_1 = ex.GetPersImageFeature(pd1, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_1, res), 0))
            ax.set_title("Persistent Image [dim = 1]")
            fig.savefig(buf, format="png")
            st.image(buf)

            st.subheader("Persistent Landscape")
            PL_0 = ex.GetPersLandscapeFeature(pd0, num=100)
            fig, ax = plt.subplots()
            ax.plot(PL_0[:100])
            ax.plot(PL_0[100:200])
            ax.plot(PL_0[200:300])
            ax.set_title("Persistent Landscape [dim = 0]")
            fig.savefig(buf, format="png")
            st.image(buf)

            PL_1 = ex.GetPersLandscapeFeature(pd1, num=100)
            fig, ax = plt.subplots()
            ax.plot(PL_1[:100])
            ax.plot(PL_1[100:200])
            ax.plot(PL_1[200:300])
            ax.set_title("Persistent Landscape [dim = 1]")
            fig.savefig(buf, format="png")
            st.image(buf)

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
