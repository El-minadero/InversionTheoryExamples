'''
Created on Sep 25, 2017

@author: kevinmendoza
'''
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import PIL.Image     as image
"""
        "observation locations"     : data.getXLocations(),
        "observation data"          : data.getXData(),
        "model"                     : m,
        "synthetic data"            : d,
        "model span"                : ((model.x0,model.x1,model.divx),(model.z0,model.z1,model.divz))
"""
def plot(**kwargs):
    
    gs = grd.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[20,1], wspace=0.2,hspace=1)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    
    _plot_series(ax1,kwargs)
    _plot_model_data(ax2,ax3,kwargs)
    
    plt.show()

def _plot_cbar(axis,data):
    img = data["model image"]
    min = int(np.round(np.min(data["model"]),-2))
    max = int(np.round(np.max(data["model"]),-2))
    step = int((max - min)/5)
    cbar = plt.colorbar(img,cax=axis)
    cbar.set_label('Density Anomaly (kg/m^3)')
    axis.yaxis.tick_left()

def _plot_model_data(axis,axis2,data):
    x_span = data["model span"][0]
    z_span = data["model span"][1]
    model_array = _build_model_image(data)
    axis.set_xlim([x_span[0],x_span[1]])
    axis.set_ylim([z_span[1],z_span[0]])
    axis.set_xlabel("Profile distance (m)")
    axis.set_ylabel("Depth (m)",fontsize=8)
   
    im = axis.imshow(model_array,aspect='auto',origin=[0,z_span[0]],extent=(x_span[0],x_span[1],z_span[1],z_span[0]),\
                cmap=mpl.cm.get_cmap("plasma",lut=400))
    axis.xaxis.tick_top()
    data["model image"] = im
    _plot_cbar(axis2,data)
    
    
def _plot_series(axis,data):
    axis.plot(data["observation locations"],data["observation data"],marker="*", linestyle="none", \
              color="red" , label = "Observed Data")
    axis.plot(data["observation locations"],data["synthetic data"], marker="x", linestyle="dashed", \
              color="blue", label = "Predicted Data")
    axis.legend()
    axis.set_xlabel("Profile distance (m)")
    axis.set_ylabel("Gravity anomaly (mGal)",fontsize=8)
    axis.set_title("Predicted vs Observed Gravity Data",\
                   fontsize=10,bbox=dict(facecolor='none', edgecolor='none', pad=20.0))
    
    x_max = np.max(data["observation locations"])
    y_max = np.max(data["observation data"])
    
    axis.set_xlim(0,x_max)
    axis.set_ylim(0,y_max*1.1)
    
def _build_model_image(kwargs):
    columns     = kwargs["model span"][1][2]
    rows        = kwargs["model span"][0][2]
    img         = kwargs["model"]
    img         = img
    img.shape   = (rows,columns)
    return np.flipud(img.T)
    
