'''
Created on Sep 25, 2017

@author: kevinmendoza
'''
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy         as np
import PIL.Image     as image
"""
        "data"            : (d,d0,x)
        "data metadata"   : (y_units,_x_units,graph title)
        "model"           : (m,x)
        "model metadata"  : (y_units,model_units, graph title)
"""
def plot(**kwargs):
    
    gs = grd.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[20,1], wspace=0.2,hspace=1)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    
    _plot_series(ax1,**kwargs)
    _plot_model_data(ax2,ax3,**kwargs)
    
    plt.show()

def _plot_cbar(axis,**data):
    img = data["model image"]
    min = int(np.round(np.min(data["model"][0]),-2))
    max = int(np.round(np.max(data["model"][0]),-2))
    step = int((max - min)/5)
    cbar = plt.colorbar(img,cax=axis)
    cbar.set_label(data["model metadata"][1])
    axis.yaxis.tick_left()

def _plot_model_data(axis,axis2,**data):
    model_array = _build_model_image(**data)
    z_data = data["model"][1]
    zmin = np.min(z_data)
    zmax = np.max(z_data)
    axis.set_ylim(zmax,zmin)
    axis.set_ylabel(data["model metadata"][0],fontsize=8)
   
    im = axis.imshow(model_array,aspect='auto',origin=[0,zmax],extent=(0,1,zmin,zmax),\
                cmap=mpl.cm.get_cmap("plasma", lut=400))
    axis.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    data["model image"] = im
    _plot_cbar(axis2,**data)
    
    
def _plot_series(axis,**data):
    data_space = data["data"]
    data_units = data['data metadata']
    axis.plot(data_space[2],data_space[0],marker="*", linestyle="none", \
              color="red" , label = "Observed Data")
    axis.plot(data_space[2],data_space[1], marker="x", linestyle="dashed", \
              color="blue", label = "Predicted Data")
    axis.legend()
    axis.set_xlabel(data_units[1])
    axis.set_ylabel(data_units[0],fontsize=8)
    axis.set_title( data_units[2],\
                   fontsize=10,bbox=dict(facecolor='none', edgecolor='none', pad=20.0))
    
    x_max = np.max(data_space[2])
    y_max = np.max(data_space[1])
    
    axis.set_xlim(0,x_max)
    axis.set_ylim(0,y_max*1.1)
    
def _build_model_image(**kwargs):
    rows        = len(kwargs["model"][0])
    img         = kwargs["model"][0]
    img.shape   = (rows,1)
    return img
    
