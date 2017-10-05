'''
Created on Sep 25, 2017

@author: kevinmendoza
'''
import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import PIL.Image     as image

def plot(model,data):
    
    data_dict                       = _prepare_data(model,data)
    data_dict["Image"]              = _build_model_image(model)
    
    gs = grd.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[20,1], wspace=0.2,hspace=1)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    
    _plot_time_series(ax1,data_dict)
    _plot_model_data(ax2,ax3,data_dict)
    
    plt.show()

def _plot_model_data(axis,axis2,data):
    x_span = data["model x span"]
    z_span = data["model z span"]
    
    axis.set_xlim([x_span[0],x_span[1]])
    axis.set_ylim([z_span[1],z_span[0]])
    axis.set_xlabel("Profile distance (m)")
    axis.set_ylabel("Depth (m)",fontsize=8)
   
    im = axis.imshow(data["Image"],aspect='auto',origin=[0,z_span[0]],\
                cmap=mpl.cm.get_cmap("plasma",lut=400),extent=(x_span[0],x_span[1],-225,-25))
    axis.xaxis.tick_top()
    cbar = plt.colorbar(im,cax=axis2)
    cbar.set_label('Density Anomaly (kg/m^3)')
    axis2.yaxis.tick_left()
    
    
def _plot_time_series(axis,data):
    axis.plot(data["Location data"],data["Observations"],marker="*", linestyle="none", \
              color="red" , label = "Observed Data")
    axis.plot(data["Location data"],data["Modeled data"], marker="x", linestyle="dashed", \
              color="blue", label = "Predicted Data")
    axis.legend()
    axis.set_xlabel("Profile distance (m)")
    axis.set_ylabel("Gravity anomaly (mGal)",fontsize=8)
    axis.set_title("Predicted vs Observed Gravity Data",\
                   fontsize=10,bbox=dict(facecolor='none', edgecolor='none', pad=20.0))
    
    x_max = np.max(data["Location data"])
    y_max = np.max(data["Observations"])
    
    axis.set_xlim(0,x_max)
    axis.set_ylim(0,y_max*1.1)
    
def _build_model_image(model=None):
    
    cells       = np.vstack((model.coefficients,model.x_offsets,model.z_offsets))
    img         = _make_cell_image(cells,model)
    return img
    
def _prepare_data(model=None,data=None):
    
    synthetic_data             = model.get_synthetic_data(data)
    
    gz_data                    = data.getXData()
    x_data                     = data.getXLocations()
    
    data_dict                  = {}
    data_dict["model z span"]  = (model.z0,model.z1)
    data_dict["model x span"]  = (model.x0,model.x1)
    data_dict["Observations"]  = gz_data
    data_dict["Modeled data"]  = synthetic_data
    data_dict["Location data"] = x_data
    
    return data_dict

def _make_cell_image(cells,model):
    data = _color_data(cells[0])
    size = (model.divx*10,model.divz*10)
    cellx = size[0]/model.divx
    cellz = size[1]/model.divz
    img_x_coords = np.arange(0,size[0],cellx)
    img_z_coords = np.arange(0,size[1],cellz)
    img = image.new("RGBA", (size[0],size[1]),0)
    data_index = 0
    for x in img_x_coords:
        for z in img_z_coords:
            newz = int(size[1]- z - cellz)
            cell = PILCell(size=(int(cellx),int(cellz)),value=data[data_index])
            img.paste(cell.getImg(),(int(x),newz),mask=cell.getImg())
            data_index+=1
    return img

    
def _color_data(values):  
    normVals = mpl.colors.Normalize()
    mapper   = mpl.cm.ScalarMappable(normVals, cmap=mpl.cm.get_cmap("plasma",lut=400))
    
    return mapper.to_rgba(values,bytes=True)

class PILCell():
    
    def __init__(self, size=(10, 10), border=1, value=0):
        self.img = image.new('RGBA', size, tuple(value))    
        
    def getImg(self):
        return self.img