import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#########################################################################################################################################
#########################################################################################################################################
###################################### Simple CLASS for SDOF system in free vibration ###################################################
#########################################################################################################################################
#########################################################################################################################################

class Simple_free_motion():
    def __init__(self, xo, xvo, w, to, dt, tf):
        self.xo = xo
        self.xvo = xvo
        self.w = w
        self.to = to
        self.dt = dt
        self.tf = tf
    
    def sim_free_sdof_nodamp(self):
        xo = self.xo
        xvo = self.xvo
        w = self.w
        to = self.to
        dt = self.dt
        tf = self.tf
        x = np.zeros((int((tf-to)/dt), 1))
        ti = np.zeros((len(x), 1))

        j = 0
        for t in np.arange(to,tf, dt):
            x[j] = xo * np.cos(w*t) + xvo / w * np.sin(w*t)
            ti[j] = t
            j = j+1

        return x, ti

#########################################################################################################################################
#########################################################################################################################################
############################################# Simple CLASS for Plot Amplitude vs Time ###################################################
#########################################################################################################################################
#########################################################################################################################################

class plt_amp_tim():
    def __init__(self, x = any, ti = any, color = [0, 0, 0], title = 'Amplitude vs Time',
                 ylabel = 'Amplitude', grl = 1):
        self.x = x
        self.ti = ti
        self.color = color
        self.title = title
        self.ylabel = ylabel
        self.grl = grl

    def plot_Avst(self):
        x = self.x
        ti = self.ti
        color = self.color
        title = self.title
        ylabel = self.ylabel
        grl = self.grl
        #------------------------- PLOTS -----------------------------#
        fig, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(ti, x, color = color, alpha = 0.5 ,lw = grl, ls = '-', marker = 'o', markersize = 0, label = ylabel)
        ax.set_title(title, fontsize = 12, color = [0,0,1], fontweight = 'bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time [s]')
        ax.grid(visible= True, axis= 'x')
        ax.set_xlim(ti[0], ti[-1])
        plt.show()
        
#########################################################################################################################################
#########################################################################################################################################
################################# Simple CLASS for SDOF system in free vibration (Sub Damping) ##########################################
#########################################################################################################################################
#########################################################################################################################################

class Simple_free_motion_sub_damping():
    def __init__(self,T = 0.1, xo = 1.0, xvo = 0.0, zi = 0.05, to = 0, tf = 5.0, dt = 0.001):
        self.T = T
        self.xo = xo
        self.xvo = xvo
        self.zi = zi
        self.to = to
        self.tf = tf
        self.dt = dt
    
    def sim_free_sdof_SubDamping(self):
        T = self.T
        xo = self.xo
        xvo = self.xvo
        zi = self.zi
        to = self.to
        tf = self.tf
        dt = self.dt
        
        w = (2 * np.pi)/T
        xsub = np.zeros(int(tf/dt + 1))
        ti = np.zeros(int(tf/dt + 1))
        wsub = w * np.sqrt(1 - zi**2)
        j = 0
        for t in np.arange(to,tf + dt,dt):
            xsub[j] = np.exp(-zi*w*t)*(xo*np.cos(wsub*t) + (xvo + zi*w*xo)/(wsub)*np.sin(wsub*t))
            ti[j] = t
            j = j+1
        
        return xsub, ti
        