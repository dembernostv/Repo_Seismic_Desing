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
        
#########################################################################################################################################
#########################################################################################################################################
############################################ Response step by step (B-Newmark [Jr]) #####################################################
#########################################################################################################################################
#########################################################################################################################################

class Step_by_Step_BNewmark():
    
    def __init__(self, T= 1.0, M = 1.0, zi = 0.05, accel_record = any, vector_time = any,
                 colorSeism = (0.5,0.5,0.5), colorRaccel = (0,0,1),colorRTaccel = (1,0,0)):
        self.T = T
        self.M = M
        self.zi = zi
        self.SG = accel_record 
        self.TI = vector_time
        self.colorSeism = colorSeism
        self.colorRaccel = colorRaccel
        self.colorRTaccel = colorRTaccel
        
    
    def Bnewmark_Jr(self):
        T = self.T
        M = self.M
        zi = self.zi
        SG = self.SG
        TI = self.TI
        
        dt = TI[1] - TI[0]
        
        w = (2*np.pi)/T
        k = ((2*np.pi)/T)**(2)*M
        xo = 0
        xvo = 0

        xvn = xvo
        xn = xo
        xao = (-SG[0] * M - 2*zi*w*M*xvo -w**(2)*xo) * 1/M

        print(len(SG))

        xn1 = np.zeros(len(SG))
        xvn1 = np.zeros(len(SG))
        xan1 = np.zeros(len(SG))
        at = np.zeros(len(SG))

        xan = xao

        xan1[0] = xao
        xvn1[0] = xvo
        xn1[0] = xo  

        for i in np.arange(1, len(SG),1):
            xn1[i] = xn + dt*xvn + (dt**2)/2*xan
            xan1[i] = 1 / (M + (1/2)*(2*zi*w*M*dt)) * (-SG[i]*M - k*xn1[i] - 2*zi*w*M*(xvn + dt*(1 - 1/2)*xan))
            xvn1[i] = xvn + dt*((1-1/2)*xan + (1/2)*xan1[i])
            at[i] = SG[i] + xan1[i]
            
            xan = xan1[i]
            xvn = xvn1[i]
            xn = xn1[i]
        
        return at, xan1, xvn1, xn1  
    
    def plot_seis_Raccel(self, xan1, at):
        TI = self.TI
        SG = self.SG
        T = self.T
        zi = self.zi
        colorSeism = self.colorSeism
        colorRaccel = self.colorRaccel
        colorRTaccel = self.colorRTaccel
        
        maxSG = np.max(np.abs(SG))
        TI_maxSG = TI[np.argmax(np.abs(SG))]
        
        maxAT = np.max(np.abs(at))
        TI_maxAT = TI[np.argmax(np.abs(at))]
        
        fig, ax = plt.subplots(2,1, figsize = (20,10))
        fig.suptitle(f"B-Newmark, Solver", fontsize=18, color = (0,0,1), y=0.98)
        
        ax[0].plot(TI, SG, color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Seismic Record')
        ax[0].plot(TI_maxSG, SG[np.argmax(np.abs(SG))], color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'PGA = {maxSG:.4f} [g], t = {TI_maxSG:.4f} [s]')
        ax[0].set_title('Seismic Record')
        ax[0].set_ylabel('Acceleration [g]')
        ax[0].set_xlabel('Time [s]')
        ax[0].grid(visible= True, axis= 'x')
        ax[0].set_xlim(TI[0], TI[-1])
        ax[0].legend(loc='best')
        
        ax[1].plot(TI, SG, color = colorSeism, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Seismic Record')
        ax[1].plot(TI, xan1, color = colorRaccel, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 0, label = 'Acceleration Response')
        ax[1].plot(TI, at, color = colorRTaccel, alpha = 1.0 ,lw = 1.0, ls = '--', marker = 'o', 
                markersize = 0, label = 'Total Acceleration Response')
        ax[1].plot(TI_maxAT, at[np.argmax(np.abs(at))], color = colorRTaccel, alpha = 1.0 ,lw = 1.0, ls = '-', marker = 'o', 
                markersize = 5, label = f'maxAT = {maxAT:.4f} [g], t = {TI_maxAT:.4f} [s]')
        ax[1].set_title(f'Acceleration Response, T = {T:.2f} [s], zi = {zi * 100:.2f} [%]')
        ax[1].set_ylabel('Acceleration [g]')
        ax[1].set_xlabel('Time [s]')
        ax[1].grid(visible= True, axis= 'x')
        ax[1].set_xlim(TI[0], TI[-1])
        ax[1].legend(loc='best')
        
        plt.show()        
            
            
            
            
        
          
    
