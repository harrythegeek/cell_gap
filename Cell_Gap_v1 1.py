import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

from scipy.optimize import curve_fit
 ###Peak detection and analysis
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
# C0=1.499703612
# C1=0.0061723
# C2=0.0

# C00=1.719165086
# C01=0.014771947
# C02=2.47E-03

########################################
# lista=pd.read_csv(r"T:\users\PatrickP\Measuring Lens opd\mid-4L-4H.txt", sep=",", header=None)[0].tolist() 
# listb=lista[13:] #shorter version
# # PretiltFileName=r'\\fefs01\Technical3\data\R\R000\GoldenSample-pretilt\GoldenSample-90deg.xlsx'

# p1_data=[]
# for i in range(len(listb)):
#     p1_data.append( listb[i].split('\t') )
    
# p1_data=np.array(p1_data)
# p1_data = p1_data.astype(np.float)

# p1_lambda=p1_data[:,0]
# dat_from=760
# dat_to=1330
# p1_lambda=p1_lambda[dat_from:dat_to]
# p1_data_numpy=p1_data[dat_from:dat_to]

# plt.plot(p1_data_numpy)

##################

CellGapFileName=r'T:\users\PatrickP\UV-VIS\40umTac.Raw.csv'
# PretiltFileName=r'\\fefs01\Technical3\data\R\R253\optical\Pre-tilt\R253-1-8.xlsx'
PretiltFileName=r'T:\data\B\B124\optical\Pre-tilt\B124-7.xlsx'

# PretiltFileName=r'\\fefs01\Technical3\users\PatrickP\pre-tilt\Liz sample - B-20-04-2023-1\B-20-04-2023-1-vertical.xlsx'
All_csv=[]

with open(CellGapFileName, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar=' ')
      for row in spamreader:
          aa=row
          if row[0]=='nm,':
            aa[0]=(aa[0][0:-1])
            aa[1]=(aa[1])
            aa=np.array((aa))
            aa_initial=aa
            # All_csv.append(aa)
            
          elif row[0]!='nm,':   
              
              ###For Lumerical Format
              # aa_new=aa[0].split(",")
              # aa=[np.float(aa_new[0]), np.float(aa_new[1])]
              # aa[1]=np.float(aa[1])
              # aa=np.array(aa)
              # All_csv.append(aa)
              ###For UV-Vis format
              aa[0]=np.float(aa[0][0:-1])
              aa[1]=np.float(aa[1])
              aa=np.array(aa)
              All_csv.append(aa)
              
              
              
              # while True:
              #     try:
              #       # aa[0]=np.float(aa[0][0:-1])
                    
              #       # aa[1]=np.float(aa[1])
              #       # aa=np.array(aa)
              #       # All_csv.append(aa)
              #       break
              #     except ValueError:
              #       aa_new=aa[0].split(",")
              #       aa=[np.float(aa_new[0]), np.float(aa_new[1])]
                        
              #       aa[1]=np.float(aa[1])
              #       aa=np.array(aa)
              #       All_csv.append(aa)
   

p1_data=np.array(All_csv)
p1_data=p1_data[::-1]
       
dat_from=0 #starting further away to avoid high absorption uv region #100 normally
# dat_to=len(All_csv)
dat_to=len(p1_data)

p1_lambda=p1_data[:,0][dat_from:dat_to]
p1_data_numpy=p1_data[:,1][dat_from:dat_to]

plt.plot(p1_data_numpy)


####ficus detrending the data code
# slope_n=(800-750)*10+1
# p1_lambda=np.ce(750,800,slope_n)
# p1_lambda=p1_lambda[0:-1]

# deslope_all=[]
# for i in range(len(p1_lambda)):
#     y_slope=0.6046*p1_lambda[i]-422.2
#     deslope_all.append(y_slope)
    
# p1_data_numpy_new=[]
# for i in range(len(p1_lambda)): 
#         p1_data_numpy_new.append( np.float(p1_data_numpy[i]+deslope_all[-i-1]) )

# plt.plot(p1_data_numpy_new)

# p1_data_numpy=p1_data_numpy_new # new y desloped
# ##########################


# p1_data=pd.read_excel(r"\\fefs01\Technical3\users\PatrickP\Hyperspectral\Test58-FOCALPLANEBOTTOM-vL2-Vh6-LENSON-flat-narrow-litroom-5nmsteps\PointA.xlsx")

# # # p1_data=pd.read_excel(r"T:\users\PatrickP\CellGapScans\Surround-vs-middle\new lc split\Q945-5-6 -mid.xlsx")

# p1_lambda=np.array(p1_data["nm"]) #indexing to take only 430nm to 670nm data 6878:1393

# p1_lambda=np.array(p1_data["nm"][0:-150])#[758:1300]) #indexing to take only 430nm to 670nm data 6878:1393
# #prev[692:1394]

# dat_col=len(p1_data.columns)

# p1_data_numpy=np.array(p1_data)#[758:1300]) #reformating for for loop 688:1393
#
# plt.plot(p1_lambda,p1_data_numpy)
# ==================================================
# import os

# # folder path
# # dir_path = r'T:\users\PatrickP\CellGapScans\Surround-vs-middle'
# dir_path=r'T:\users\PatrickP\Thermoformed-Experiments'

# # list to store files
# all_fs = []
# # Iterate directory
# for file in os.listdir(dir_path):
#     # check only text files
#     if file.endswith('.txt'):
#         all_fs.append(file)
# print(all_fs)

# p_all_data=[] #work in progress
# for i in range(len(all_fs)): #i=2
#     f = open(dir_path + '\\' + all_fs[i], "r")
#     content = f.read().splitlines()
#     del content[0:14]
#     # content_arr=np.float(content)
    
#     content_arr_all=[]
#     content_arr_all=np.empty([len(content),2])
#     for j in range(len(content)): #i=0
#         # wave=float(content[i].split("\t")[0]) #wavelength
#         # I=float(content[i].split("\t")[1]) #I
#         content_arr_all[j,0]=float(content[j].split("\t")[0]) #wavelength j=1
#         content_arr_all[j,1]=float(content[j].split("\t")[1]) #I
    
    
#     f.close()
#     p_all_data.append(dir_path + all_fs[i])
    
# p1_lambda=np.array(content_arr_all[:,0][720:1400]) 
# dat_col=1
# # p1_data_numpy=np.array(content_arr_all[:,1][600:1300]) #reformating for for loop 688:1393
# p1_data_numpy=content_arr_all[720:1400]
# plt.plot(p1_data_numpy[:,0],p1_data_numpy[:,1])
# ===================================================

def cauchy_findinder(a0,b0,c0,d0,ae,be,ce,de,lamb_a,lamb_b,lamb_c,lamb_d): #finds Cauchy coeffs. based on no and ne values
    
    #initial guess values
    C0_g=1.5
    C1_g=0
    C2_g=0
    
    # C00_g=1.7
    # C01_g=0
    # C02_g=0.001
    
    #sample data
    x = np.array([lamb_a*0.001,lamb_b*0.001,lamb_c*0.001,lamb_d*0.001]) #scaled from nm to um
    y = np.array([[a0,b0,c0,d0], [ae,be,ce,de]])
    
    
    Cauchy_all=[]
    for i in range (0,2): 
            
        # define type of function to search
        def chy_func(x, C0, C1, C2):
            return C0 + (C1/x**2) + (C2/x**4)
         
        # curve fit
        p0 = (1.0, 1, 0) # starting search koefs
        opt, pcov = curve_fit(chy_func, x, y[i], p0)
        a, k, b = opt
        
        # test result
        x2 = np.linspace(lamb_a*0.001, lamb_d*0.001, 1000)
        y2 = chy_func(x2, a, k, b)
        fig, ax = plt.subplots()
        ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f + ( {%.3f x^2} % + .3f$ x^4)' % (a,k,b))
        ax.plot(x, y[i], 'bo', label='data')
        ax.legend(loc='best')
        plt.xlabel('nm/1000')
        plt.ylabel('n')
        plt.show()
        
        Cauchy_all.append(opt)
 
    return Cauchy_all

def cauchy_findinder_expanded(a0,b0,c0,d0,e0,f0,g0,h0,i0,j0,k0,ae,be,ce,de,ee,fe,ge,he,ie,je,ke,lamb_a,lamb_b,lamb_c,lamb_d,lamb_e,lamb_f,lamb_g,lamb_h,lamb_i,lamb_j,lamb_k):#advanced

    #testing
    #HTW114200-100LC
    # a0=1.5209 #no
    # b0=1.5098 #no
    # c0=1.5068 #no
    # d0=1.5015 #no

    
    # ae=1.8524 #ne
    # be=1.7951 #ne
    # ce=1.7843 #ne
    # de=1.7672 #ne
    
    # lamb_a=450 #nm
    # lamb_b=546 #nm
    # lamb_c=589 #nm
    # lamb_d=650 #nm
    
    #initial guess values
    C0_g=1.5
    C1_g=0
    C2_g=0
    
    # C00_g=1.7
    # C01_g=0
    # C02_g=0.001
    
    #sample data
    x = np.array([lamb_a*0.001,lamb_b*0.001,lamb_c*0.001,lamb_d*0.001,lamb_e*0.001,lamb_f*0.001,lamb_g*0.001,lamb_h*0.001,lamb_i*0.001,lamb_j*0.001,lamb_k*0.001]) #scaled from nm to um
    y = np.array([[a0,b0,c0,d0,e0,f0,g0,h0,i0,j0,k0], [ae,be,ce,de,ee,fe,ge,he,ie,je,ke]])
    
    
    Cauchy_all=[]
    for i in range (0,2): 
            
        # define type of function to search
        def chy_func(x, C0, C1, C2):
            return C0 + (C1/x**2) + (C2/x**4)
         
        # curve fit
        p0 = (1.0, 1, 0) # starting search koefs
        opt, pcov = curve_fit(chy_func, x, y[i], p0)
        a, k, b = opt
        
        # test result
        x2 = np.linspace(lamb_a*0.001, lamb_k*0.001, 1000)
        y2 = chy_func(x2, a, k, b)
        fig, ax = plt.subplots()
        ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f + ( {%.3f x^2} % + .3f$ x^4)' % (a,k,b))
        ax.plot(x, y[i], 'bo', label='data')
        ax.legend(loc='best')
        plt.xlabel('nm/1000')
        plt.ylabel('n')
        plt.show()
        
        Cauchy_all.append(opt)
 
    return Cauchy_all

# chy=cauchy_findinder(1.5209,1.5098,1.5068,1.5015,1.8524,1.7951,1.7843,1.7672,450,546,589,650) #Cauchy values for HTW114200_100LC

# chy=cauchy_findinder(1.532,1.516,1.504,1.501,1.848,1.782,1.757,1.743,450,546,589,650) #Cauchy values for HTW114200_100LC UPDATED BY MAY

# chy=cauchy_findinder(1.532,1.516,1.504,1.501,1.848,1.782,1.757,1.743,450,546,589,650) #Cauchy values for HTW11700-200-LC

# chy=cauchy_findinder(1.55,1.53,1.52,1.518,1.85,1.8,1.78,1.76,450,546,589,650) #Cauchy values for HTW11700-200-LC - TEST ONLY

chy=cauchy_findinder_expanded(1.532,1.528,1.523,1.518,1.514,1.508,1.503,1.502,1.501,1.496,1.494, 1.848,1.833,1.814,1.792,1.776,1.765,1.751,1.746,1.743,1.730,1.724,450,475,500,525,550,575,600,625,650,675,700) #Cauchy values for HTW111700-200-LC  - simulation data

# chy=cauchy_findinder(1.531,1.513,1.51,1.504,1.855,1.786,1.767,1.757,450,546,589,650)#Origami LC HTG118200-200

def main_pretilt(): #commenting out for speed
    wave_nm=np.linspace(237.73,800.24,1623)
    # df = pd.read_excel("Lambda-expanded.xlsx")
    # wave_nm=np.array(df["lambda"])
    
    wave_um=wave_nm/1000
    
    # wave_t_dat=np.array(df2["Wavelength (nm)"])
    # wave_t_dat_angle=np.array(df2["Angle"])
    
    
    no_all=[]
    ne_all=[]
    for i in range(len(wave_um)):  
        no=chy[0][0] + (chy[0][1]/wave_um[i]**2) + (chy[0][2]/wave_um[i]**4)
        ne= chy[1][0] + (chy[1][1]/wave_um[i]**2) + (chy[1][2]/wave_um[i]**4)
        
        
        # no=C0 + (C1/wave_um[i]**2) + (C2/wave_um[i]**4)
        # ne= C00 + (C01/wave_um[i]**2) + (C02/wave_um[i]**4)
        
        no_all.append(no)
        ne_all.append(ne)
      
    
    # plt.plot(no_all)
    # plt.plot(ne_all)
    
    # tilt_data=pd.read_excel("pre-tilt B-All.xlsx")
    tilt_data = pd.read_excel(PretiltFileName)

    # tilt_data=pd.read_excel("R017-4-12-green-1.xlsx") 
    tilt_angle=np.array(tilt_data["Angles"])
    tilt_green=np.array(tilt_data["Green"])
    # tilt_green=np.array(tilt_data["test"])
    # tilt_green=np.array(tilt_data["SampleR017 4"])
    tilt_blue=np.array(tilt_data["Blue"])
    tilt_red=np.array(tilt_data["Red"])
    
    
    
    green_skew=tilt_data["Green"].skew()
    # green_skew=tilt_data["test"].skew()
    # plt.plot(tilt_angle,tilt_green) #green
    print(round(green_skew,3))
    
    blue_skew=tilt_data["Blue"].skew()
    # plt.plot(tilt_angle,tilt_blue) #blue
    print(round(blue_skew,3))
    
    red_skew=tilt_data["Red"].skew()
    # plt.plot(tilt_angle,tilt_red) #red
    print(round(red_skew,3))
    
    p1_intensity_smoothed=savgol_filter(tilt_green,15,1) # p1_intensity=-1*p1_intensity #normall 11,1
    p1_intensity_unsmoothed=tilt_green #unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed)
    plt.plot(tilt_angle,p1_intensity_smoothed, '-')
    
    peakstilt, _ = find_peaks(p1_intensity_smoothed,distance=21, prominence=(.1))
    
    
    just_peakstilt=[]
    peaks_waveltilt=[]
    for j in range(len(peakstilt)):
        just_peakstilt.append( p1_intensity_smoothed[ (peakstilt[j])] )
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])
    
    plt.plot(tilt_angle,p1_intensity_smoothed,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt,just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    #######
    p1_intensity_smoothed2=savgol_filter(-1*tilt_green,15,1) # p1_intensity=-1*p1_intensity 11,1
    p1_intensity_unsmoothed2=-1*tilt_green #unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed2)
    plt.plot(tilt_angle,p1_intensity_smoothed2, '-')
    
    peakstilt2, _ = find_peaks(p1_intensity_smoothed2,distance=15, prominence=(.01))
    
    
    just_peakstilt2=[]
    peaks_waveltilt2=[]
    for j in range(len(peakstilt2)):
        just_peakstilt2.append( p1_intensity_smoothed2[ (peakstilt2[j])] )
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])
        
    
    plt.plot(tilt_angle,p1_intensity_smoothed2,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2,just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    peaks_troughs=[[peaks_waveltilt,just_peakstilt], [peaks_waveltilt2,just_peakstilt2]]
    
    nearest_to_0_abs=[]
    nearest_to_0=[]
    for i in range( len(peaks_troughs) ):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append( abs(peaks_troughs[i][0][k]) ) #i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])
    
    green_centre=nearest_to_0[ nearest_to_0_abs.index(min(nearest_to_0_abs)) ]# finds peak or trough closes to 0 and uses it for subsequent analysis
    print(green_centre)
    # if len(peaks_waveltilt2)==2:#Relying on two peaks with assumed middle in between them
    #     green_centre=peaks_waveltilt[0]+peaks_waveltilt[1] 
    # if len(peaks_waveltilt)==2:#Relying on two peaks with assumed middle in between them
    #     green_centre=peaks_waveltilt2[0]+peaks_waveltilt2[1]  #green_centre=0
    
    ############
    p1_intensity_smoothed=savgol_filter(tilt_blue,7,1) # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed=tilt_blue #unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed)
    plt.plot(tilt_angle,p1_intensity_smoothed, '-')
    
    peakstilt, _ = find_peaks(p1_intensity_smoothed,distance=15, prominence=(.1))
    
    
    just_peakstilt=[]
    peaks_waveltilt=[]
    for j in range(len(peakstilt)):
        just_peakstilt.append( p1_intensity_smoothed[ (peakstilt[j])] )
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])
    
    plt.plot(tilt_angle,p1_intensity_smoothed,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt,just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    #######
    p1_intensity_smoothed2=savgol_filter(-1*tilt_blue,15,1) # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed2=-1*tilt_blue #unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed2)
    plt.plot(tilt_angle,p1_intensity_smoothed2, '-')
    
    peakstilt2, _ = find_peaks(p1_intensity_smoothed2,distance=15, prominence=(.1))
    
    
    just_peakstilt2=[]
    peaks_waveltilt2=[]
    for j in range(len(peakstilt2)):
        just_peakstilt2.append( p1_intensity_smoothed2[ (peakstilt2[j])] )
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])
        
    
    plt.plot(tilt_angle,p1_intensity_smoothed2,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2,just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    peaks_troughs=[[peaks_waveltilt,just_peakstilt], [peaks_waveltilt2,just_peakstilt2]]
    
    nearest_to_0_abs=[]
    nearest_to_0=[]
    for i in range( len(peaks_troughs) ):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append( abs(peaks_troughs[i][0][k]) ) #i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])
    
    # blue_centre=min(nearest_to_0) # finds peak or trough closes to 0 and uses it for subsequent analysis
    blue_centre=nearest_to_0[ nearest_to_0_abs.index(min(nearest_to_0_abs)) ]# finds peak or trough closes to 0 and uses it for subsequent analysis
    print(blue_centre)
    # if len(peaks_waveltilt)==2:#Relying on two peaks with assumed middle in between them
    #     blue_centre=peaks_waveltilt[0]+peaks_waveltilt[1] 
    # if len(peaks_waveltilt2)==2:#Relying on two peaks with assumed middle in between them
    #     blue_centre=peaks_waveltilt2[0]+peaks_waveltilt2[1] #blue_centre=-11
    # ###

    ############
    p1_intensity_smoothed=savgol_filter(tilt_red,15,1) # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed=tilt_red#unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed)
    plt.plot(tilt_angle,p1_intensity_smoothed, '-')
    
    peakstilt, _ = find_peaks(p1_intensity_smoothed,distance=20, prominence=(.5))
    
    
    just_peakstilt=[]
    peaks_waveltilt=[]
    for j in range(len(peakstilt)):
        just_peakstilt.append( p1_intensity_smoothed[ (peakstilt[j])] )
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])
    
    plt.plot(tilt_angle,p1_intensity_smoothed,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt,just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    #######
    p1_intensity_smoothed2=savgol_filter(-1*tilt_red,15,1) # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed2=-1*tilt_red #unsmoothed
    plt.plot(tilt_angle,p1_intensity_unsmoothed2)
    plt.plot(tilt_angle,p1_intensity_smoothed2, '-')
    
    peakstilt2, _ = find_peaks(p1_intensity_smoothed2,distance=20, prominence=(.05))
    
    
    just_peakstilt2=[]
    peaks_waveltilt2=[]
    for j in range(len(peakstilt2)):
        just_peakstilt2.append( p1_intensity_smoothed2[ (peakstilt2[j])] )
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])
        
    
    plt.plot(tilt_angle,p1_intensity_smoothed2,'-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2,just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.show()
    
    
        
    peaks_troughs=[[peaks_waveltilt,just_peakstilt], [peaks_waveltilt2,just_peakstilt2]]
    
    nearest_to_0_abs=[]
    nearest_to_0=[]
    for i in range( len(peaks_troughs) ):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append( abs(peaks_troughs[i][0][k]) ) #i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])
    
    red_centre=nearest_to_0[ nearest_to_0_abs.index(min(nearest_to_0_abs)) ]# finds peak or trough closes to 0 and uses it for subsequent analysis
    print(red_centre)
    # if len(peaks_waveltilt2)==2:#Relying on two peaks with assumed middle in between them
    #     red_centre=peaks_waveltilt[0]+peaks_waveltilt[1] 
    # if len(peaks_waveltilt)==2:#Relying on two peaks with assumed middle in between them
    #     red_centre=peaks_waveltilt2[0]+peaks_waveltilt2[1]  #red_centre=8

    ##########################
    
    plt.plot(tilt_angle,tilt_green,color='g') #green
    plt.axvline(green_centre,color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('Green wavelength ' + str(green_centre) + " degrees", color='green')
    
    green=532 # or 589?
    def closest(wave_nm, green):   
        return wave_nm[min(range(len(wave_nm)), key = lambda i: abs(wave_nm[i]-green))]
    
    wave_green_match=closest(wave_nm,green)
    
    for i in range(len(wave_nm)):
        if wave_green_match==wave_nm[i]:
            wave_green_match_index=i 
    
    no_lambda_green=no_all[wave_green_match_index]
    neff_lambda_green=ne_all[wave_green_match_index]
    delta_green=neff_lambda_green-no_lambda_green
    pretilt_green_deg=math.asin(math.sin( (green_centre)*math.pi/180) /(no_lambda_green + neff_lambda_green))*180/math.pi
    
    
    plt.plot(tilt_angle,tilt_red,color='r') 
    plt.axvline(red_centre,color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('Red wavelength ' + str(red_centre) + " degrees", color='red')
    
    red=639
    def closest(wave_nm, red):   
        return wave_nm[min(range(len(wave_nm)), key = lambda i: abs(wave_nm[i]-red))]
    
    wave_red_match=closest(wave_nm,red)
    
    for i in range(len(wave_nm)):
        if wave_red_match==wave_nm[i]:
            wave_red_match_index=i 
    
    no_lambda_red=no_all[wave_red_match_index]
    neff_lambda_red=ne_all[wave_red_match_index]
    delta_red=neff_lambda_red-no_lambda_red
    pretilt_red_deg=math.asin(math.sin( (red_centre)*math.pi/180) /(no_lambda_red + neff_lambda_red))*180/math.pi
    
    
    plt.plot(tilt_angle,tilt_blue,color='b') #green
    plt.axvline(blue_centre,color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title("Blue wavelength " + str(blue_centre) + " degrees" , color='blue')
    
    blue=439
    def closest(wave_nm, blue):   
        return wave_nm[min(range(len(wave_nm)), key = lambda i: abs(wave_nm[i]-blue))]
    
    wave_blue_match=closest(wave_nm,blue)
    
    
    for i in range(len(wave_nm)):
        if wave_blue_match==wave_nm[i]:
            wave_blue_match_index=i 
    
    no_lambda_blue=no_all[wave_blue_match_index]
    neff_lambda_blue=ne_all[wave_blue_match_index]
    delta_blue=neff_lambda_blue-no_lambda_blue
    pretilt_blue_deg=math.asin(math.sin( (blue_centre)*math.pi/180) /(no_lambda_red + neff_lambda_red))*180/math.pi
    
    print('pre-tilt green' + '=' + str(round(pretilt_green_deg,3)))
    print('pre-tilt red' + '=' + str(round(pretilt_red_deg,3)))
    print('pre-tilt blue' + '=' + str(round(pretilt_blue_deg,3)))

    avg_pretilt=(pretilt_green_deg+pretilt_red_deg+pretilt_blue_deg)/3
    print('avg pre-tilt' + '=' + str(round(avg_pretilt,1)))
    
    
    
    
    ##################
    avg_pretilt=1.5 #0 for testing
      
    neff_all=[]
    for j in range(len(wave_um)):  #j=0
        neff=1/ ( math.sqrt( ((math.sin(avg_pretilt*math.pi/180)/no_all[j])**2 ) + ((math.cos(avg_pretilt*math.pi/180)/ne_all[j])**2) ))   
        neff_all.append(neff)
        
        
    delta_neff_all=[]
    for i in range(len(neff_all)):
        delta_neff_all.append( neff_all[i]- no_all[i])
    
    
    # neff_delta_all=[]
    delta_n_all=[]
    for k in range(len(wave_um)):
        # neff_delta=neff_all[k]-no_all[k]
        delta_n=ne_all[k]-no_all[k]
        
        # neff_delta_all.append(neff_delta)
        delta_n_all.append(delta_n)
        
        
    # plt.plot(delta_neff_all, '*')
    
    # plt.plot(wave_nm,no_all, label="no")
    # plt.ylabel('#') #no
    # plt.xlabel('Wavelength (nm)')
    # plt.plot(wave_nm,ne_all, label="ne")
    # plt.legend(loc="upper right")
    # plt.ylabel('#') #ne
    # plt.xlabel('Wavelength (nm)')
    # plt.show()
    
    return (avg_pretilt,delta_neff_all,wave_nm,no_all,neff_all,wave_nm)

# main_pretilt()

avg_pretilt=main_pretilt()[0]

delta_neff_all=main_pretilt()[1]

wave_nm=main_pretilt()[2]

no_all=main_pretilt()[3]

neff_all=main_pretilt()[4]

wave_nm=main_pretilt()[5]

# #air only
# delta_neff_all=np.ones([len(delta_neff_all),1])
# neff_all=np.ones([len(neff_all),1])
# no_all=np.ones([len(no_all),1])


# neff_all=1.5*neff_all #test
# no_all=1.5*no_all #test
# delta_neff_all=1*delta_neff_all #test

###############

 
# p1_data=pd.read_excel("P1.xlsx")
# p1_lambda=np.array(p1_data["lambda"])
# p1_intensity=np.array(p1_data["P1"])



#p1_data=pd.read_excel(r'C:\Users\Patrikas.Prusinskas\OneDrive - Flexenable\Desktop\MicroscopeData\Q9890-3-16r\Q9890-3-16r.xlsx')
# p1_data=pd.read_excel(r"T:\users\PatrickP\UV-VIS-ITO\Scan - Lambda 850+ 29 March 2022 15_00 GMT Summer Time\50pSlitWidth-Tungstenonly-SR.xlsx")
# p1_data=pd.read_excel(r"T:\users\PatrickP\CellGapScans\UV-VIS-ITO\Scan - Lambda 850+ 29 March 2022 15_00 GMT Summer Time\50pSlitWidth-Tungstenonly-SR.xlsx")
# p1_data=pd.read_excel(r"C:\Users\Patrikas.Prusinskas\.spyder-py3\10um-ThorlabsCell.xlsx")
# p1_data=pd.read_excel(r"T:\users\PatrickP\CellGapScans\0p1 vs 1nm scans Q945-5-5\Q945-5-5-1nm_CellGap.xlsx")
# p1_data=pd.read_excel(r"T:\users\PatrickP\CellGapScans\3\Q929-4-17-1d-15um-CellGapScan - On15.xlsx")


# p1_lambda=np.array(p1_data["lambda"][128:len(["lambda"]) ]) #indexing to take only 430nm to 670nm data 6878:1393
# p1_lambda=np.array(["lambda"]) #indexing to take only 430nm to 670nm data 6878:1393

# plt.plot(_numpy[:,i][128:len(_numpy[:,i])],'-')

# dat_col=len(.columns)

# _numpy=np.array(p1_data[128:len(p1_data["lambda"]) ]) #reformating for for loop 688:1393
# p1_data_numpy=np.array(p1_data) #reformating for for loop 688:1393
dat_col=1
all_wave_I_pos_neg=[]
for i in range(1,dat_col+1): #skips x axis data i=1
    wave_I_pos_neg=[]
    for k in range(0,2): #k=0
        if k==0:
            # p1_intensity=p1_data_numpy[:,i] #i=0
            p1_intensity=p1_data[:,i][dat_from:dat_to] #i=0 for .txt files
            # p1_intensity=np.array(p1_data_numpy) #new one for ficus
            if min(p1_intensity)<0:
                p1_intensity=p1_intensity-min(p1_intensity)
                
            p1_intensity=p1_intensity/max(p1_intensity)
            
            
            
        if k==1:
            # p1_intensity=-1*p1_data_numpy[:,i] #flips the data to use the trough data
            p1_intensity=-1*p1_data[:,i][dat_from:dat_to] #i=0.txt files
            # p1_intensity=-1*np.array(p1_data_numpy) #new one for ficus
            if min(p1_intensity)<0:
                p1_intensity=p1_intensity-min(p1_intensity)
            p1_intensity=p1_intensity/max(p1_intensity)
            
        # plt.plot(p1_lambda,p1_intensity)
        # plt.ylabel('Intensity')
        # plt.xlabel('Wavelength [nm]')
    
        
        # p1_intensity_smoothed_initial=savgol_filter(p1_intensity,3,1) # p1_intensity=-1*p1_intensity
        # p1_intensity_smoothed_initial=p1_intensity # p1_intensity=-1*p1_intensity #not smoothed test
        p1_intensity_smoothed_initial=p1_intensity
        #interpolation
        
        p1_intensity_smoothed_in = UnivariateSpline(p1_lambda, p1_intensity_smoothed_initial)
        p1_intensity_smoothed_in.set_smoothing_factor(0)
        
        p1_labmda_interp = np.linspace(p1_lambda[0], p1_lambda[-1], len(p1_lambda)*100)
        
        plt.plot(p1_labmda_interp, p1_intensity_smoothed_in(p1_labmda_interp), 'r', lw = 1)
        
        p1_intensity_smoothed=p1_intensity_smoothed_in(p1_labmda_interp)

        plt.plot(p1_lambda,p1_intensity)
        plt.show()
        

        # p1_labmda_interp=np.ce(p1_lambda[0], p1_lambda[-1], num=1001, endpoint=True)
        # p1_intensity_smoothed=interpolate.interp1d(p1_lambda,p1_intensity_smoothed_initial, kind='linear')
        
        # p1_intensity_smoothed=[]
        # for p in range(len(p1_labmda_interp)):
        #     p1_intensity_smoothed.append( np.interp(p1_labmda_interp[p],p1_lambda,p1_intensity_smoothed_initial)) #interpolated intenstsity
        
        
        plt.plot(p1_labmda_interp,p1_intensity_smoothed,'-')
        # end of interpolation
        
        p1_intensity_unsmoothed=p1_intensity #unsmoothed
        plt.plot(p1_lambda,p1_intensity_unsmoothed)
       
        
        
        peaks, _ = find_peaks(p1_intensity_smoothed,distance=1, prominence=(0.01))
        
        just_peaks=[]
        peaks_wavel=[]
        for j in range(len(peaks)):
            just_peaks.append( p1_intensity_smoothed[ (peaks[j])] )
            peaks_wavel.append(p1_labmda_interp[(peaks[j])])
            
        avg_peak=sum(just_peaks)/len(just_peaks)
        max_pk=max(just_peaks)
        
        # plt.plot(p1_lambda,p1_intensity,'*')
        # plt.ylabel('Intensity')
        # plt.xlabel('Wavelength [nm]')
        # plt.plot(p1_lambda,p1_intensity,'')
        plt.plot(p1_labmda_interp,p1_intensity_smoothed,'-')
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength [nm]')
        plt.plot(peaks_wavel,just_peaks, 'o')
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength [nm]')
        plt.show()
        
        wave_I=np.column_stack((peaks_wavel,just_peaks))
        start_peaks_del = input("Delete Early Peaks #")
        wave_I_del1=wave_I #copy list for filtering
        
        for j in range(int(start_peaks_del)):
            wave_I_del1=np.delete(wave_I_del1, 0, 0)
            # print(i)
            
        end_peaks_del = input("Delete Late Peaks #")
        for j in range(int(end_peaks_del)):
            wave_I_del1=np.delete(wave_I_del1, -1, 0)
            
        wave_I_pos_neg.append(wave_I_del1)
    all_wave_I_pos_neg.append(wave_I_pos_neg)
    

wave_nm_list=wave_nm.tolist()
avg_cell_gap=[]
for o in range(len(all_wave_I_pos_neg)): #o=0
    
    deltaneff_wave=[all_wave_I_pos_neg[o][0][:,0],all_wave_I_pos_neg[o][1][:,0]]  # match deltaneff to peak wavelength


    

    wave_nm_list_closest=[]
    wave_nm_list_closest_all=[]
    for i in range(len(deltaneff_wave)): #i=0
        wave_nm_list_closest=[]
        for j in range(len(deltaneff_wave[i])):
            wave_nm_list_closest.append( min(wave_nm_list, key=lambda x:abs(x-deltaneff_wave[i][j] )) )
        wave_nm_list_closest_all.append(wave_nm_list_closest)
    
    plt.plot(wave_nm_list_closest,'-*')

    missing_dat_all=[]
    res_pos = [wave_nm_list.index(i) for i in wave_nm_list_closest_all[0]]    
    if len(res_pos)==0:
        res_pos=[0]
    res_neg = [wave_nm_list.index(i) for i in wave_nm_list_closest_all[1]]  
    if len(res_neg)==0:
        res_pos=[0]
    
    deltaneff_found_pos=[]
    neff_pos_peaks=[]
    for j in range(len(res_pos)): #j=6
        deltaneff_found_pos.append( delta_neff_all[res_pos[j]] )
        neff_pos_peaks.append( neff_all[res_pos[j]] )
        
    
    deltaneff_found_neg=[]
    neff_neg_peaks=[]
    for j in range(len(res_neg)):
        deltaneff_found_neg.append( delta_neff_all[res_neg[j]] )
        neff_neg_peaks.append( neff_all[res_neg[j]] )
            
    deltaneff_found=[deltaneff_found_pos,deltaneff_found_neg]
    neff_found=[neff_pos_peaks,neff_neg_peaks]
    # plt.plot(deltaneff_found)
    #Method1 - simple
    pk_count_pos=len(deltaneff_found_pos)-1 
    pk_count_neg=len(deltaneff_found_neg)-1
    
    pk_count=[pk_count_pos,pk_count_neg]
    
    cell_gap=[]
    for i in range(len(all_wave_I_pos_neg[0])): #i=0
        cell_gap.append ( round(pk_count[i] * (all_wave_I_pos_neg[0][i][0][0] * all_wave_I_pos_neg[0][i][-1][0]) /( ( deltaneff_found[i][0] * all_wave_I_pos_neg[0][i][-1][0]) - ( deltaneff_found[i][-1] * all_wave_I_pos_neg[0][i][0][0])),2) )
    
    avg_cell_gap.append((cell_gap[0]+cell_gap[1])/2 )
    
    #Finding all adjecent cell gap values
    cell_gap_adjecent=[]
    for i in range(len(all_wave_I_pos_neg[0])): #i=0
        for j in range(len(all_wave_I_pos_neg[0][i]) -1):
            cell_gap_adjecent.append( 2*all_wave_I_pos_neg[0][i][j][0] * all_wave_I_pos_neg[0][i][j+1][0] / (( neff_found[i][j] * all_wave_I_pos_neg[0][i][j+1][0] ) - ( neff_found[i][j+1] * all_wave_I_pos_neg[0][i][j][0])) )
    
# plt.plot(cell_gap_adjecent,'*')
avg_cell_gap[0]=avg_cell_gap[0]/1000 #*0.5 #air-no pre-tilt in um
# cell_gap[0]=cell_gap[0]*0.5
# cell_gap[1]=cell_gap[1]*0.5

str(CellGapFileName[-23:-15]) #cell_gap=10

print('peak-trough cell gap' + str(cell_gap))
print( str(CellGapFileName[-26:-15])+ '  averaged cell gap = ' + str(round(avg_cell_gap[0],3)))
print ('Peak Count' + '=' + str(pk_count_pos+1))
# missing_dat = float((str(saved_error)).split(' ')[0])


export_name=CellGapFileName
words = export_name.split('\\') 
export_name_short=words[0]+"\\"+words[1]+"\\"+words[2]+"\\"+words[3]+"\\"+words[4]+"\\"+words[5]+"\\"
end_name=words[-1][0:6]+"_cell-gap"
                          
cell_gap_export= open(export_name_short+"\\"+ str(end_name) + ".txt","w+")
for i in range(1):
     cell_gap_export.write( str(words[-1][0:6])+ "," + str(round(avg_cell_gap[0],3)) +"um" )
cell_gap_export.close()
