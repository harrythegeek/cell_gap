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




def cauchy_findinder(a0, b0, c0, d0, ae, be, ce, de, lamb_a, lamb_b, lamb_c,
                     lamb_d):  # finds Cauchy coeffs. based on no and ne values

    # initial guess values
    C0_g = 1.5
    C1_g = 0
    C2_g = 0

    # C00_g=1.7
    # C01_g=0
    # C02_g=0.001

    # sample data
    x = np.array([lamb_a * 0.001, lamb_b * 0.001, lamb_c * 0.001, lamb_d * 0.001])  # scaled from nm to um
    y = np.array([[a0, b0, c0, d0], [ae, be, ce, de]])

    Cauchy_all = []
    for i in range(0, 2):
        # define type of function to search
        def chy_func(x, C0, C1, C2):
            return C0 + (C1 / x ** 2) + (C2 / x ** 4)

        # curve fit
        p0 = (1.0, 1, 0)  # starting search koefs
        opt, pcov = curve_fit(chy_func, x, y[i], p0)
        a, k, b = opt

        # test result
        x2 = np.linspace(lamb_a * 0.001, lamb_d * 0.001, 1000)
        y2 = chy_func(x2, a, k, b)
        fig, ax = plt.subplots()
        ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f + ( {%.3f x^2} % + .3f$ x^4)' % (a, k, b))
        ax.plot(x, y[i], 'bo', label='data')
        ax.legend(loc='best')
        plt.xlabel('nm/1000')
        plt.ylabel('n')
        plt.show()

        Cauchy_all.append(opt)

    return Cauchy_all


def cauchy_findinder_expanded(a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, ae, be, ce, de, ee, fe, ge, he, ie, je, ke,
                              lamb_a, lamb_b, lamb_c, lamb_d, lamb_e, lamb_f, lamb_g, lamb_h, lamb_i, lamb_j,
                              lamb_k):  # advanced

    # initial guess values
    C0_g = 1.5
    C1_g = 0
    C2_g = 0

    # sample data
    x = np.array(
        [lamb_a * 0.001, lamb_b * 0.001, lamb_c * 0.001, lamb_d * 0.001, lamb_e * 0.001, lamb_f * 0.001, lamb_g * 0.001,
         lamb_h * 0.001, lamb_i * 0.001, lamb_j * 0.001, lamb_k * 0.001])  # scaled from nm to um
    y = np.array([[a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0], [ae, be, ce, de, ee, fe, ge, he, ie, je, ke]])

    Cauchy_all = []
    for i in range(0, 2):
        # define type of function to search
        def chy_func(x, C0, C1, C2):
            return C0 + (C1 / x ** 2) + (C2 / x ** 4)

        # curve fit
        p0 = (1.0, 1, 0)  # starting search koefs
        opt, pcov = curve_fit(chy_func, x, y[i], p0)
        a, k, b = opt

        # test result
        x2 = np.linspace(lamb_a * 0.001, lamb_k * 0.001, 1000)
        y2 = chy_func(x2, a, k, b)
        fig, ax = plt.subplots()
        ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f + ( {%.3f x^2} % + .3f$ x^4)' % (a, k, b))
        ax.plot(x, y[i], 'bo', label='data')
        ax.legend(loc='best')
        plt.xlabel('nm/1000')
        plt.ylabel('n')
        plt.title('expanded cauchy')
        plt.show()

        Cauchy_all.append(opt)

    return Cauchy_all
chy=cauchy_findinder_expanded(1.532,1.528,1.523,1.518,1.514,1.508,1.503,1.502,1.501,1.496,1.494, 1.848,1.833,1.814,1.792,1.776,1.765,1.751,1.746,1.743,1.730,1.724,450,475,500,525,550,575,600,625,650,675,700) #Cauchy values for HTW111700-200-LC  - simulation data

PretiltFileName=r'C:\Users\Harry.Delalis\OneDrive - Flexenable\Desktop\salamander_prettrial\real panel 1\xlsx files\salamander_pretrial_RN4524_panel_01_cell_21.xlsx'
def main_pretilt():  # commenting out for speed
    wave_nm = np.linspace(237.73, 800.24, 1623)
    # df = pd.read_excel("Lambda-expanded.xlsx")
    # wave_nm=np.array(df["lambda"])

    wave_um = wave_nm / 1000

    # wave_t_dat=np.array(df2["Wavelength (nm)"])
    # wave_t_dat_angle=np.array(df2["Angle"])

    no_all = []
    ne_all = []
    for i in range(len(wave_um)):
        no = chy[0][0] + (chy[0][1] / wave_um[i] ** 2) + (chy[0][2] / wave_um[i] ** 4)
        ne = chy[1][0] + (chy[1][1] / wave_um[i] ** 2) + (chy[1][2] / wave_um[i] ** 4)

        no_all.append(no)
        ne_all.append(ne)

    tilt_data = pd.read_excel(PretiltFileName)

    tilt_angle = np.array(tilt_data["Angles"])

    #these are intensities
    tilt_green = np.array(tilt_data["Green"])
    tilt_blue = np.array(tilt_data["Blue"])
    tilt_red = np.array(tilt_data["Red"])


    green_skew = tilt_data["Green"].skew() #skew is a measurement of assymetry for a distribution
    print('green skew')
    print(round(green_skew, 3))

    blue_skew = tilt_data["Blue"].skew()
    print('blue skew')
    print(round(blue_skew, 3))

    red_skew = tilt_data["Red"].skew()
    print('red skew')
    print(round(red_skew, 3))

    p1_intensity_smoothed = savgol_filter(tilt_green, 15, 1)  # p1_intensity=-1*p1_intensity #normall 11,1
    p1_intensity_unsmoothed = tilt_green  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed)
    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.title('green smoothed and unsmoothed')
    plt.show()

    peakstilt, _ = find_peaks(p1_intensity_smoothed, distance=21, prominence=(.1)) #not sure what the underscore is
    print('peak points for green',peakstilt)

    #here we are associating tilt angles with intensity values
    just_peakstilt = []   #intensities
    peaks_waveltilt = []  #angles for the above intensities

    for j in range(len(peakstilt)):
        just_peakstilt.append(p1_intensity_smoothed[(peakstilt[j])])
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])
    print('just_peakstilt',just_peakstilt)
    print('peaks_wavetilt',peaks_waveltilt)

    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt, just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('plot peak points on the graph')
    plt.show()



    #######
    p1_intensity_smoothed2 = savgol_filter(-1 * tilt_green, 15, 1)  # p1_intensity=-1*p1_intensity 11,1
    p1_intensity_unsmoothed2 = -1 * tilt_green  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed2)
    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.title('inverse of green')
    plt.show()

    peakstilt2, _ = find_peaks(p1_intensity_smoothed2, distance=15, prominence=(.01))


    just_peakstilt2 = []
    peaks_waveltilt2 = []
    for j in range(len(peakstilt2)):
        just_peakstilt2.append(p1_intensity_smoothed2[(peakstilt2[j])])
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])

    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2, just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('peaks on the green inverse graph')
    plt.show()


    peaks_troughs = [[peaks_waveltilt, just_peakstilt], [peaks_waveltilt2, just_peakstilt2]]
    print(peaks_troughs)

    nearest_to_0_abs = []
    nearest_to_0 = []
    for i in range(len(peaks_troughs)):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append(abs(peaks_troughs[i][0][k]))  # i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])

    green_centre = nearest_to_0[nearest_to_0_abs.index(
        min(nearest_to_0_abs))]  # finds peak or trough closes to 0 and uses it for subsequent analysis
    print('green center')
    print(green_centre)

    p1_intensity_smoothed = savgol_filter(tilt_blue, 7, 1)  # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed = tilt_blue  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed)
    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.title('blue smoothed and not smoothed')
    plt.show()

    peakstilt, _ = find_peaks(p1_intensity_smoothed, distance=15, prominence=(.1))

    just_peakstilt = []
    peaks_waveltilt = []
    for j in range(len(peakstilt)):
        just_peakstilt.append(p1_intensity_smoothed[(peakstilt[j])])
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])

    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt, just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('peaks for blue graph')
    plt.show()

    #######
    p1_intensity_smoothed2 = savgol_filter(-1 * tilt_blue, 15, 1)  # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed2 = -1 * tilt_blue  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed2)
    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.title('inverse blue')
    plt.show

    peakstilt2, _ = find_peaks(p1_intensity_smoothed2, distance=15, prominence=(.1))

    just_peakstilt2 = []
    peaks_waveltilt2 = []
    for j in range(len(peakstilt2)):
        just_peakstilt2.append(p1_intensity_smoothed2[(peakstilt2[j])])
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])

    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2, just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('plot peaks on inverse blue graph')
    plt.show()

    peaks_troughs = [[peaks_waveltilt, just_peakstilt], [peaks_waveltilt2, just_peakstilt2]]

    nearest_to_0_abs = []
    nearest_to_0 = []
    for i in range(len(peaks_troughs)):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append(abs(peaks_troughs[i][0][k]))  # i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])

    # blue_centre=min(nearest_to_0) # finds peak or trough closes to 0 and uses it for subsequent analysis
    blue_centre = nearest_to_0[nearest_to_0_abs.index(
        min(nearest_to_0_abs))]  # finds peak or trough closes to 0 and uses it for subsequent analysis
    print('blue center')
    print(blue_centre)
    # if len(peaks_waveltilt)==2:#Relying on two peaks with assumed middle in between them
    #     blue_centre=peaks_waveltilt[0]+peaks_waveltilt[1]
    # if len(peaks_waveltilt2)==2:#Relying on two peaks with assumed middle in between them
    #     blue_centre=peaks_waveltilt2[0]+peaks_waveltilt2[1] #blue_centre=-11
    # ###

    ############
    p1_intensity_smoothed = savgol_filter(tilt_red, 15, 1)  # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed = tilt_red  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed)
    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.title('red smooth and unsmooth')
    plt.show()

    peakstilt, _ = find_peaks(p1_intensity_smoothed, distance=20, prominence=(.5))

    just_peakstilt = []
    peaks_waveltilt = []
    for j in range(len(peakstilt)):
        just_peakstilt.append(p1_intensity_smoothed[(peakstilt[j])])
        peaks_waveltilt.append(tilt_angle[(peakstilt[j])])

    plt.plot(tilt_angle, p1_intensity_smoothed, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt, just_peakstilt, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('peaks on red graph')
    plt.show()

    #######
    p1_intensity_smoothed2 = savgol_filter(-1 * tilt_red, 15, 1)  # p1_intensity=-1*p1_intensity
    p1_intensity_unsmoothed2 = -1 * tilt_red  # unsmoothed
    plt.plot(tilt_angle, p1_intensity_unsmoothed2)
    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.title('inverse of red graph')
    plt.show()

    peakstilt2, _ = find_peaks(p1_intensity_smoothed2, distance=20, prominence=(.05))

    just_peakstilt2 = []
    peaks_waveltilt2 = []
    for j in range(len(peakstilt2)):
        just_peakstilt2.append(p1_intensity_smoothed2[(peakstilt2[j])])
        peaks_waveltilt2.append(tilt_angle[(peakstilt2[j])])

    plt.plot(tilt_angle, p1_intensity_smoothed2, '-')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.plot(peaks_waveltilt2, just_peakstilt2, 'o')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('peaks on inverse graph')
    plt.show()

    peaks_troughs = [[peaks_waveltilt, just_peakstilt], [peaks_waveltilt2, just_peakstilt2]]

    nearest_to_0_abs = []
    nearest_to_0 = []
    for i in range(len(peaks_troughs)):
        for k in range(len(peaks_troughs[i][0])):
            nearest_to_0_abs.append(abs(peaks_troughs[i][0][k]))  # i=0  k=0
            nearest_to_0.append(peaks_troughs[i][0][k])

    red_centre = nearest_to_0[nearest_to_0_abs.index(
        min(nearest_to_0_abs))]  # finds peak or trough closes to 0 and uses it for subsequent analysis
    print('red center')
    print(red_centre)

    plt.plot(tilt_angle, tilt_green, color='g')  # green
    plt.axvline(green_centre, color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('Green wavelength ' + str(green_centre) + " degrees", color='green')


    green = 532  # or 589?

    def closest(wave_nm, green):
        return wave_nm[min(range(len(wave_nm)), key=lambda i: abs(wave_nm[i] - green))]

    wave_green_match = closest(wave_nm, green)

    for i in range(len(wave_nm)):
        if wave_green_match == wave_nm[i]:
            wave_green_match_index = i

    no_lambda_green = no_all[wave_green_match_index]
    neff_lambda_green = ne_all[wave_green_match_index]
    print('no green',no_lambda_green)
    print('neff green',neff_lambda_green)
    delta_green = neff_lambda_green - no_lambda_green
    pretilt_green_deg = math.asin(
        math.sin((green_centre) * math.pi / 180) / (no_lambda_green + neff_lambda_green)) * 180 / math.pi

    plt.plot(tilt_angle, tilt_red, color='r')
    plt.axvline(red_centre, color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    plt.title('Red wavelength ' + str(red_centre) + " degrees", color='red')


    red = 639

    def closest(wave_nm, red):
        return wave_nm[min(range(len(wave_nm)), key=lambda i: abs(wave_nm[i] - red))]

    wave_red_match = closest(wave_nm, red)

    for i in range(len(wave_nm)):
        if wave_red_match == wave_nm[i]:
            wave_red_match_index = i

    no_lambda_red = no_all[wave_red_match_index]
    neff_lambda_red = ne_all[wave_red_match_index]
    print('no red',no_lambda_red)
    print('neff red',neff_lambda_red)
    delta_red = neff_lambda_red - no_lambda_red
    pretilt_red_deg = math.asin(
        math.sin((red_centre) * math.pi / 180) / (no_lambda_red + neff_lambda_red)) * 180 / math.pi

    plt.plot(tilt_angle, tilt_blue, color='b')  # green
    plt.axvline(blue_centre, color='k')
    plt.ylabel('Intensity')
    plt.xlabel('Tilt')
    #plt.title("Blue wavelength " + str(blue_centre) + " degrees", color='blue')
    plt.title('all graphs overlayed with centers')
    plt.show()

    blue = 439

    def closest(wave_nm, blue):
        return wave_nm[min(range(len(wave_nm)), key=lambda i: abs(wave_nm[i] - blue))]

    wave_blue_match = closest(wave_nm, blue)

    for i in range(len(wave_nm)):
        if wave_blue_match == wave_nm[i]:
            wave_blue_match_index = i

    no_lambda_blue = no_all[wave_blue_match_index]
    neff_lambda_blue = ne_all[wave_blue_match_index]
    print('no blue',no_lambda_blue)
    print('neff blue',neff_lambda_blue)
    delta_blue = neff_lambda_blue - no_lambda_blue
    pretilt_blue_deg = math.asin(
        math.sin((blue_centre) * math.pi / 180) / (no_lambda_red + neff_lambda_red)) * 180 / math.pi

    print('pre-tilt green' + '=' + str(round(pretilt_green_deg, 3)))
    print('pre-tilt red' + '=' + str(round(pretilt_red_deg, 3)))
    print('pre-tilt blue' + '=' + str(round(pretilt_blue_deg, 3)))

    avg_pretilt = (pretilt_green_deg + pretilt_red_deg + pretilt_blue_deg) / 3
    print('avg pre-tilt' + '=' + str(round(avg_pretilt, 1)))



    ##################
    avg_pretilt = 1.5  # 0 for testing

    neff_all = []
    for j in range(len(wave_um)):  # j=0
        neff = 1 / (math.sqrt(((math.sin(avg_pretilt * math.pi / 180) / no_all[j]) ** 2) + (
                    (math.cos(avg_pretilt * math.pi / 180) / ne_all[j]) ** 2)))
        neff_all.append(neff)

    delta_neff_all = []
    for i in range(len(neff_all)):
        delta_neff_all.append(neff_all[i] - no_all[i])

    # neff_delta_all=[]
    delta_n_all = []
    for k in range(len(wave_um)):
        # neff_delta=neff_all[k]-no_all[k]
        delta_n = ne_all[k] - no_all[k]

        # neff_delta_all.append(neff_delta)
        delta_n_all.append(delta_n)

    return (avg_pretilt, delta_neff_all, wave_nm, no_all, neff_all, wave_nm)


#main_pretilt()

#avg_pretilt = main_pretilt()[0]

#delta_neff_all = main_pretilt()[1]

#wave_nm = main_pretilt()[2]

#no_all = main_pretilt()[3]

#neff_all = main_pretilt()[4]

#wave_nm = main_pretilt()[5]

avg_pretilt, delta_neff_all, wave_nm, no_all, neff_all, wave_nm = main_pretilt()

#print(delta_neff_all)
#print(wave_nm)
#print(no_all)
#print(neff_all)




