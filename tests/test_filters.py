# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:10:18 2020

@author: paclk
"""
import numpy as np
import xarray as xr
import os
import matplotlib
import matplotlib.pyplot as plt
import time

import subfilter.subfilter as sf
import subfilter.filters as filt
import subfilter

options = {
#       'FFT_type': 'FFTconvolve',
#       'FFT_type': 'FFT',
       'FFT_type': 'RFFT',
#       'FFT_type': 'DIRECT',
          }

op_dir = 'test_figs/'

os.makedirs(op_dir, exist_ok=True)

wavelength_list = [3200.0, 1600.0, 800.0, 400.0, 200.0, 100.0]
#wavelength_list = [200.0]

filter_types = ['gaussian',
                'wave_cutoff',
                'circular_wave_cutoff',
                'running_mean',
                'one_two_one'
                ]

N = 128
dx = 100

x = np.linspace(0,(N-1)*dx,N )
y = x.copy()
xc, yc = np.meshgrid(x, y)

Lx  = (N)*dx/10
Ly  = (N)*dx/40
Ly2 = (N)*dx/5

kx = 2.0 * np.pi / Lx
ky = 2.0 * np.pi / Ly

Ltot = 2 * np.pi / np.sqrt(kx**2 +ky**2)

print(Ltot)

field = np.cos(kx * xc) * np.cos(ky * yc) \
      + np.cos(kx * xc) + np.cos(ky * yc) \
      + np.cos(2.0 * np.pi / Ly2 *yc) \
      + np.cos(np.pi / dx * xc) * np.cos(np.pi / dx * yc) \
      + np.cos(np.pi / dx * xc) + np.cos(np.pi / dx * yc)

var = xr.DataArray(field, name='test', coords=[x, y], dims=['x_p', 'y_p'])
print(var)

levs=np.linspace(-4,8,61)

xc /= 1000
yc /= 1000

x /= 1000
y /= 1000

id = 0
filter_list = list([])
for filter_name in filter_types:

    if filter_name == 'one_two_one':
        filter_id = 'filter_{:02d}'.format(id)
        twod_filter = filt.Filter(filter_id, filter_name,
                                  wavenumber= -1,
                                  sigma=dx, width=-1, npoints = N,
                                  delta_x=dx, set_fft=True)
        twod_filter.attributes['wavelength'] = 2*dx

#        print(twod_filter)
        filter_list.append(twod_filter)
        id += 1

    else:

        for i,wavelength in enumerate(wavelength_list):

            filter_id = 'filter_{:02d}'.format(id)

            if filter_name == 'running_mean':
                width = max(min(N-2,np.round( wavelength/dx+1)),1)
                sigma=-1
                wavenumber = -1
            elif filter_name == 'gaussian':
                sigma = wavelength/4.0
                width = -1
                wavenumber = -1
            else:
                width = -1
                wavenumber = 2*np.pi/wavelength
                sigma=-1

            twod_filter = filt.Filter(filter_id, filter_name,
                                      wavenumber= wavenumber,
                                      sigma=sigma, width=width, npoints = N,
                                      delta_x=dx, set_fft=True)
            twod_filter.attributes['wavelength'] = wavelength
#            print(twod_filter)
            filter_list.append(twod_filter)
        id += 1


print(filter_list)

for i, twod_filter in enumerate(filter_list):

    filter_name = twod_filter.attributes['filter_type']

    wavelength = twod_filter.attributes['wavelength']

#    sigma=twod_filter.attributes['sigma']

    nx, ny = np.shape(twod_filter.data)

    x = np.linspace(-(nx//2), (nx-1)//2, nx) * twod_filter.attributes['delta_x']/1000

    f, ax = plt.subplots(2,2,figsize=(12,12))

    c = ax[0,0].contour(x, x, twod_filter.data,20,colors='blue')
    ax[0,0].set_xlabel("x/km")
    ax[0,0].set_ylabel("y/km")

#    ax[0,0].legend()


    p1 = ax[0,1].plot(x, twod_filter.data[nx//2, :],label='Sampled')
    ax[0,1].set_xlabel("x/km")
    ax[0,1].legend()
    p2 = ax[1,0].plot(x, twod_filter.data[:, ny//2],label='Sampled')
    ax[1,0].set_xlabel("x/km")
    ax[1,0].legend()
    p3 = ax[1,1].plot(x*np.sqrt(2),twod_filter.data.diagonal(),label='Sampled')
    ax[1,1].set_xlabel("x-y/km")
    ax[1,1].legend()
    f.tight_layout()

    time_1 = time.perf_counter()
    (var_r, var_s) = sf.filtered_field_calc(var, options, twod_filter )
    time_2 = time.perf_counter()
    elapsed_time = time_2 - time_1
    print(f'Elapsed time = {elapsed_time}')
    # var_r = var_r['data']
    # var_s = var_s['data']

    E_var = np.mean(var * var)
    E_var_r = np.mean(var_r * var_r)
    E_var_s = np.mean(var_s * var_s)

    fc, axc = plt.subplots(3,3,figsize=(16,12))
    c_f = var.plot.imshow(x='x_p', y='y_p', ax = axc[0,0], levels=levs)
    #c_f = axc[0,0].imshow(var.values, levs)
    axc[0,0].set_xlabel("x/km")
    axc[0,0].set_ylabel("y/km")
    axc[0,0].set_title(r"Field E={:1.4f}".format(E_var.data))

#    c_r = axc[1,0].contourf(xc, yc, var_r, levs)
    c_r = var_r.plot.imshow(x='x_p', y='y_p', ax = axc[1,0], levels=levs)
    axc[1,0].set_xlabel("x/km")
    axc[1,0].set_ylabel("y/km")
    axc[1,0].set_title(r"Field$^r$ E={:1.4f}".format(E_var_r.data))

#    c_s = axc[2,0].contourf(xc, yc, var_s, levs)
    c_s = var_s.plot.imshow(x='x_p', y='y_p', ax = axc[2,0], levels=levs)
    axc[2,0].set_xlabel("x/km")
    axc[2,0].set_ylabel("y/km")
    axc[2,0].set_title(r"Field$^s$ E={:1.4f}".format(E_var_s.data))

    py_f = axc[0,1].plot(x, var[:, N//2])
    axc[0,1].set_xlabel("x/km")
    axc[0,1].set_title(r"$1/(k_y\lambda)$={:3.2f}".format(1/(ky*wavelength)))
    py_r = axc[1,1].plot(x, var_r[:, N//2])
    axc[1,1].set_xlabel("x/km")
    py_s = axc[2,1].plot(x, var_s[:, N//2])
    axc[2,1].set_xlabel("x/km")


    px_f = axc[0,2].plot(y, var[N//2,:])
    axc[0,2].set_title(r"$1/(k_x\lambda)$={:3.2f}".format(1/(kx*wavelength)))
    axc[0,2].set_xlabel("y/km")
    px_r = axc[1,2].plot(y, var_r[N//2,:])
    axc[1,2].set_xlabel("y/km")
    px_s = axc[2,2].plot(y, var_s[N//2,:])
    axc[2,2].set_xlabel("y/km")

    fc.tight_layout()

    f.savefig(f'{op_dir}{filter_name}_{wavelength:04.0f}_filter.png')
    fc.savefig(f'{op_dir}{filter_name}_{wavelength:04.0f}_test_data.png')

    plt.close(f)
    plt.close(fc)

#plt.show()
