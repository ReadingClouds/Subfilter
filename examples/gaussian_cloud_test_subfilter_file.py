# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:27:25 2018

@author: Peter Clark
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask
#from dask.distributed import Client
#from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

import subfilter.subfilter as sf
import subfilter.filters as filt


from subfilter.utils.string_utils import get_string_index
from subfilter.io.dataout import save_field
from subfilter.io.datain import (configure_model_resolution,
                                 get_data_on_grid,
                                 )
from subfilter.utils.default_variables import (get_default_variable_list,
                                      get_default_variable_pair_list)


import subfilter
test_case = 0
run_quad_fields = True
#run_quad_fields = False
override = True

plot_type = '.png'


def main():
    """Top level code, a bit of a mess."""
    if test_case == 0:
        config_file = 'config_test_case_0.yaml'
        indir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
        odir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
        file = 'diagnostics_3d_ts_21600.nc'
        ref_file = 'diagnostics_ts_21600.nc'
    elif test_case == 1:
        config_file = 'config_test_case_1.yaml'
        indir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
        odir = 'C:/Users/paclk/OneDrive - University of Reading/traj_data/CBL/'
        file = 'diagnostics_3d_ts_13200.nc'
        ref_file = None
    options, update_config = sf.subfilter_options(config_file)


#dir = 'C:/Users/paclk/OneDrive - University of Reading/Git/python/Subfilter/test_data/BOMEX/'
#odir = 'C:/Users/paclk/OneDrive - University of Reading/Git/python/Subfilter/test_data/BOMEX/'


#file = 'diagnostics_ts_18000.0.nc'
#ref_file = 'diagnostics_ts_18000.0.nc'

    fname = 'gaussian_cloud'

    field_list = [
        "w",
        "th",
        "th_L",
        "q_vapour",
        "q_cloud_liquid_mass",
        "q_total",
        "cloud_fraction"
        ]
    quad_field_list = [
                ["w","th"],
                ["w","th_L"],
                ["w","q_total"],
                ["th_L","th_L"],
                ["th_L","q_total"],
                ["q_total","q_total"],
              ]


    odir = odir + fname +'_'+ options['FFT_type']+'/'
    os.makedirs(odir, exist_ok = True)

    plot_dir = odir + 'plots/'
    os.makedirs(plot_dir, exist_ok = True)

    # Avoid accidental large chunks and read dask_chunks
    if not subfilter.global_config['no_dask']:
        dask.config.set({"array.slicing.split_large_chunks": True})
        dask_chunks = subfilter.global_config['dask_chunks']

    subfilter.global_config['test_level'] = 2
    # Read data
    dataset = xr.open_dataset(indir+file, chunks=dask_chunks)

    print(dataset)

    if ref_file is not None:
        ref_dataset = xr.open_dataset(indir+ref_file)
    else:
        ref_dataset = None

    # Get model resolution values
    dx, dy, options = configure_model_resolution(dataset, options)

    [itime, iix, iiy, iiz] = get_string_index(dataset.dims, ['time', 'x', 'y', 'z'])
    [timevar, xvar, yvar, zvar] = [list(dataset.dims)[i] for i in [itime, iix, iiy, iiz]]

    npoints = dataset.dims[xvar]

# For plotting
#    ilev = 15
    ilev = 40
#    iy = 40
    iy = 95

    opgrid = 'p'

    derived_data, exists = \
        sf.setup_derived_data_file( indir+file, odir, fname,
                                   options, override=override)
    if exists :
        print('Derived data file exists' )
        print("Variables in derived dataset.")
        print(derived_data['ds'].variables)


# Now create list of filter definitions.

#    filter_name = update_config['filters']['filter_name']
#    sigma_list = update_config['filters']['sigma_list']
    filter_name = "gaussian"
    sigma_list = [100.0, 200.0, 400.0, 800.0]
    filter_list = list([])

    for i,sigma in enumerate(sigma_list):
        if filter_name == 'gaussian':
            filter_id = 'filter_ga{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                      filter_name,
                                      npoints=npoints,
                                      sigma=sigma,
                                      delta_x=dx)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                      filter_name,
                                      npoints=npoints,
                                      wavenumber=np.pi/(2*sigma),
                                      delta_x=dx)
        elif filter_name == 'circular_wave_cutoff':
            filter_id = 'filter_cwc{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                      filter_name,
                                      npoints=npoints,
                                      wavenumber=np.pi/(2*sigma),
                                      delta_x=dx)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}v3'.format(i)
            width = int(np.round( sigma/dx *  np.sqrt(2.0 *np.pi))+1)
            twod_filter = filt.Filter(filter_id,
                                      filter_name,
                                      npoints=npoints,
                                      width=width,
                                      delta_x=dx)

        print(twod_filter)
        filter_list.append(twod_filter)

    # Add whole domain filter
    filter_name = 'domain'
    filter_id = 'filter_do'
    twod_filter = filt.Filter(filter_id, filter_name, delta_x=dx)
    print(twod_filter)
    filter_list.append(twod_filter)

# Process data with each filter.

    for twod_filter in filter_list:

        print("Processing using filter: ")
        print(twod_filter)

        filtered_data, exists = \
            sf.setup_filtered_data_file( indir+file, odir, fname,
                                       options, twod_filter, override=override)

        if exists :
            print('Filtered data file exists' )
            print(filtered_data['ds'])

        field_list = sf.filter_variable_list(dataset, ref_dataset,
                                            derived_data, filtered_data,
                                            options,
                                            twod_filter,
                                            var_list=field_list,
                                            grid = opgrid)

        if run_quad_fields:
            quad_field_list = sf.filter_variable_pair_list(dataset,
                                            ref_dataset,
                                            derived_data, filtered_data,
                                            options,
                                            twod_filter,
                                            var_list=quad_field_list,
                                            grid = opgrid)

        print('--------------------------------------')

        print(filtered_data)

        print('--------------------------------------')

        filtered_data['ds'].close()

        if twod_filter.attributes['filter_type'] != 'domain' :
            fig1 = plt.figure(1)
            plt.contourf(twod_filter.data,20)
            plt.savefig(plot_dir+'Filter_'+\
                        twod_filter.id+plot_type)
            plt.close()

            fig2 = plt.figure(2)
            plt.plot(twod_filter.data[np.shape(twod_filter.data)[0]//2,:])
            plt.savefig(plot_dir+'Filter_y_xsect_'+\
                        twod_filter.id+plot_type)
            plt.close()

        for field in field_list:
            print(f"Plotting {field}")
            plot_field(field, filtered_data, twod_filter, plot_dir,
                       ilev, iy,
                       grid=opgrid)

        if run_quad_fields:
            for field in quad_field_list :
                print(f"Plotting {field}")
                plot_quad_field(field, filtered_data, twod_filter, plot_dir,
                                ilev, iy,
                                grid=opgrid)

        filtered_data['ds'].close()
    print('--------------------------------------')

    print(derived_data)

    print('--------------------------------------')
    derived_data['ds'].close()
    dataset.close()

def plot_field(var_name, filtered_data, twod_filter, plot_dir,
               ilev, iy, grid='p'):

    var_r = filtered_data['ds'][f"f({var_name}_on_{grid})_r"]
    var_s = filtered_data['ds'][f"f({var_name}_on_{grid})_s"]

    [iix, iiy, iiz] =  get_string_index(var_s.dims, ['x', 'y', 'z'])
    [xvar, yvar, zvar] = [list(var_s.dims)[i] for i in [iix, iiy, iiz]]

    for it, time in enumerate(var_r.coords['time']):

        print(f'it:{it}')

        if twod_filter.attributes['filter_type']=='domain' :

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = var_r.isel(time=it).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            meanfield= var_r.isel(time=it).mean(dim=(xvar, yvar))
            pltdat = var_r.isel(time=it)#-meanfield)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

#            axa[0,0].set_title(r'%s$^r$ pert level %03d'%(var_name,ilev))

            Cs2 = var_s.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)
#             axa[0,1].set_title(r'%s$^s$ level %03d'%(var_name,ilev))

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

#             axa[1,0].set_title(r'%s$^r$ pert at iy %03d'%(var_name,iy))
            Cs4 = var_s.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)
#             axa[1,1].set_title(r'%s$^s$ at iy %03d'%(var_name,iy))

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = var_s.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])
#             x=(np.arange(0,var_r.shape[2])-0.5*var_r.shape[2])*0.1
            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)

            plt.close()

    return

def plot_quad_field(var_name, filtered_data, twod_filter, plot_dir,
                    ilev, iy, grid='p'):

    v1 = var_name[0]
    v2 = var_name[1]

    v1_r = filtered_data['ds'][f"f({v1}_on_{grid})_r"]
    v2_r = filtered_data['ds'][f"f({v2}_on_{grid})_r"]

#    print(v1,v2)
    s_v1v2 = filtered_data['ds'][f"s({v1},{v2})_on_{grid}"]
#    print(s_v1v2)

    [iix, iiy, iiz] = get_string_index(s_v1v2.dims, ['x', 'y', 'z'])
    if iix is not None:
        xvar = s_v1v2.dims[iix]
        yvar = s_v1v2.dims[iiy]
    zvar = s_v1v2.dims[iiz]

    for it, time in enumerate(s_v1v2.coords['time']):

        print(f'it:{it}')


        if twod_filter.attributes['filter_type']=='domain' :

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = s_v1v2.isel(time=it).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :

            var_r = (v1_r.isel(time=it) - v1_r.isel(time=it).mean(dim=(xvar, yvar))) * \
                    (v2_r.isel(time=it) - v2_r.isel(time=it).mean(dim=(xvar, yvar)))


            pltdat = var_r

#            pltdat = (var_r - var_r.mean(dim=(xvar, yvar)))

#            pltdat = v1_r.isel(time=it) * v2_r.isel(time=it)

            pltdat.name = 'f('+v1+')_r.'+'f('+v2+')_r'

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = s_v1v2.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

            Cs4 = s_v1v2.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = s_v1v2.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])

            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()

    return

def plot_single(var, filtered_data, twod_filter, plot_dir,
               ilev, iy, grid='p', levels = None):


    for it, time in enumerate(var.coords['time']):

        print(f'it:{it}')


        if twod_filter.attributes['filter_type']=='domain' :
            [iiz] =  get_string_index(var.dims, ['z'])
            zvar = var.dims[iiz]

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = var.isel({'time':it, zvar:slice(1,None)}).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var.name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :

            [iix, iiy, iiz] =  get_string_index(var.dims, ['x', 'y', 'z'])
            [xvar, yvar, zvar] = [list(var.dims)[i] for i in [iix, iiy, iiz]]
            meanfield= var.isel(time=it).mean(dim=(xvar, yvar))
            pltdat = (var.isel(time=it)-meanfield)

            nlevels = 40
            if levels is None:
                levels = 40
            plt.clf

            fig1, axa = plt.subplots(2,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=levels)
            Cs3 = pltdat.isel({yvar:iy, zvar:slice(1,None)}).plot.imshow(x=xvar, y=zvar, ax=axa[0,1], levels=levels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[1,0])
#            p2 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[1,1])

            plt.tight_layout()

            plt.savefig(plot_dir+var.name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)

            plt.close()

    return


if __name__ == "__main__":
    main()
