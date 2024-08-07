# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:27:25 2018

@author: Peter Clark

Tested 25/10/2023 at Version 0.6.0
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dask

import subfilter
import subfilter.subfilter as sf
import subfilter.filters as filt
import monc_utils.data_utils.deformation as defm
import monc_utils.data_utils.cloud_monc as cldm
import monc_utils.thermodynamics.thermodynamics as th


from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.dataout import save_field
from monc_utils.io.datain import (configure_model_resolution,
                                 get_data_on_grid,
                                 )
from subfilter.utils.default_variables import (get_default_variable_list,
                                      get_default_variable_pair_list)


import monc_utils
from loguru import logger

logger.remove()
logger.add(sys.stderr, 
           format = "<c>{time:HH:mm:ss.SS}</c>"\
                  + " | <level>{level:<8}</level>"\
                  + " | <green>{function:<22}</green> : {message}", 
           colorize=True, 
           level="INFO")
    
logger.enable("subfilter")
logger.enable("monc_utils")

logger.info("Logging 'INFO' or higher messages only")
test_case = 0
# run_quad_fields = True
run_quad_fields = False
run_deformation_fields = True
# run_deformation_fields = False
# run_cloud_fields = True
run_cloud_fields = False
# run_ri = True
run_ri = False
override = True

plot_type = '.png'


def main():
    """Top level code, a bit of a mess."""
    # fileroot = 'C:/Users/paclk/OneDrive - University of Reading/'
    # fileroot = 'C:/Users/xm904103/OneDrive - University of Reading/'
    fileroot = 'D:/Data/'
    if test_case == 0:
        config_file = 'config_test_case_0.yaml'
        indir = fileroot + 'ug_project_data/'
        odir = fileroot + 'ug_project_data/'
        file = 'diagnostics_3d_ts_21600.nc'
        ref_file = 'diagnostics_ts_21600.nc'
    elif test_case == 1:
        config_file = 'config_test_case_1.yaml'
        indir = fileroot + 'traj_data/CBL/'
        odir = fileroot + 'traj_data/CBL/'
        file = 'diagnostics_3d_ts_13200.nc'
        ref_file = None
    options, update_config = sf.subfilter_options(config_file)

    # var_list = ["q_cloud_liquid_mass", "cloud_fraction"]
    # var_list = ["q_total", "th_L"]
    # var_list = ["th", "dbydx(th)"] 
    var_list = ["th"] 
    # var_list = None

#dir = fileroot + 'Git/python/Subfilter/test_data/BOMEX/'
#odir = fileroot + 'Git/python/Subfilter/test_data/BOMEX/'


#file = 'diagnostics_ts_18000.0.nc'
#ref_file = 'diagnostics_ts_18000.0.nc'

    fname = 'test_rewrite_vn6'

    odir = odir + fname +'_'+ options['FFT_type']+'/'
    os.makedirs(odir, exist_ok = True)

    plot_dir = odir + 'plots/'
    os.makedirs(plot_dir, exist_ok = True)

    # Avoid accidental large chunks and read dask_chunks
    if not monc_utils.global_config['no_dask']:
        dask.config.set({"array.slicing.split_large_chunks": True})
        dask_chunks = monc_utils.global_config['dask_chunks']
        
    # monc_utils.set_global_config({'output_precision':"float32"})
        
    monc_utils.global_config['output_precision'] = "float32"

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

    filter_name = update_config['filters']['filter_name']
#    sigma_list = update_config['filters']['sigma_list']
    sigma_list = [500.0]
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
#            filter_id = 'filter_rm{:02d}'.format(i)
#            filter_id = 'filter_rm{:02d}n'.format(i)
            filter_id = 'filter_rm{:02d}v3'.format(i)
#            width = int(np.round( sigma/dx * np.pi * 2.0 / 3.0)+1)
#            width = int(np.round( sigma/dx * 2.0 * np.sqrt(3.0))+1)
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
                                       options, twod_filter, override=True)

        if exists :
            print('Filtered data file exists' )
            print("Variables in filtered dataset.")
            print(filtered_data['ds'].variables)
            field_list = get_default_variable_list()
        else :
            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                derived_data, filtered_data,
                                                options,
                                                twod_filter, var_list=var_list,
                                                grid = opgrid)

            if run_quad_fields:
                quad_field_list = sf.filter_variable_pair_list(dataset,
                                                ref_dataset,
                                                derived_data, filtered_data,
                                                options,
                                                twod_filter, var_list=None,
                                                grid = opgrid)


            if run_deformation_fields:
                deformation_r, deformation_s = sf.filtered_deformation(
                                                dataset,
                                                ref_dataset,
                                                derived_data,
                                                filtered_data, options,
                                                twod_filter, grid=opgrid)


                Sn_ij_r, mod_Sn_r = defm.shear(deformation_r)
                Sn_ij_r.name = 'f('+Sn_ij_r.name + ')_r'
                Sn_ij_r = save_field(filtered_data, Sn_ij_r)
                mod_Sn_r.name = 'f('+mod_Sn_r.name + ')_r'
                mod_Sn_r = save_field(filtered_data, mod_Sn_r)

                Sn_ij_s, mod_Sn_s = defm.shear(deformation_s)
                Sn_ij_s.name = 'f('+Sn_ij_s.name + ')_s'
                Sn_ij_s = save_field(filtered_data, Sn_ij_s)
                mod_Sn_s.name = 'f('+mod_Sn_s.name + ')_s'
                mod_Sn_s = save_field(filtered_data, mod_Sn_s)

                # S_ij_r, mod_S_r = defm.shear(deformation_r, no_trace = False)
                # S_ij_r.name = 'f('+S_ij_r.name + ')_r'
                # S_ij_r = sf.save_field(filtered_data, S_ij_r)
                # mod_S_r.name = 'f('+mod_S_r.name + ')_r'
                # mod_S_r = sf.save_field(filtered_data, mod_S_r)

                # S_ij_s, mod_S_s = defm.shear(deformation_s, no_trace = False)
                # S_ij_s.name = 'f('+S_ij_s.name + ')_s'
                # S_ij_s = sf.save_field(filtered_data, S_ij_s)
                # mod_S_s.name = 'f('+mod_S_s.name + ')_s'
                # mod_S_s = sf.save_field(filtered_data, mod_S_s)

                print(Sn_ij_r)

                v_r = defm.vorticity(deformation_r)
                v_r.name = 'f('+v_r.name + ')_r'
                v_r = save_field(filtered_data, v_r)

                print(v_r)

        if run_cloud_fields:
            options['save_all'] = 'No'
            th_ref = get_data_on_grid(dataset, ref_dataset, 'thref', 
                                      derived_dataset=derived_data, 
                                      options=options,
                                      grid=opgrid)
            p_ref  = get_data_on_grid(dataset, ref_dataset, 'pref', 
                                      derived_dataset=derived_data, 
                                      options=options,
                                      grid=opgrid)
            parms = th.cloud_params_monc(th_ref, p_ref)
            s_qt_qt = filtered_data['ds']["s(q_total,q_total)_on_w"]
            s_thL_qt = filtered_data['ds']["s(th_L,q_total)_on_w"]
            s_thL_thL = filtered_data['ds']["s(th_L,th_L)_on_w"]
            sigma_s = cldm.sigma_s(s_qt_qt, s_thL_qt, s_thL_thL, parms)
            sigma_s = save_field(filtered_data, sigma_s)

        if run_ri:

            sf.filter_variable_list(dataset, ref_dataset,
                                    derived_data, filtered_data,
                                    options,
                                    twod_filter,
                                    var_list=['moist_dbdz'],
                                    grid = opgrid)
            dbdz_r = filtered_data["ds"][f"f(moist_dbdz_on_{opgrid})_r"]

            ri_r = cldm.richardson(mod_Sn_r, dbdz_r)
            ri_r.name = 'moist_Ri_r'
            ri_r = save_field(filtered_data, ri_r)


        print('--------------------------------------')

        print(filtered_data)

        print('--------------------------------------')

        filtered_data['ds'].close()


        filtered_data['ds'] = xr.open_dataset(filtered_data['file'])

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

        if run_deformation_fields:

            print("Plotting mod_Sn")
            plot_shear(mod_Sn_r, mod_Sn_s, twod_filter, plot_dir, ilev, iy,
                        no_trace = True)
        #     print("Plotting mod_S")
        #     plot_shear(mod_S_r, mod_S_s, z, twod_filter, plot_dir, ilev, iy,
        #                 no_trace = False)
        if run_cloud_fields:
            for field in [sigma_s]:
                print(f"Plotting {field.name}")
                plot_single(field, filtered_data, twod_filter, plot_dir,
                            ilev, iy,
                            grid=opgrid)

        if run_ri:

            print(f"Plotting {dbdz_r.name}")
            plot_single(dbdz_r, filtered_data, twod_filter, plot_dir,
                        ilev, iy,
                        grid=opgrid)
            print(f"Plotting {ri_r.name}")
            plot_single(ri_r, filtered_data, twod_filter, plot_dir,
                        ilev, iy,
                        grid=opgrid, levels=np.linspace(-1,1,41))

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

            fn = plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type
            plt.savefig(fn)
            plt.close()
        else :

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            meanfield= var_r.isel(time=it).mean(dim=(xvar, yvar))
            pltdatxy = (var_r.isel(time=it)-meanfield)
            Cs1 = pltdatxy.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = var_s.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            meanfield= var_r.isel(time=it).mean(dim=(xvar))
            pltdatxz = (var_r.isel(time=it)-meanfield)
            Cs3 = pltdatxz.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

            Cs4 = var_s.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdatxy.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = var_s.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])

            plt.tight_layout()

            fn = plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type
            print(fn)
            plt.savefig(fn)

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

def plot_shear(var_r, var_s, twod_filter, plot_dir, ilev, iy, no_trace = True):
    var_name = var_r.name
    if no_trace : var_name = var_name+'n'

    [iix, iiy, iiz] = get_string_index(var_s.dims, ['x', 'y', 'z'])
    [xvar, yvar, zvar] = [list(var_s.dims)[i] for i in [iix, iiy, iiz]]

    for it, time in enumerate(var_r.coords['time']):
        print(f'it:{it}')

        if twod_filter.attributes['filter_type']=='domain' :

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = var_r.isel({'time':it, zvar:slice(1,None)}).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            pltdat = var_r.isel(time=it)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = var_s.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

#             axa[1,0].set_title(r'%s$^r$ pert at iy %03d'%(var_name,iy))
            Cs4 = var_s.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = var_s.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])
            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
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
