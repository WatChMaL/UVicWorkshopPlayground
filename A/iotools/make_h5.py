from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, h5py, tables, os
from utils import progress_bar

def make_h5(file_list,output_file):
    """
    Args: file_list ... list, files to be taken as input
          output_file ... string, name of an output h5 file
    """

    print('Will process',len(file_list),'files...')
    progress = display(progress_bar(0,len(file_list)),display_id=True)
    # Create output file
    FILTERS   = tables.Filters(complib='zlib', complevel=5)
    output    = tables.open_file(output_file,mode='w',filters=FILTERS)
    out_ndarray = {}
    out_1darray = {}
    label     = None

    # Loop over files, read data & store
    # For labels, since it's a small 1D array, we store all at the end
    # For event_data, they will be appended file-by-file
    for file_index,file_name in enumerate(file_list):
        # Open file
        f = np.load(file_name)
        if ( file_index==0 or (file_index+1)%100==0) :
            progress.update( progress_bar(file_index+1,len(file_list)," Processing file"))
        for key in f.keys():
            data_shape = f[key].shape
            if len(data_shape) < 2:
                print(key)
                if not key in out_1darray: out_1darray[key]=f[key].astype(np.float32)
                else: out_1darray[key] = np.hstack([out_1darray[key],f[key].astype(np.float32)])
            else:
                if not key in out_ndarray:
                    chunk_shape = [1] + list(data_shape[1:])
                    data_shape  = [0] + list(data_shape[1:])
                    out_ndarray[key] = output.create_earray(output.root,key,tables.Float32Atom(),chunkshape=chunk_shape,shape=data_shape)
                out_ndarray[key].append(f[key].astype(np.float32))

        sys.stdout.write('Progress: %1.3f\r' % (float(file_index+1)/len(file_list)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    # Create chunked-array to store 1D arrays
    for key in out_1darray:
        data = out_1darray[key]
        out_data = output.create_carray(output.root, key, tables.Float32Atom(), shape=data.shape)
        out_data[:] = data

    # Report what's stored
    print('\nFinished!\n')
    # Close output file
    output.close()

    import h5py
    f=h5py.File(output_file,mode='r')
    print('Stored keys:',f.keys())
    for key in f.keys():
        print('    %s ... shape %s' % (key,f[key].shape))






