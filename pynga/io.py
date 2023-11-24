from os import listdir
from os import mkdir
from os import path
import os
import datetime
import re
import numpy as np
import glob
from mpi4py import MPI

str_short       = 8
str_medium      = 64

class case:
    '''Handle data in a NGA-compressible case'''

    # Members
    comm                = None
    case_path           = '.'
    config_path         = '.'
    input_path          = '.'
    data_init_path      = '.'
    simulation_name     = None
    nx                  = None
    ny                  = None
    nz                  = None
    nover               = None
    x                   = None
    y                   = None
    z                   = None
    Lx                  = 0.
    Ly                  = 0.
    Lz                  = 0.
    icyl                = 0
    is_per              = np.zeros((3)) 
    npx                 = 4
    npy                 = 1
    npz                 = 1
    nxo         = nyo         = nzo         = None
    nxo_        = nyo_        = nzo_        = None
    # Fortran index starting from 1
    imino       = jmino       = kmino       = None
    imaxo       = jmaxo       = kmaxo       = None
    imino_      = jmino_      = kmino_      = None
    imaxo_      = jmaxo_      = kmaxo_      = None
    # Pythond index starting from 0
    iMino       = jMino       = kMino       = None
    iMaxo       = jMaxo       = kMaxo       = None
    iMino_      = jMino_      = kMino_      = None
    iMaxo_      = jMaxo_      = kMaxo_      = None
    # Slices
    slx         = sly         = slz         = None
    slx0 = slx1 = sly0 = sly1 = slz0 = slz1 = None    
    # Communicators
    cart3d      = None
    cart2d_yz   = cart2d_xz   = cart2d_xy   = None
    cart1d_x    = cart1d_y    = cart1d_z    = None
    coord3d     = None
    irank_x     = irank_y     = irank_z     = None
    iproc       = jproc       = kproc       = None
    xpes        = ypes        = zpes        = None

    # Views
    subType3D           = None
    subType2D_yz        = None      # Plane normal to x=const
    subType2D_xz        = None      # Plane normal to y=const
    subType2D_xy        = None      # Plane normal to z=const


    # TBD: Constructor - initialize from (nx,ny,nz) and (npx,npy,npz)

    # Constructor - initialize from data.init, input
    def __init__(self, comm, case_path, config="data.init", input="input", data_init="data.init", fullpath = True, nover = 1):

        # Initiatialize paths
        self.file_path = os.getcwd()
        if not fullpath:
            self.case_path = os.path.abspath(case_path)
        else:
            self.case_path = case_path
        self.config_path     = os.path.join(case_path, config)
        self.input_path      = os.path.join(case_path, input) 
        self.data_init_path  = os.path.join(case_path, data_init)
        self.nover = nover

        # Read config file
        amode = MPI.MODE_RDONLY
        fh = MPI.File.Open(comm, self.config_path, amode)
        # simulation name
        buffer_str = np.empty(1, dtype="S64")
        fh.Read_all([buffer_str, 64, MPI.CHARACTER])
        self.simulation_name = buffer_str[0].decode('ascii')
        # icyl, is_per[0:3], nx, ny, nz
        item_count = 7
        buffer_int = np.empty(item_count, dtype='i')
        fh.Read_all(buffer_int)
        self.icyl = buffer_int[0]
        self.is_per = buffer_int[1:4]
        self.nx = buffer_int[4]
        self.ny = buffer_int[5]
        self.nz = buffer_int[6]
        # x, y, z coordinates
        item_count = self.nx+1
        buffer_double = np.empty(item_count, dtype='d')
        fh.Read_all(buffer_double)
        self.x = np.array(buffer_double)
        item_count = self.ny+1
        buffer_double = np.empty(item_count, dtype='d')
        fh.Read_all(buffer_double)
        self.y = np.array(buffer_double)
        item_count = self.nz+1
        buffer_double = np.empty(item_count, dtype='d')
        fh.Read_all(buffer_double)
        self.z = np.array(buffer_double)
        self.Lx = self.x[-1] - self.x[0]
        self.Ly = self.y[-1] - self.y[0]
        self.Lz = self.z[-1] - self.z[0]
        fh.Close() 

        # Initialize parallel topology
        self.npes = comm.Get_size()
        self.myid = comm.Get_rank()
        self.mypn = MPI.Get_processor_name()    
        # 3D
        self.cart3d     = comm.Create_cart(dims = [self.npx,self.npy,self.npz],periods = self.is_per,reorder=False)
        self.coord3d    = self.cart3d.Get_coords(self.myid)
        self.iproc      = self.coord3d[0] + 1
        self.jproc      = self.coord3d[1] + 1
        self.kproc      = self.coord3d[2] + 1
        # 2D
        self.cart2d_xy  = self.cart3d.Sub(remain_dims=[True, True, False])
        self.cart2d_xz  = self.cart3d.Sub(remain_dims=[True, False, True])
        self.cart2d_yz  = self.cart3d.Sub(remain_dims=[False, True, True])
        # 1D
        self.cart1d_x  = self.cart3d.Sub(remain_dims=[True, False, False])
        self.xpes      = self.cart1d_x.Get_rank() 
        self.irank_x   = self.cart1d_x.Get_rank() + 1 # = coord3d[0]+1
        self.cart1d_y  = self.cart3d.Sub(remain_dims=[False, True, False])
        self.ypes      = self.cart1d_y.Get_rank() 
        self.irank_y   = self.cart1d_y.Get_rank() + 1 # = coord3d[1]+1
        self.cart1d_z  = self.cart3d.Sub(remain_dims=[False, False, True])
        self.zpes      = self.cart1d_z.Get_rank() 
        self.irank_z   = self.cart1d_z.Get_rank() + 1 # = coord3d[2]+1

        # Initialize global index
        self.nxo = self.nx + 2*self.nover
        self.nyo = self.ny + 2*self.nover
        self.nzo = self.nz + 2*self.nover
        self.imino = 1; self.imin  = self.imino + self.nover; self.imax  = self.imin  + self.nx - 1; self.imaxo = self.imax  + self.nover
        self.jmino = 1; self.jmin  = self.jmino + self.nover; self.jmax  = self.jmin  + self.ny - 1; self.jmaxo = self.jmax  + self.nover
        self.kmino = 1; self.kmin  = self.kmino + self.nover; self.kmax  = self.kmin  + self.nz - 1; self.kmaxo = self.kmax  + self.nover

        # Initialize local index
        q = int(self.nx/self.npx); r = np.mod(self.nx,self.npx)
        if (self.iproc<=r):
            self.nx_   = q+1
            self.imin_ = self.imin + (self.iproc-1)*(q+1)
        else:
            self.nx_   = q
            self.imin_ = self.imin + r*(q+1) + (self.iproc-r-1)*q
        q = int(self.ny/self.npy); r = np.mod(self.ny,self.npy)
        if (self.jproc<=r):
            self.ny_   = q+1
            self.jmin_ = self.jmin + (self.jproc-1)*(q+1)
        else:
            self.ny_   = q
            self.jmin_ = self.jmin + r*(q+1) + (self.jproc-r-1)*q
        q = int(self.nz/self.npz); r = np.mod(self.nz,self.npz)
        if (self.kproc<=r):
            self.nz_   = q+1
            self.kmin_ = self.kmin + (self.kproc-1)*(q+1)
        else:
            self.nz_   = q
            self.kmin_ = self.kmin + r*(q+1) + (self.kproc-r-1)*q
        self.nxo_   = self.nx_ + 2*self.nover; self.imax_  = self.imin_ + self.nx_ - 1; self.imino_ = self.imin_ - self.nover; self.imaxo_ = self.imax_ + self.nover
        self.nyo_   = self.ny_ + 2*self.nover; self.jmax_  = self.jmin_ + self.ny_ - 1; self.jmino_ = self.jmin_ - self.nover; self.jmaxo_ = self.jmax_ + self.nover
        self.nzo_   = self.nz_ + 2*self.nover; self.kmax_  = self.kmin_ + self.nz_ - 1; self.kmino_ = self.kmin_ - self.nover; self.kmaxo_ = self.kmax_ + self.nover 
        

        # Initialize index of python - starting from 0
        self.iMino_ = 0;    self.iMaxo_ = self.nxo_ - 1
        self.jMino_ = 0;    self.jMaxo_ = self.nyo_ - 1
        self.kMino_ = 0;    self.kMaxo_ = self.nzo_ - 1
        self.iMin_  = nover; self.iMax_ = self.iMaxo_ - nover
        self.jMin_  = nover; self.jMax_ = self.jMaxo_ - nover
        self.kMin_  = nover; self.kMax_ = self.kMaxo_ - nover

        self.slx  = slice(self.iMin_,   self.iMax_+1,   1)
        self.slx0 = slice(self.iMino_,  self.iMin_,     1)
        self.slx1 = slice(self.iMax_+1, self.iMaxo_+1,  1)
        self.sly  = slice(self.jMin_,   self.jMax_+1,   1)
        self.sly0 = slice(self.jMino_,  self.jMin_,     1)
        self.sly1 = slice(self.jMax_+1, self.jMaxo_+1,  1)
        self.slz  = slice(self.kMin_,   self.kMax_+1,   1)
        self.slz0 = slice(self.kMino_,  self.kMin_,     1)
        self.slz1 = slice(self.kMax_+1, self.kMaxo_+1,  1)

        # Initialize paralle views
        # xyz
        gsizes = [self.nx, self.ny, self.nz]
        lsizes = [self.nx_,self.ny_,self.nz_]    
        starts = [self.imin_-self.imin, self.jmin_-self.jmin, self.kmin_-self.kmin]
        self.subType3D = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order = MPI.ORDER_F)
        self.subType3D.Commit()
        # x-yz
        gsizes    = [self.ny,  self.nz]
        lsizes    = [self.ny_, self.nz_]
        starts = [self.jmin_-self.nover-1, self.kmin_-self.nover-1] # -1 is for python index starting from 0
        subType2D_yz = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order = MPI.ORDER_F)
        subType2D_yz.Commit()
        # y-xz
        gsizes    = [self.nx,  self.nz]
        lsizes    = [self.nx_, self.nz_]
        starts = [self.imin_-self.nover-1, self.kmin_-self.nover-1] # -1 is for python index starting from 0
        subType2D_xz = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order = MPI.ORDER_F)
        subType2D_xz.Commit()
        # z-xy
        gsizes    = [self.nx,  self.ny]
        lsizes    = [self.nx_, self.ny_]
        starts = [self.imin_-self.nover-1, self.jmin_-self.nover-1] # -1 is for python index starting from 0
        subType2D_xy = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order = MPI.ORDER_F)
        subType2D_xy.Commit()

    # Parallel NGA data reader
    def mpi_read_data(self, comm, dat_name):        

        res = {}
        amode = MPI.MODE_RDONLY
        fh = MPI.File.Open(comm, dat_name, amode)

        # nx, ny, nz, nvar
        item_count = 4
        buffer_int = np.empty(item_count, dtype='i')
        fh.Read_all(buffer_int)
        res["nx"]   = buffer_int[0]
        res["ny"]   = buffer_int[1]
        res["nz"]   = buffer_int[2]
        res["nvar"] = buffer_int[3]
        # dt and time
        item_acount = 2
        buffer_double = np.empty(item_count, dtype=np.double)
        fh.Read_all([buffer_double, 2, MPI.DOUBLE_PRECISION])
        res["dt"] = buffer_double[0]
        res["time"] = buffer_double[1]

        fnames = []
        for i in range(0, res["nvar"]):
          buffer_str = np.empty(1, dtype="S8")
          fh.Read_all([buffer_str, 8, MPI.CHARACTER])
          fnames.append(buffer_str[0].decode('ascii').rstrip())
        res["field_names"] = fnames

        nbytes_d = buffer_double.itemsize
        nbytes_i = buffer_int.itemsize
        nbytes_s = 8
        displacement = nbytes_i*4 + nbytes_d*2 + nbytes_s*res["nvar"]
        fh.Set_view(displacement, filetype=self.subType3D)
        buffer_double = np.zeros((self.nx_, self.ny_, self.nz_), dtype=np.double, order='F')
        for i in range(0, res["nvar"]):
          #buffer_double = np.empty((self.nx_*self.ny_*self.nz_), dtype="double")
          fh.Read_all(buffer_double)
          res[fnames[i]] = np.ones((self.nxo_, self.nyo_, self.nzo_), dtype=np.double, order='F')
          res[fnames[i]][self.slx, self.sly, self.slz] = buffer_double 

        fh.Close()
        return(res)

    def mpi_write_slice(self, field, fn, idir, index_F):
        is_write_id = False
        if idir == 1:
            if (self.imin_-self.nover<=index_F and self.imax_-self.nover>=index_F):
                is_write_id = True
                tid = index_F + self.nover - self.imin_ # Python index starts from 0
        elif idir == 2:
            if (self.jmin_-self.nover<=index_F and self.jmax_-self.nover>=index_F):
                is_write_id = True
                tid = index_F + self.nover - self.jmin_ # Python index starts from 0
        elif idir == 3:
            if (self.kmin_-self.nover<=index_F and self.kmax_-self.nover>=index_F):
                is_write_id = True
                tid = index_F + self.nover - self.kmin_ # Python index starts from 0
        if (is_write_id):

            dict_dir = {"x":1,"y":2,"z":3}
            bmode = MPI.MODE_CREATE | MPI.MODE_WRONLY
            fno = fn 
            if idir == 1:
                fs = MPI.File.Open(self.cart2d_yz, fno, bmode)
                buffer_double = np.ascontiguousarray(field[tid, self.sly, self.slz])
                lsizes    = [self.ny_,   self.nz_]
                gsizes    = [self.ny,    self.nz]
                istarts_g = [self.jmin_-self.nover-1, self.kmin_-self.nover-1]
                subType2D = MPI.DOUBLE.Create_subarray(gsizes, lsizes, istarts_g, order = MPI.ORDER_C)
                subType2D.Commit()
            elif idir == 2:
                fs = MPI.File.Open(self.cart2d_xz, fno, bmode)
                buffer_double = np.ascontiguousarray(field[self.slx, tid, self.slz])
                lsizes    = [self.nx_,   self.nz_]
                gsizes    = [self.nx,    self.nz]
                istarts_g = [self.imin_-self.nover-1, self.kmin_-self.nover-1]
                subType2D = MPI.DOUBLE.Create_subarray(gsizes, lsizes, istarts_g, order = MPI.ORDER_C)
                subType2D.Commit()
            elif idir == 3:
                fs = MPI.File.Open(self.cart2d_xy, fno, bmode)
                buffer_double = np.ascontiguousarray(field[self.slx, self.sly, tid])
                lsizes    = [self.nx_,   self.ny_]
                gsizes    = [self.nx,    self.ny]
                istarts_g = [self.imin_-self.nover-1, self.jmin_-self.nover-1]
                subType2D = MPI.DOUBLE.Create_subarray(gsizes, lsizes, istarts_g, order = MPI.ORDER_C)
                subType2D.Commit()  
            fs.Set_view(0, filetype=subType2D)
            fs.Write_all(buffer_double)
            fs.Close()

    # Parallel NGA data reader
    def get_slice_inner(self):        
      return self.slx, self.sly, self.slz

    def calc_mean_all(self, comm, field):
        suml = np.sum(field[self.slx,self.sly,self.slz])
        sumls = np.empty(self.npes, dtype=np.double)
        comm.Allgather([suml, MPI.DOUBLE], [sumls, MPI.DOUBLE])
        return np.sum(sumls)/(self.nx*self.ny*self.nz)

def get_keys_dat(s):
    ss = re.split('/', s)[-1]
    ss = re.split('_', ss)[-1]
    return float(ss)

def data_names(cpath, add_data_init=None):
    files = glob.glob(cpath + '/ufs:data_*E*')
    fn = sorted(files, key=get_keys_dat)
    if add_data_init != None:
        fn.insert(0, cpath+"/"+add_data_init)
    return fn

def timelist(cpath, add_data_init=None):
    fn  = data_names(cpath)
    tl = []
    for i in fn:
        tl.append(get_keys_dat(i))
    if add_data_init != None:
        tl.insert(0, 0.0)
    return tl




