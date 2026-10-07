"""CUDA and CPU Gaussian density kernels used by volinterior."""
from __future__ import annotations
import numpy as np
from .config import GridSpec

def _vmd_cuda_expf(x: np.ndarray | float) -> np.ndarray:
    """Single-precision exponential used by VMD CUDAMDFF."""
    return np.exp(np.asarray(x, dtype=np.float32))

def _validate_atoms(coords_A,radii_A):
    c=np.ascontiguousarray(coords_A,dtype=np.float32); r=np.ascontiguousarray(radii_A,dtype=np.float32)
    if c.ndim!=2 or c.shape[1]!=3 or c.shape[0]==0: raise ValueError("coords_A must have shape (n_atoms, 3) and be non-empty")
    if r.shape!=(c.shape[0],) or np.any(r<=0): raise ValueError("radii_A must have one positive value per coordinate")
    return c,r

def quicksurf_density_cpu(coords_A,radii_A,grid:GridSpec,radius_scale_A:float,cutoff_sigma:float=2.0):
    c,r=_validate_atoms(coords_A,radii_A); d=np.zeros(grid.shape,dtype=np.float32); org=grid.origin_A.astype(np.float32); sp=np.float32(grid.spacing_A)
    rel=c-org; acsp=np.float32(cutoff_sigma)*np.float32(radius_scale_A)*np.max(r); iv=np.float32(1.0)/acsp
    cs=np.maximum(1,np.floor(np.asarray(grid.shape,dtype=np.float32)*sp/acsp).astype(np.int32)); nc=int(np.prod(cs,dtype=np.int64))
    xyz=np.floor(rel/acsp).astype(np.int32); xyz=np.maximum(xyz,0); xyz=np.minimum(xyz,cs-1); h=(xyz[:,2]*cs[1]+xyz[:,1])*cs[0]+xyz[:,0]
    order=np.argsort(h,kind="stable"); h=h[order]; rel=rel[order]; scaled=np.float32(radius_scale_A)*r[order]
    ar=-np.float32(0.5)*np.float32(1.4426950408889634)/(scaled*scaled); counts=np.bincount(h,minlength=nc).astype(np.int64); starts=np.empty(nc,dtype=np.int64); starts[0]=0
    if nc>1: starts[1:]=np.cumsum(counts[:-1],dtype=np.int64)
    ends=starts+counts; nx,ny,nz=grid.shape
    for bx0 in range(0,nx,8):
        bx1=bx0+8; x0=max(0,int(np.float32(np.float32(bx0)*sp-acsp)*iv)); x1=min(int(cs[0])-1,int(np.float32(np.float32(bx1)*sp+acsp)*iv))
        xs=np.arange(bx0,min(bx1,nx),dtype=np.float32)*sp
        for by0 in range(0,ny,8):
            by1=by0+8; y0=max(0,int(np.float32(np.float32(by0)*sp-acsp)*iv)); y1=min(int(cs[1])-1,int(np.float32(np.float32(by1)*sp+acsp)*iv))
            ys=np.arange(by0,min(by1,ny),dtype=np.float32)*sp
            for bz0 in range(0,nz,8):
                bz1=bz0+8; z0=max(0,int(np.float32(np.float32(bz0)*sp-acsp)*iv)); z1=min(int(cs[2])-1,int(np.float32(np.float32(bz1)*sp+acsp)*iv))
                zs=np.arange(bz0,min(bz1,nz),dtype=np.float32)*sp; block=np.zeros((xs.size,ys.size,zs.size),dtype=np.float32)
                for cz in range(z0,z1+1):
                    for cy in range(y0,y1+1):
                        for cx in range(x0,x1+1):
                            cell=(cz*int(cs[1])+cy)*int(cs[0])+cx
                            for atom in range(int(starts[cell]),int(ends[cell])):
                                dx=xs[:,None,None]-rel[atom,0]; dy=ys[None,:,None]-rel[atom,1]; dz=zs[None,None,:]-rel[atom,2]
                                block+=_vmd_cuda_expf(ar[atom]*(dx*dx+dy*dy+dz*dz))
                d[bx0:min(bx1,nx),by0:min(by1,ny),bz0:min(bz1,nz)]=block
    return d

_DENSITY_RAWKERNEL=None
_DENSITY_KERNEL=r"""
extern "C" __global__ void gaussdensity_atom_reference(const float4* atoms,int nx,int ny,int nz,int cnx,int cny,int cnz,float ox,float oy,float oz,float acx,float acy,float acz,float sp,float acsp,const int* starts,const int* ends,float* out){int ix=blockIdx.x*blockDim.x+threadIdx.x,iy=blockIdx.y*blockDim.y+threadIdx.y,iz=blockIdx.z*blockDim.z+threadIdx.z;if(ix>=nx||iy>=ny||iz>=nz)return;float iv=1.0f/acsp;int bx0=blockIdx.x*8,by0=blockIdx.y*8,bz0=blockIdx.z*8,bx1=bx0+8,by1=by0+8,bz1=bz0+8;int x0=(int)((ox+(float)bx0*sp-acx-acsp)*iv),y0=(int)((oy+(float)by0*sp-acy-acsp)*iv),z0=(int)((oz+(float)bz0*sp-acz-acsp)*iv),x1=(int)((ox+(float)bx1*sp-acx+acsp)*iv),y1=(int)((oy+(float)by1*sp-acy+acsp)*iv),z1=(int)((oz+(float)bz1*sp-acz+acsp)*iv);x0=max(0,x0);y0=max(0,y0);z0=max(0,z0);x1=min(cnx-1,x1);y1=min(cny-1,y1);z1=min(cnz-1,z1);float px=ox+(float)ix*sp,py=oy+(float)iy*sp,pz=oz+(float)iz*sp,d=0.0f;for(int cz=z0;cz<=z1;++cz)for(int cy=y0;cy<=y1;++cy)for(int cx=x0;cx<=x1;++cx){int cell=(cz*cny+cy)*cnx+cx;for(int a=starts[cell];a<ends[cell];++a){float4 q=atoms[a];float dx=px-q.x,dy=py-q.y,dz=pz-q.z;d+=__expf(q.w*(dx*dx+dy*dy+dz*dz));}}out[(ix*ny+iy)*nz+iz]=d;}
"""

def cupy_available():
    try:
        import cupy; return bool(cupy.cuda.is_available())
    except Exception: return False

def _quicksurf_density_cuda_atom(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma=2.0):
    return _quicksurf_density_cuda_cell_list(
        coords_A, radii_A, grid, radius_scale_A, cutoff_sigma, None, reference=True
    )

_DENSITY_CELL_KERNEL=r"""
extern "C" __global__ void gaussdensity_cell_list(const float4* atoms,int nx,int ny,int nz,int cnx,int cny,int cnz,float ox,float oy,float oz,float acx,float acy,float acz,float sp,float acsp,const int* starts,const int* ends,float* out){constexpr int U=4;int ix=blockIdx.x*blockDim.x+threadIdx.x,iy=blockIdx.y*blockDim.y+threadIdx.y,iz0=(blockIdx.z*blockDim.z+threadIdx.z)*U;if(ix>=nx||iy>=ny)return;float iv=1.0f/acsp;int bx0=blockIdx.x*blockDim.x,by0=blockIdx.y*blockDim.y,bz0=blockIdx.z*blockDim.z*U,bx1=(blockIdx.x+1)*blockDim.x,by1=(blockIdx.y+1)*blockDim.y,bz1=(blockIdx.z+1)*blockDim.z*U;int x0=(int)((ox+(float)bx0*sp-acx-acsp)*iv),y0=(int)((oy+(float)by0*sp-acy-acsp)*iv),z0=(int)((oz+(float)bz0*sp-acz-acsp)*iv),x1=(int)((ox+(float)bx1*sp-acx+acsp)*iv),y1=(int)((oy+(float)by1*sp-acy+acsp)*iv),z1=(int)((oz+(float)bz1*sp-acz+acsp)*iv);x0=max(0,x0);y0=max(0,y0);z0=max(0,z0);x1=min(cnx-1,x1);y1=min(cny-1,y1);z1=min(cnz-1,z1);for(int uz=0;uz<U;++uz){int iz=iz0+uz;if(iz>=nz)continue;float px=ox+(float)ix*sp,py=oy+(float)iy*sp,pz=oz+(float)iz*sp,d=0.0f;for(int cz=z0;cz<=z1;++cz)for(int cy=y0;cy<=y1;++cy)for(int cx=x0;cx<=x1;++cx){int cell=(cz*cny+cy)*cnx+cx;for(int a=starts[cell];a<ends[cell];++a){float4 q=atoms[a];float dx=px-q.x,dy=py-q.y,dz=pz-q.z;d+=__expf(q.w*(dx*dx+dy*dy+dz*dz));}}out[(ix*ny+iy)*nz+iz]=d;}}
"""
_DENSITY_CELL_RAWKERNEL=None

def _quicksurf_density_cuda_cell_list(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma,accel_grid_spacing_A,*,reference=False):
    c,r=_validate_atoms(coords_A,radii_A)
    try: import cupy as cp
    except ImportError as e: raise RuntimeError("backend='cuda' requires CuPy") from e
    if not cp.cuda.is_available(): raise RuntimeError("CuPy is installed but no CUDA device is available")
    acsp=float(accel_grid_spacing_A if accel_grid_spacing_A is not None else cutoff_sigma*radius_scale_A*float(np.max(r)))
    if acsp<=0: raise ValueError("accel_grid_spacing_A must be positive")
    dc,dr=cp.asarray(c),cp.asarray(r); rel=dc-cp.asarray(grid.origin_A,dtype=cp.float32); log2e=np.float32(np.log2(np.float32(2.718281828))); sc=np.float32(radius_scale_A)*dr; ar=-np.float32(0.5)*log2e/(sc*sc); atoms0=cp.concatenate((rel,ar[:,None]),axis=1)
    cs=np.maximum(1,np.floor(np.asarray(grid.shape,dtype=np.float32)*np.float32(grid.spacing_A)/np.float32(acsp)).astype(np.int32)); nc=int(np.prod(cs,dtype=np.int64));
    if nc>=2**31: raise ValueError("acceleration cell list is too large")
    xyz=cp.floor(rel/np.float32(acsp)).astype(cp.int32); xyz=cp.maximum(xyz,0); xyz=cp.minimum(xyz,cp.asarray(cs-1,dtype=cp.int32)); h=(xyz[:,2]*int(cs[1])+xyz[:,1])*int(cs[0])+xyz[:,0]; order=cp.argsort(h); atoms=atoms0[order]; counts=cp.bincount(h,minlength=nc).astype(cp.int32); starts=cp.empty_like(counts); starts[0]=0
    if nc>1: starts[1:]=cp.cumsum(counts[:-1],dtype=cp.int32)
    ends=starts+counts
    global _DENSITY_CELL_RAWKERNEL, _DENSITY_RAWKERNEL
    if reference:
        if _DENSITY_RAWKERNEL is None:_DENSITY_RAWKERNEL=cp.RawKernel(_DENSITY_KERNEL,"gaussdensity_atom_reference")
        kernel=_DENSITY_RAWKERNEL; th=(8,8,8)
    else:
        if _DENSITY_CELL_RAWKERNEL is None:_DENSITY_CELL_RAWKERNEL=cp.RawKernel(_DENSITY_CELL_KERNEL,"gaussdensity_cell_list")
        kernel=_DENSITY_CELL_RAWKERNEL; th=(8,8,2)
    out=cp.empty(grid.shape,dtype=cp.float32); bl=((grid.shape[0]+7)//8,(grid.shape[1]+7)//8,(grid.shape[2]+7)//8)
    kernel(bl,th,(atoms,np.int32(grid.shape[0]),np.int32(grid.shape[1]),np.int32(grid.shape[2]),np.int32(cs[0]),np.int32(cs[1]),np.int32(cs[2]),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(grid.spacing_A),np.float32(acsp),starts,ends,out)); return out

def quicksurf_density_cuda(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma=2.0,*,kernel_mode="cell_list",accel_grid_spacing_A=None):
    if kernel_mode=="atom": return _quicksurf_density_cuda_atom(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma)
    if kernel_mode=="cell_list": return _quicksurf_density_cuda_cell_list(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma,accel_grid_spacing_A)
    raise ValueError("kernel_mode must be 'cell_list' or 'atom'")
