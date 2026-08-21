"""CUDA and CPU Gaussian density kernels used by volinterior."""
from __future__ import annotations
import numpy as np
from .config import GridSpec

def _vmd_aexpfnx(x: np.ndarray | float) -> np.ndarray:
    v=np.asarray(x,dtype=np.float32); o=np.zeros_like(v,dtype=np.float32); m=v>=np.float32(-10.0)
    if np.any(m):
        xm=v[m]; mb=xm*np.float32(-1.4426950408889634); fl=mb.astype(np.int32); d=fl.astype(np.float32)-mb
        sy=np.float32(1.0)+d*(np.float32(0.6987082824680118)+d*(np.float32(0.2633174272827404)+d*(np.float32(0.0923611991471395)+d*np.float32(0.0277520543324108))))
        bits=((np.int32(127)-fl)<<np.int32(23)).astype(np.int32); o[m]=sy*bits.view(np.float32)
    return o

def _validate_atoms(coords_A,radii_A):
    c=np.ascontiguousarray(coords_A,dtype=np.float32); r=np.ascontiguousarray(radii_A,dtype=np.float32)
    if c.ndim!=2 or c.shape[1]!=3 or c.shape[0]==0: raise ValueError("coords_A must have shape (n_atoms, 3) and be non-empty")
    if r.shape!=(c.shape[0],) or np.any(r<=0): raise ValueError("radii_A must have one positive value per coordinate")
    return c,r

def quicksurf_density_cpu(coords_A,radii_A,grid:GridSpec,radius_scale_A:float,cutoff_sigma:float=2.0):
    c,r=_validate_atoms(coords_A,radii_A); d=np.zeros(grid.shape,dtype=np.float32); org=grid.origin_A.astype(np.float32); sp=np.float32(grid.spacing_A); shape=np.asarray(grid.shape,dtype=np.int64); lim=np.float32(cutoff_sigma)*np.float32(radius_scale_A)*np.max(r); lim2=lim*lim
    for pos,rad in zip(c,r,strict=True):
        rel=pos-org; scaled=np.float32(radius_scale_A)*np.float32(rad); lo=np.maximum(0,(rel/sp-lim/sp).astype(np.int64)); hi=np.minimum(shape-1,(rel/sp+lim/sp).astype(np.int64))
        if np.any(lo>hi): continue
        ar=-np.float32(0.5)*np.float32(1.4426950408889634)/(scaled*scaled)
        for iz in range(int(lo[2]),int(hi[2])+1):
            dz=np.float32(iz)*sp-rel[2]
            for iy in range(int(lo[1]),int(hi[1])+1):
                dy=np.float32(iy)*sp-rel[1]; x=np.arange(int(lo[0]),int(hi[0])+1,dtype=np.float32); r2=(x*sp-rel[0])**2+dy*dy+dz*dz; val=_vmd_aexpfnx(ar*r2); val[r2>=lim2]=np.float32(0); d[int(lo[0]):int(hi[0])+1,iy,iz]+=val
    return d

_DENSITY_RAWKERNEL=None
_DENSITY_KERNEL=r"""
__device__ __forceinline__ float vmd_aexpfnx(float x){if(x<-10.0f)return 0.0f;const float m=-1.4426950408889634f;float mb=x*m;int f=(int)mb;float d=((float)f)-mb;float s=1.0f+d*(0.6987082824680118f+d*(0.2633174272827404f+d*(0.0923611991471395f+d*0.0277520543324108f)));int bits=(127-f)<<23;return s*__int_as_float(bits);}
extern "C" __global__ void splat_density(const float* c,const float* r,int n,float ox,float oy,float oz,int nx,int ny,int nz,float sp,float rs,float sig,float mr,float* out){int a=blockDim.x*blockIdx.x+threadIdx.x;if(a>=n)return;float px=c[3*a]-ox,py=c[3*a+1]-oy,pz=c[3*a+2]-oz,sc=rs*r[a],lim=sig*rs*mr,iv=1.0f/sp;int x0=max(0,(int)(px*iv-lim*iv)),y0=max(0,(int)(py*iv-lim*iv)),z0=max(0,(int)(pz*iv-lim*iv)),x1=min(nx-1,(int)(px*iv+lim*iv)),y1=min(ny-1,(int)(py*iv+lim*iv)),z1=min(nz-1,(int)(pz*iv+lim*iv));float ar=-0.5f*1.4426950408889634f/(sc*sc),lim2=lim*lim;for(int ix=x0;ix<=x1;++ix){float dx=(float)ix*sp-px;for(int iy=y0;iy<=y1;++iy){float dy=(float)iy*sp-py;for(int iz=z0;iz<=z1;++iz){float dz=(float)iz*sp-pz,r2=dx*dx+dy*dy+dz*dz;if(r2>=lim2)continue;atomicAdd(&out[(ix*ny+iy)*nz+iz],vmd_aexpfnx(ar*r2));}}}}
"""

def cupy_available():
    try:
        import cupy; return bool(cupy.cuda.is_available())
    except Exception: return False

def _quicksurf_density_cuda_atom(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma=2.0):
    c,r=_validate_atoms(coords_A,radii_A)
    try: import cupy as cp
    except ImportError as e: raise RuntimeError("backend='cuda' requires CuPy") from e
    if not cp.cuda.is_available(): raise RuntimeError("CuPy is installed but no CUDA device is available")
    global _DENSITY_RAWKERNEL; dc,dr=cp.asarray(c),cp.asarray(r); out=cp.zeros(grid.shape,dtype=cp.float32)
    if _DENSITY_RAWKERNEL is None:_DENSITY_RAWKERNEL=cp.RawKernel(_DENSITY_KERNEL,"splat_density")
    t=128; _DENSITY_RAWKERNEL(((c.shape[0]+t-1)//t,),(t,),(dc,dr,np.int32(c.shape[0]),np.float32(grid.origin_A[0]),np.float32(grid.origin_A[1]),np.float32(grid.origin_A[2]),np.int32(grid.shape[0]),np.int32(grid.shape[1]),np.int32(grid.shape[2]),np.float32(grid.spacing_A),np.float32(radius_scale_A),np.float32(cutoff_sigma),np.float32(np.max(r)),out)); return out

_DENSITY_CELL_KERNEL=r"""
extern "C" __global__ void gaussdensity_cell_list(const float4* atoms,int nx,int ny,int nz,int cnx,int cny,int cnz,float ox,float oy,float oz,float acx,float acy,float acz,float sp,float acsp,const int* starts,const int* ends,float* out){constexpr int U=4;int ix=blockIdx.x*blockDim.x+threadIdx.x,iy=blockIdx.y*blockDim.y+threadIdx.y,iz0=(blockIdx.z*blockDim.z+threadIdx.z)*U;if(ix>=nx||iy>=ny)return;float iv=1.0f/acsp;int bx0=blockIdx.x*blockDim.x,by0=blockIdx.y*blockDim.y,bz0=blockIdx.z*blockDim.z*U,bx1=(blockIdx.x+1)*blockDim.x,by1=(blockIdx.y+1)*blockDim.y,bz1=(blockIdx.z+1)*blockDim.z*U;int x0=(int)((ox+(float)bx0*sp-acx-acsp)*iv),y0=(int)((oy+(float)by0*sp-acy-acsp)*iv),z0=(int)((oz+(float)bz0*sp-acz-acsp)*iv),x1=(int)((ox+(float)bx1*sp-acx+acsp)*iv),y1=(int)((oy+(float)by1*sp-acy+acsp)*iv),z1=(int)((oz+(float)bz1*sp-acz+acsp)*iv);x0=max(0,x0);y0=max(0,y0);z0=max(0,z0);x1=min(cnx-1,x1);y1=min(cny-1,y1);z1=min(cnz-1,z1);for(int uz=0;uz<U;++uz){int iz=iz0+uz;if(iz>=nz)continue;float px=ox+(float)ix*sp,py=oy+(float)iy*sp,pz=oz+(float)iz*sp,d=0.0f;for(int cz=z0;cz<=z1;++cz)for(int cy=y0;cy<=y1;++cy)for(int cx=x0;cx<=x1;++cx){int cell=(cz*cny+cy)*cnx+cx;for(int a=starts[cell];a<ends[cell];++a){float4 q=atoms[a];float dx=px-q.x,dy=py-q.y,dz=pz-q.z;d+=__expf(q.w*(dx*dx+dy*dy+dz*dz));}}out[(ix*ny+iy)*nz+iz]=d;}}
"""
_DENSITY_CELL_RAWKERNEL=None

def _quicksurf_density_cuda_cell_list(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma,accel_grid_spacing_A):
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
    global _DENSITY_CELL_RAWKERNEL
    if _DENSITY_CELL_RAWKERNEL is None:_DENSITY_CELL_RAWKERNEL=cp.RawKernel(_DENSITY_CELL_KERNEL,"gaussdensity_cell_list")
    out=cp.empty(grid.shape,dtype=cp.float32); th=(8,8,2); bl=((grid.shape[0]+7)//8,(grid.shape[1]+7)//8,(grid.shape[2]+7)//8)
    _DENSITY_CELL_RAWKERNEL(bl,th,(atoms,np.int32(grid.shape[0]),np.int32(grid.shape[1]),np.int32(grid.shape[2]),np.int32(cs[0]),np.int32(cs[1]),np.int32(cs[2]),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(0),np.float32(grid.spacing_A),np.float32(acsp),starts,ends,out)); return out

def quicksurf_density_cuda(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma=2.0,*,kernel_mode="cell_list",accel_grid_spacing_A=None):
    if kernel_mode=="atom": return _quicksurf_density_cuda_atom(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma)
    if kernel_mode=="cell_list": return _quicksurf_density_cuda_cell_list(coords_A,radii_A,grid,radius_scale_A,cutoff_sigma,accel_grid_spacing_A)
    raise ValueError("kernel_mode must be 'cell_list' or 'atom'")
