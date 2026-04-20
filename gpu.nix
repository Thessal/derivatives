{ pkgs, ... }:
{
  cuda = ''
     export CUDA_PATH=${pkgs.cudatoolkit}
     export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidiaPackages.stable}/lib:${pkgs.cudaPackages.cuda_nvrtc.lib}/lib:$LD_LIBRARY_PATH
     # LD_LIBRARY_PATH=/run/opengl-driver/lib
     export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidiaPackages.stable}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
     export EXTRA_CCFLAGS="-I/usr/include"
  ''; 
  rocm = ''
   export ROCM_PATH=${pkgs.rocmPackages.clr}
   export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.rocblas}/lib:$LD_LIBRARY_PATH
   export EXTRA_LDFLAGS="-L${pkgs.rocmPackages.clr}/lib -L${pkgs.rocmPackages.rocblas}/lib"
   export EXTRA_CCFLAGS="-I${pkgs.rocmPackages.clr}/include"
  '';
}
