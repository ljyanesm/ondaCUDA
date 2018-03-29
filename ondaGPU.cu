/*
Desarrollado por Luis Yanes yanes.luis@gmail.com
Basado en codigo de NVIDIA Developer Toolkit
*/


#include <cuda.h>
#include <stdio.h>
#include "SimParams.h"

#define k_blockDimX 16
#define k_blockDimY 16
#define k_blockSizeMin 64
#define RADIUS 4


__constant__ SimParams param;


__global__ void impulseKernel(float *p1,    // Presion salida (OUT)
                              const int dimx,
                              const int dimy,
                              const int dimz,
                              const int ixsource,
                              const int iysource,
                              const int izsource,
                              const float val)
{
  int gtidx = blockIdx.x * blockDim.x + threadIdx.x;    // Posicion en X
  int gtidy = blockIdx.y * blockDim.y + threadIdx.y;    // Posicion en Y
  if (gtidx == ixsource && gtidy == iysource){
    int stride_y    = dimx;
    int stride_z    = stride_y*dimy;

    int outIdx0 = (izsource ) * stride_z + (gtidy - 1)* stride_y + (gtidx - 1);
    int outIdx1 = (izsource ) * stride_z + (gtidy - 1)* stride_y + (gtidx );
    int outIdx2 = (izsource ) * stride_z + (gtidy )* stride_y + (gtidx - 1);
    int outIdx3 = (izsource ) * stride_z + (gtidy )* stride_y + (gtidx );
    int outIdx4 = (izsource - 1) * stride_z + (gtidy - 1)* stride_y + (gtidx - 1);
    int outIdx5 = (izsource - 1) * stride_z + (gtidy - 1)* stride_y + (gtidx );
    int outIdx6 = (izsource - 1) * stride_z + (gtidy )* stride_y + (gtidx - 1);
    int outIdx7 = (izsource - 1) * stride_z + (gtidy )* stride_y + (gtidx );
  
    p1[outIdx0] += val;
    p1[outIdx1] += val;
    p1[outIdx2] += val;
    p1[outIdx3] += val;
    p1[outIdx4] += val;
    p1[outIdx5] += val;
    p1[outIdx6] += val;
    p1[outIdx7] += val;
  }
}

__global__ void fofdKernel(float *p2,        // Presion salida   (OUT)
                           const float *p1,        // Presion actual   (IN OUT)
                           const float *p0,  // Presion anterior (IN)
                           const float *c,
                           const int dimx,
                           const int dimy,
                           const int dimz)
{
  //printf("Y = %d\n", blockIdx.y);
  bool valid = true;
  int gtidx = blockIdx.x * blockDim.x + threadIdx.x;    // Posicion en X
  int gtidy = blockIdx.y * blockDim.y + threadIdx.y;    // Posicion en Y
  int ltidx = threadIdx.x;
  int ltidy = threadIdx.y;
  int workx = blockDim.x;
  int worky = blockDim.y;
  __shared__ float tile[k_blockDimX + 2 * 1][k_blockDimY + 2 * 1];   // Bloque de trabajo    
  int stride_y    = dimx;
  int stride_z    = stride_y * dimy;
  int inputIndex  = gtidy * stride_y + gtidx;   // Posicion [x,y,0] transformada a una dimension para acceder al vector de entrada.
  int outputIndex = 0;

  float infront;
  float behind;
  float current;
  float tr1, tr2, tr3, tr4;
  float value = 0.0f;

	int tx = ltidx + 1;
	int ty = ltidy + 1;
	
  if (gtidx <= dimx-1 && gtidy <= dimy-1)
  {  
    if (gtidx < 1 || gtidx >= (dimx - 1))   // Out of bounds?
      valid = false;
    if (gtidy < 1 || gtidy >= (dimy - 1))   // Out of bounds?
      valid = false;

      // For simplicity we assume that the global size is equal to the actual
      // problem size; since the global size must be a multiple of the local size
      // this means the problem size must be a multiple of the local size (or
      // padded to meet this constraint).
      // Preload the "infront" and "behind" data
    behind = p1[inputIndex];    // Posiciones [x, y, z=0]
    inputIndex += stride_z;  // Z = 1
    
    current = p1[inputIndex];
    outputIndex = inputIndex;
    // Step through the xy-planes
    for (int z = 1 ; z < (dimz - 1) ; z++)
    {
      inputIndex += stride_z;   // Z = Z+1
      infront = p1[inputIndex];

      __syncthreads();          // Nos aseguramos que todos los hilos tengan los indices adecuados

      // Update the data slice in the local tile
      // Halo above & below
      if (ltidy < 1)
      {
        tile[ltidy][tx]             = p1[outputIndex - (1 * stride_y)];
        tile[ltidy + worky + 1][tx] = p1[outputIndex + (worky * stride_y)];
      }
      // Halo left & right
      if (ltidx < 1)
      {
        tile[ty][ltidx]             = p1[outputIndex - 1];
        tile[ty][ltidx + workx + 1] = p1[outputIndex + workx];
      }
      tile[ty][tx] = current;
          
      __syncthreads();  // Esperamos que todos los hilos hallan cargado la memoria compartida.

      tr1 = (tile[ty][tx-1] + tile[ty][tx+1] - 2.0f*current) / param.dx2;
      tr2 = (tile[ty-1][tx] + tile[ty+1][tx] - 2.0f*current) / param.dy2;
      tr3 = (infront + behind - 2.0f*current) / param.dz2;
      tr4 = pow(c[outputIndex], 2) * param.dt2 * (tr1 + tr2 + tr3);
      value = tr4 + (2.0f * p1[outputIndex]) - p0[outputIndex];
      // Store the output value
      if (! valid) value = current;
      p2[outputIndex] = value;
      
      behind = current;
      outputIndex = inputIndex;
      current = infront;
    }
  }
}


__global__ void bcKernel(float *p2,
                         float *p1,
                         const int dimx,
                         const int dimy,
                         const int dimz,
                         const int dampx,
                         const int dampy,
                         const int dampz)
{
  
  int gtidx = blockIdx.x * blockDim.x + threadIdx.x;    // Posicion en X
  int gtidy = blockIdx.y * blockDim.y + threadIdx.y;    // Posicion en Y

  int stride_y    = dimx;
  int stride_z    = stride_y * dimy;
  int inputIndex  = gtidy * stride_y + gtidx;   // Posicion [x,y,0] transformada a una dimension para acceder al vector de entrada.
  int bcIdx = 0;
  
  if (gtidx <= dimx-1 && gtidy <= dimy-1)
  {
    for (int z = 0 ; z < dimz ; z++)
    {
      bcIdx = z * stride_z + inputIndex;
      // X BC Width
      if (gtidx < dampx){   
        p2[bcIdx] *= exp(-pow(0.015*(gtidx),2));
        p1[bcIdx] *= exp(-pow(0.015*(gtidx),2));
      }

      if (gtidx > dimx - dampx){
        p2[bcIdx] *= exp(-pow(0.015*(dimx - gtidx),2));
        p1[bcIdx] *= exp(-pow(0.015*(dimx - gtidx),2));
      }
      
      // Y BC Heigth
      if (gtidy == 0){
        p2[bcIdx] = 0.0f;
        //p2[bcIdx] *= exp(-pow(0.015*(gtidy),2));
        //p1[bcIdx] *= exp(-pow(0.015*(gtidy),2));
      }
      
      if (gtidy > dimy - dampy){
        p2[bcIdx] *= exp(-pow(0.015*(dimy - gtidy),2));
        p1[bcIdx] *= exp(-pow(0.015*(dimy - gtidy),2));
      }
            
      // Z BC Depth
      // FREE SURFACE
      if (z < dampz){
        //p2[bcIdx] = 0.0f;
        p2[bcIdx] *= exp(-pow(0.015*(dampz - z),2));
        p1[bcIdx] *= exp(-pow(0.015*(dampz - z),2));
      }

      if (z > dimz - dampz){
        p2[bcIdx] *= exp(-pow(0.015*( dimz - dampz - z),2));
        p1[bcIdx] *= exp(-pow(0.015*( dimz - dampz - z),2));
      }

    }
  }
}

extern "C" void gpuGetNumDevices(int &v){
	int num;

	cudaGetDeviceCount(&num);

	v = num;
}

extern "C" void gpuSetDevice(int d){
  cudaSetDevice(d);
}

extern "C" void gpuMemcpyToDevice(float* dest, const float* src, unsigned int size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

extern "C" void gpuMemcpyToHost(float* dest, const float* src, unsigned int size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

extern "C" void gpuMemcpyToSymbol(const void* src, unsigned int size)
{

    cudaMemcpyToSymbol(param, src, size);
}

extern "C" void* gpuMalloc(unsigned int size)
{
  void *pntr1;
  cudaMalloc(&pntr1, size);
  return pntr1;
}

extern "C" void gpuImpulse(float* p1, const int dx, const int dy, const int dz, const int ix, const int iy, const int iz, const float val)
{
  dim3  dimBlock, dimGrid;

  dimBlock.x = k_blockDimX;
  dimBlock.y = k_blockDimY;
  dimGrid.x = (unsigned int) ceil( (float) dx / dimBlock.x );
  dimGrid.y = (unsigned int) ceil( (float) dy / dimBlock.y );
  
  cudaThreadSynchronize();
  impulseKernel<<<dimGrid, dimBlock>>>(p1, dx, dy, dz, ix, iy, iz, val);
}

extern "C" void gpuFirstOrderFD(float *p2, const float *p1, const float *p0, const float *c, const int dimx, const int dimy, const int dimz)
{
  dim3  dimBlock, dimGrid;

  dimBlock.x = k_blockDimX;
  dimBlock.y = k_blockDimY;
  dimGrid.x = (unsigned int) ceil( (float) dimx / dimBlock.x );
  dimGrid.y = (unsigned int) ceil( (float) dimy / dimBlock.y );
  
  cudaThreadSynchronize();
  fofdKernel<<<dimGrid, dimBlock>>>(p2, p1, p0, c, dimx, dimy, dimz);
}

extern "C" void gpuBoundaryConditions(float *p2, float *p1, const float *p0, const float *c, const int dimx, const int dimy, const int dimz, const int dampx, const int dampy, const int dampz)
{
  dim3  dimBlock, dimGrid;

  dimBlock.x = k_blockDimX;
  dimBlock.y = k_blockDimY;
  dimGrid.x = (unsigned int) ceil( (float) dimx / dimBlock.x );
  dimGrid.y = (unsigned int) ceil( (float) dimy / dimBlock.y );

  cudaThreadSynchronize();
  bcKernel<<<dimGrid, dimBlock>>>(p2, p1, dimx, dimy, dimz, dampx, dampy, dampz);

}

extern "C" void gpuFree(void* pntr)
{
  cudaFree(pntr);
}
