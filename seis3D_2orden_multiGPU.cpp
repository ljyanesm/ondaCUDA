/******************************************************************************/
/*  TERRASA - Version 2.0 - Secuential Wave propagation - Definition
 Copyright (C) 2007

 Developed by: Jose Ramirez (jbarrios@.com)
 Jose Colmenares (jcolmenares5@uc.edu.ve)
 Deybi Exposito (expositodeybi@gmail.com)

 CUDA module developed by: Luis Yanes (yanes.luis@gmail.com)

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/******************************************************************************/

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <pthread.h>
#include "cSwapBytes.h"
#include "SimParams.h"
using namespace std;

extern "C" void gpuGetNumDevices(int &v);
extern "C" void gpuSetDevice(int device);
extern "C" void gpuMemcpyToDevice(void* dest, const void* src,
    unsigned int size);
extern "C" void gpuMemcpyToHost(void* dest, const void* src, unsigned int size);
extern "C" void gpuMemcpyToSymbol(const void* src, unsigned int size);
extern "C" void* gpuMalloc(unsigned int size);
extern "C" void gpuImpulse(float* p1, const int dx, const int dy, const int dz,
    const int ix, const int iy, const int iz, const float val);
extern "C" void gpuFirstOrderFD(float *p2, const float *p1, const float *p0,
    const float *c, const int dimx, const int dimy, const int dimz);
extern "C" void gpuBoundaryConditions(float *p2, float *p1, const float *p0,
    const float *c, const int dimx, const int dimy, const int dimz,
    const int dampx, const int dampy, const int dampz);
extern "C" void gpuFree(void* pntr);

bool ricker(float f, float dt, FILE *file);
bool gaussian(float freq, float dt, int nt, FILE *file);
void liberarEstructuras(int *ixrec, int *iyrec, int *izrec, int *ixsource,
    int *iysource, int *izsource, float *source, int *nRecsByLine, float **p,
    int np1, float ***c, int nc1, int nc2, float **multplx, int nm1,
    char *usedRecsBySource, cDataShot *usedShots);

void *hiloGPU(void *data) {
  ThreadData *my_data;
  int cuda;
  int yy, jj, kk, nn, mm, ii;
  float tr1, tr2, tr3, tr4;
  float dx2, dy2, dz2;
  float mul;
  int offset1, offset2;
  int nx, ny, nz, nt, nrecs, nsamp, idsnap, idt, nsource;
  int itemp, tmp;
  int device;
  int iouttest, itrtest, msgScreen;
  int mulcnt;
  int iniShot;
  int tmpShot, tmpShotRel;
  int ddamp, wdamp;
  char relationalFile[200];
  char snapshots[200];
  char cmd[200];
  char str[500];
  char *usedRecsBySource;
  float **multplx = NULL;
  int *ixsource, *iysource, *izsource;
  int nLinesR;
  int *nRecsByLine;
  int volumeSize;
  int *ixrec, *iyrec, *izrec;
  float *source;
  float ***c = NULL;
  float *h_c = NULL;
  float xo, xf, yo, yf;
  float depth;
  int numsnaps;
  float dx, dy, dz, dt2;
  float **p;
  float *d_p0, *d_p1, *d_p2, *d_c;
  float time1;
  cDataShot *shotList = NULL;
  SimParams h_params;
  FILE *relFile, *cubo, *salida, *auxfile, *fileRecsNoUsed;

  my_data = (ThreadData*) data;
  shotList = my_data->usedShots;
  nsamp = my_data->nsamp;
  nrecs = my_data->nrecs;
  idt = my_data->idt;
  cuda = my_data->cuda;
  ixsource = my_data->ixsource;
  iysource = my_data->iysource;
  izsource = my_data->izsource;
  source = my_data->source;
  nsource = my_data->nSource;
  ixrec = my_data->ixrec;
  iyrec = my_data->iyrec;
  izrec = my_data->izrec;
  nLinesR = my_data->nLinesR;
  nRecsByLine = my_data->nRecsByLine;
  volumeSize = my_data->volumeSize;
  c = my_data->c;
  h_c = my_data->h_c;
  h_params = my_data->h_params;
  xo = my_data->xo;
  yo = my_data->yo;
  xf = my_data->xf;
  yf = my_data->yf;
  depth = my_data->depth;
  numsnaps = my_data->numsnaps;
  dx = my_data->dx;
  dy = my_data->dy;
  dz = my_data->dz;
  time1 = my_data->time1;
  ddamp = my_data->ddamp;
  wdamp = my_data->wdamp;
  device = my_data->device;
  nt = my_data->nt;
  idsnap = my_data->idsnap;
  nx = my_data->nx;
  ny = my_data->ny;
  nz = my_data->nz;
  nsource = my_data->nSource;
  iniShot = my_data->iniShot;

  dx2 = dx*dx;
  dy2 = dy*dy;
  dz2 = dz*dz;
  dt2 = h_params.dt2;

  if (cuda) {
    gpuSetDevice(device);
    d_c = (float*) gpuMalloc(volumeSize * sizeof(float));
    d_p0 = (float*) gpuMalloc(volumeSize * sizeof(float));
    d_p1 = (float*) gpuMalloc(volumeSize * sizeof(float));
    d_p2 = (float*) gpuMalloc(volumeSize * sizeof(float));

    /*** Copiar parametros numericos a CUDA ***/
    gpuMemcpyToSymbol((void *) &h_params, sizeof(SimParams));

    /* Hacer copias a CUDA */
    gpuMemcpyToDevice(d_c, h_c, volumeSize * sizeof(float));
  }

  p = (float**) malloc(3 * sizeof(float*));
  if (p == NULL) {
    printf("Error en memoria 9\n");
  }
  for (ii = 0; ii < 3; ii++) {
    kk = (nx) * (ny) * (nz);
    p[ii] = (float *) malloc((int) kk * sizeof(float));
    if (p[ii] == NULL) {
      printf("Error en memoria 10\n");
    }
  }

  multplx = (float**) malloc(nsamp * sizeof(float*));
  if (multplx == NULL) {
    printf("Error en memoria 13\n");
  }
  for (ii = 0; ii < nsamp; ii++) {
    multplx[ii] = (float*) malloc(nrecs * sizeof(float));
    if (multplx[ii] == NULL) {
      printf("Error en memoria 14\n");
    }
  }

  usedRecsBySource =(char *) malloc(nrecs * sizeof(char));

  relFile = fopen(my_data->relFile, "r");

  itemp = fscanf(relFile, "%d%*[^\n]%*c", &tmp); // Cantidad de emisores relacionados
  itemp = fscanf(relFile, "%d%*[^\n]%*c", &tmp); // Lectura de la cantidad de receptores (No es usado porque este valor ya fue leido)
  itemp = fscanf(relFile, "%[^\n]%*c", relationalFile); // Comentario

  for (int i = 0; i < iniShot; i++) {
    itemp = fscanf(relFile, "%d", &tmpShot);
    itemp = fscanf(relFile, "%d%*[^\n]%*c", &yy);
  }
  itemp = fscanf(relFile, "%d", &tmpShot);
  for (int iShot = 0; iShot < my_data->nShots; iShot++) {

    shotList[iShot].id = tmpShot;
    // Configuracion del tiro
    itemp = fscanf(relFile, "%d", &shotList[iShot].nUsedRecLines);
    itemp = fscanf(relFile, "%d", &shotList[iShot].nUsedRecs);
    itemp = fscanf(relFile, "%s", usedRecsBySource);

    // Realizar el tiro
    printf("Calculos del disparo %d del hilo %d\n", tmpShot, my_data->id);

    // Creacion del directorio de los resultados
    sprintf(cmd, "mkdir SCRATCH_%d", tmpShot);
    itemp = system(cmd);
    sprintf(cmd, "mkdir INFO_%d", tmpShot);
    itemp = system(cmd);
    sprintf(cmd, "./SCRATCH_%d/cubo_", tmpShot);

    mulcnt = 0;

    for (yy = 0; yy < nsamp; yy++)
      for (jj = 0; jj < nrecs; jj++)
        multplx[yy][jj] = 0.0f;

    printf("\nVoy a comenzar a realizar el disparo %d\n", tmpShot);

    for (kk = 1; kk <= nt; kk++) {
      iouttest = kk % idsnap;
      if (idsnap == 1) iouttest = 1;
      /*check if snapshot is needed*/
      itrtest = kk % idt;
      if (idt == 1) itrtest = 1;
      /*sample data at 1 st iteration*/
      if (msgScreen > 0 && kk % msgScreen == 1) printf(
          "Disparo %d | Iteracion %d **\n", tmpShot, kk - 1);
      /*iteration # to screen*/

      /*** set pressure (p) matrix for this iteration ***
       *** initialize if first iteration ***/
      if (kk == 1) {
        memset(p[0], 0, volumeSize * sizeof(float));
        memset(p[1], 0, volumeSize * sizeof(float));
        memset(p[2], 0, volumeSize * sizeof(float));

        /*** if point source, assign source function, initial iteration ***/
        if (cuda) {
          gpuMemcpyToDevice(d_p0, p[0], volumeSize * sizeof(float));
          gpuMemcpyToDevice(d_p1, p[1], volumeSize * sizeof(float));
          gpuMemcpyToDevice(d_p2, p[2], volumeSize * sizeof(float));

          gpuImpulse(d_p1, nx, nz, ny, ixsource[tmpShot], izsource[tmpShot],
              iysource[tmpShot], source[0]);
        }
        else {
          p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
              + (iysource[tmpShot]) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
              + (iysource[tmpShot]) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
              + (iysource[tmpShot]) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] + (izsource[tmpShot]) * (nx) + (iysource[tmpShot])
              * (nx) * (nz)] = source[0];

          p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
              + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
              + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
              + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[0];
          p[1][ixsource[tmpShot] + (izsource[tmpShot]) * (nx) + (iysource[tmpShot]
              - 1) * (nx) * (nz)] = source[0];
        }
      }
      else {
        /*** after the first iteration, move pressure array ***/
        if (cuda) {
          float *tmp;
          tmp = d_p0;
          d_p0 = d_p1;
          d_p1 = d_p2;
          d_p2 = tmp;

          memset(p[2], 0, volumeSize * sizeof(float));
        }
        else {
          float *tmp;
          tmp = p[0];
          p[0] = p[1];
          p[1] = p[2];
          p[2] = tmp;

          memset(p[2], 0, volumeSize * sizeof(float));

        }
        /*** add source if appropriate ***
         *** first for the point source ***/
        if (kk <= nsource) {
          if (cuda) {

            gpuImpulse(d_p1, nx, nz, ny, ixsource[tmpShot], izsource[tmpShot],
                iysource[tmpShot], source[kk - 1]);
          }
          else {
            p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
                + (iysource[tmpShot]) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
                    + (iysource[tmpShot]) * (nx) * (nz)];

            p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
                + (iysource[tmpShot]) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
                    + (iysource[tmpShot]) * (nx) * (nz)];

            p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
                + (iysource[tmpShot]) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
                    + (iysource[tmpShot]) * (nx) * (nz)];

            p[1][ixsource[tmpShot] + (izsource[tmpShot]) * (nx) + (iysource[tmpShot])
                * (nx) * (nz)] = source[kk - 1] + p[1][ixsource[tmpShot]
                + (izsource[tmpShot]) * (nx) + (iysource[tmpShot]) * (nx) * (nz)];

            p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
                + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot] - 1) * (nx)
                    + (iysource[tmpShot] - 1) * (nx) * (nz)];

            p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
                + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] + (izsource[tmpShot] - 1) * (nx)
                    + (iysource[tmpShot] - 1) * (nx) * (nz)];

            p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
                + (iysource[tmpShot] - 1) * (nx) * (nz)] = source[kk - 1]
                + p[1][ixsource[tmpShot] - 1 + (izsource[tmpShot]) * (nx)
                    + (iysource[tmpShot] - 1) * (nx) * (nz)];

            p[1][ixsource[tmpShot] + (izsource[tmpShot]) * (nx) + (iysource[tmpShot]
                - 1) * (nx) * (nz)] = source[kk - 1] + p[1][ixsource[tmpShot]
                + (izsource[tmpShot]) * (nx) + (iysource[tmpShot] - 1) * (nx)
                * (nz)];
          }
        }
      }

      /*** spreading wave in the interior of the grid ***/
      if (cuda) {
        gpuFirstOrderFD(d_p2, d_p1, d_p0, d_c, nx, nz, ny);

        gpuBoundaryConditions(d_p2, d_p1, d_p0, d_c, nx, nz, ny, wdamp, wdamp, ddamp);
      }
      else {
        for (yy = 1; yy < (ny) - 1; yy++) {
          for (nn = 1; nn < (nz) - 1; nn++) {
            for (mm = 1; mm < (nx) - 1; mm++) {
              tr1 = (p[1][(mm - 1) + nn * (nx) + yy * (nx) * (nz)] - 2
                  * p[1][mm + nn * (nx) + yy * (nx) * (nz)] + p[1][mm + 1 + nn
                  * (nx) + yy * (nx) * (nz)]) / dx2;

              tr2 = (p[1][mm + nn * (nx) + (yy - 1) * (nx) * (nz)] - 2
                  * p[1][mm + nn * (nx) + yy * (nx) * (nz)] + p[1][mm + nn
                  * (nx) + (yy + 1) * (nx) * (nz)]) / dy2;

              tr3 = (p[1][mm + (nn - 1) * (nx) + yy * (nx) * (nz)] - 2
                  * p[1][mm + nn * (nx) + yy * (nx) * (nz)] + p[1][mm
                  + (nn + 1) * (nx) + yy * (nx) * (nz)]) / dz2;

              tr4 = pow(c[nn][mm][yy], 2.0f) * dt2 * (tr1 + tr2 + tr3);

              p[2][mm + nn * (nx) + yy * (nx) * (nz)] = tr4 + 2.0f * p[1][mm
                  + nn * (nx) + yy * (nx) * (nz)] - p[0][mm + nn * (nx) + yy
                  * (nx) * (nz)];
            }
          }
        }

        /** BC para x=0 y x=xmax */
        for (yy = 1; yy < ny - 1; yy++)
          for (nn = 1; nn < nz - 1; nn++) {
            /*Left and Right damping zone*/
            for (mm = wdamp - 1; mm >= 1; mm--) {
              /*X=0*/
              p[2][mm + nn * (nx) + yy * (nx) * (nz)] = p[2][mm + nn * (nx)
                  + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - mm), 2.0f));
              p[1][mm + nn * (nx) + yy * (nx) * (nz)] = p[1][mm + nn * (nx)
                  + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - mm), 2.0f));
              /*X=xmax*/
              p[2][nx - (mm + 1) + nn * (nx) + yy * (nx) * (nz)] = p[2][nx
                  - (mm + 1) + nn * (nx) + yy * (nx) * (nz)] * exp(-pow(0.015f
                  * float(wdamp - 1 - mm), 2.0f));
              p[1][nx - (mm + 1) + nn * (nx) + yy * (nx) * (nz)] = p[1][nx
                  - (mm + 1) + nn * (nx) + yy * (nx) * (nz)] * exp(-pow(0.015f
                  * float(wdamp - 1 - mm), 2.0f));
            }

          }

        /** BC para z=0 y z=zmax */
        for (yy = 1; yy < ny - 1; yy++)
          for (mm = 1; mm < nx - 1; mm++) {
            /*Down Damping zone*/
            for (nn = ddamp - 1; nn >= 1; nn--) {
              /*Z=zmax*/
              p[2][mm + (nz - (nn + 1)) * (nx) + yy * (nx) * (nz)] = p[2][mm
                  + (nz - (nn + 1)) * (nx) + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(ddamp - 1 - nn), 2.0f));
              p[1][mm + (nz - (nn + 1)) * (nx) + yy * (nx) * (nz)] = p[1][mm
                  + (nz - (nn + 1)) * (nx) + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(ddamp - 1 - nn), 2.0f));
            }
            /*Boundary Conditions -- Second Order*/
            /*Z=0*/
            p[2][mm + (0) * (nx) + yy * (nx) * (nz)] = 0.0f; //FREE SURFACE
          }

        /** BC para y=0 y y=ymax */
        for (mm = 1; mm < nx - 1; mm++)
          for (nn = 1; nn < nz - 1; nn++) {
            /*Front and Back Damping zone*/
            for (yy = wdamp - 1; yy >= 1; yy--) {
              //Y=0
              p[2][mm + nn * (nx) + yy * (nx) * (nz)] = p[2][mm + nn * (nx)
                  + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - yy), 2.0f));
              p[1][mm + nn * (nx) + yy * (nx) * (nz)] = p[1][mm + nn * (nx)
                  + yy * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - yy), 2.0f));
              //Y=xmax
              p[2][mm + nn * (nx) + (ny - (yy + 1)) * (nx) * (nz)] = p[2][mm
                  + nn * (nx) + (ny - (yy + 1)) * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - yy), 2.0f));
              p[1][mm + nn * (nx) + (ny - (yy + 1)) * (nx) * (nz)] = p[1][mm
                  + nn * (nx) + (ny - (yy + 1)) * (nx) * (nz)] * exp(-pow(
                  0.015f * float(wdamp - 1 - yy), 2.0f));
            }

          }

      }
      /****************************************************************
       write out snapshots if needed
       ****************************************************************/
      /*** first see if snapshots are needed at all ***/
      /*** note: if idsnap=-1 then no snapshots requested so ***/
      if (idsnap > 0) {
        /*** check if snapshot is needed for this iteration ***/
        if (iouttest == 1 || kk == nt) {
          /*** write nx-nz grid to snapshot if desired ***/
          //numsnaps
          sprintf(snapshots, "%s%d.snp", cmd, kk);
          cubo = fopen(snapshots, "wb");

          if (cubo == NULL) {
            printf("Error en la lectura del archivo cubo.*......\n");
          }
          // Sobre el mallado

          if (cuda) {
            gpuMemcpyToHost(p[2], d_p2, volumeSize * sizeof(float));
            for (yy = 0; yy < ny; yy++)
              for (ii = 0; ii < nx; ii++)
                for (jj = 0; jj < nz; jj++) {
                  mul = p[2][ii + jj * (nx) + yy * (nx) * (nz)];
                  fwrite(&mul, sizeof(float), 1, cubo);
                }
          }
          else {
            for (yy = 0; yy < ny; yy++)
              for (ii = 0; ii < nx; ii++)
                for (jj = 0; jj < nz; jj++) {
                  mul = p[2][ii + jj * (nx) + yy * (nx) * (nz)];
                  fwrite(&mul, sizeof(float), 1, cubo);
                }
          }
          fclose( cubo);
          printf("Escrito archivo de la pelicula: %s\n", snapshots);
        }
      }
      /****************************************************************
       write data for the given sample interval (#dt*idt) for all receivers locations.
       (this intermediate data is in a multiplexed format.)
       ****************************************************************/

      /*** first check to see if iteration should be sampled ***/
      if (itrtest == 1) {
        if (cuda) {
          gpuMemcpyToHost(p[2], d_p2, volumeSize * sizeof(float));

          for (yy = 0; yy < nrecs; yy++) {
            if (usedRecsBySource[yy] == '1') multplx[mulcnt][yy]
                = p[2][ixrec[yy] + (izrec[yy]) * (nx) + (iyrec[yy]) * (nx)
                    * (nz)];
            else
              multplx[mulcnt][yy] = 0.0f; // No ha hay grabacion de los datos
          }
          ++mulcnt;
        }
        else {
          for (yy = 0; yy < nrecs; yy++) {
            if (usedRecsBySource[yy] == '1') multplx[mulcnt][yy]
                = p[2][ixrec[yy] + (izrec[yy]) * (nx) + (iyrec[yy]) * (nx)
                    * (nz)];
            else
              multplx[mulcnt][yy] = 0.0f; // No ha hay grabacion de los datos
          }
          ++mulcnt;
        }
      }
    }/*end for kk*/

    printf("\n");
    printf("Desmultiplexado...\n");
    printf("\n");

    sprintf(cmd, "./SCRATCH_%d/receiversLines.txt", tmpShot);
    salida = fopen(cmd, "w");
    if (salida == NULL) {
      printf("Error: Fallo la escritura del archivo receiversLines.txt.\n");

    }

    fprintf(salida, "%d\n", nsamp); // Cantidad de muestras
    fprintf(salida, "%d\n", nLinesR); // Cantidad de lineas de receptores
    for (ii = 0; ii < nLinesR; ++ii)
      fprintf(salida, "%d\n", nRecsByLine[ii]); // Cantidad de receptores por lineas ii-esima
    fclose( salida);

    // Escritura del sismograma
    sprintf(cmd, "./SCRATCH_%d/sismo.out", tmpShot);
    salida = fopen(cmd, "wb");
    if (salida == NULL) {
      printf("Error  en la escritura del archivo sismo.out......\n");
;
    }

    /*** write demultiplexed trace data ***/
    for (jj = 0; jj < nrecs; ++jj)
      for (ii = 0; ii < nsamp; ++ii) {
        mul = (float) multplx[ii][jj] * 100000;
        fwrite(&mul, sizeof(float), 1, salida);
      }
    fclose(salida);
    /*** close the output files ***/

    // Escritura de la resolucion del sismograma (cabecera en archivo separado)
    sprintf(cmd, "./INFO_%d/sismoHeader.txt", tmpShot);
    salida = fopen(cmd, "w");
    if (salida == NULL) {
      printf("Error en la escritura del archivo sismoHeader.txt.\n");

    }

    fprintf(salida, "SimTime(msec): %.6f\n", time1); // Tiempo de simulacion
    fprintf(salida, "nSamples: %d\n", nsamp); // Cantidad de muestras
    fprintf(salida, "nReceivers: %d\n", nrecs); // Cantidad de receptores
    fclose(salida);
    // Fin de escritura

    printf("\n");
    printf(" Sismograma completado\n");
    /*** check if snapshots were requested ***/
    if (idsnap > 0) printf("Imagenes del disparo %d guardada...\n", tmpShot); // save snapshot file

    if (idsnap > 0) {
      sprintf(cmd, "./INFO_%d/info.dat", tmpShot);
      auxfile = fopen(cmd, "w");
      if (auxfile == NULL) {
        printf("Error: No se puede crear el archivo %s\n", cmd);

      }

      // Sobre el mallado
      fprintf(auxfile, "xmin: %.6f xmax: %.6f\n", xo, xf); // Limites del cubo en X (mts)
      fprintf(auxfile, "ymin: %.6f ymax: %.6f\n", yo, yf); // Limites del cubo en Y (mts)
      fprintf(auxfile, "depth: %.6f\n", depth); // Profundidad (mts)
      fprintf(auxfile, "nx: %d\n", nx); // Resolucion en X
      fprintf(auxfile, "ny: %d\n", ny); // Resolucion en Y
      fprintf(auxfile, "nz: %d\n", nz); // Resolucion en Z
      fprintf(auxfile, "ncubes: %d\n", numsnaps + 1); // Cantidad de cubos
      fprintf(auxfile, "offsetX: %.6f\n", dx); // Offset en X (mts)
      fprintf(auxfile, "offsetY: %.6f\n", dy); // Offset en Y (mts)
      fprintf(auxfile, "offsetZ: %.6f\n", dz); // Offset en Z (mts)

      kk = 0;
      for (ii = 0; ii < numsnaps + 1; ii++) {
        sprintf(snapshots, "./SCRATCH_%d/cubo_", tmpShot);
        if (ii == 0) kk = 1;
        else if (ii < numsnaps) kk = kk + idsnap;
        else
          kk = nt;
        sprintf(str, "%d.snp", kk);
        strcat(snapshots, str);
        fprintf(auxfile, "%s\n", snapshots);
      }
      fclose( auxfile);
    }

    // archivo con el orden de los receptores en el disparo i-esimo
    sprintf(cmd, "./INFO_%d/recNoUsados.txt", tmpShot);
    fileRecsNoUsed = fopen(cmd, "w");
    if (fileRecsNoUsed == NULL) {
      printf("Error, Fallo la escritura del archivo %s ......\n", cmd);

    }
    jj = 0;
    fprintf(fileRecsNoUsed,
        "                                 | Cantidad de receptores no usados\n");
    fprintf(fileRecsNoUsed,
        "### Identificadores de los receptores no usados ###\n");
    for (ii = 0; ii < nrecs; ++ii) {
      if (usedRecsBySource[ii] == '0') {
        fprintf(fileRecsNoUsed, "%d\n", (ii + 1));
        ++jj;
      }
    }
    fseek(fileRecsNoUsed, 0, SEEK_SET);
    fprintf(fileRecsNoUsed, "%d", jj);
    fclose( fileRecsNoUsed);
    // Fin del disparo
    tmpShot++;
  }

  if (cuda) {
    gpuFree(d_p0);
    gpuFree(d_p1);
    gpuFree(d_p2);
    gpuFree(d_c);
  }

  pthread_exit(0);
}

int main(int argc, char **argv) {
  char velmodel[200], snapshots[100], str[100];
  char receiveFile[200], sourceFile[200], relationalFile[200];
  FILE *input = NULL, *fsource = NULL, *recep = NULL, *salida = NULL, *cubo =
      NULL, *auxfile = NULL, *velfile = NULL, *fileOfSources = NULL, *relFile =
      NULL, *infoSeismogramFile = NULL;
  int nt, idt, idsnap, nsource, ts;
  int nsamp, nrecs, maxrec, numsnaps, mulcnt, itrtest, iouttest, itemp;
  size_t stemp;
  int *ixrec = NULL, *izrec = NULL, *iyrec = NULL;
  int nx, ny, nz;
  float mul, hzval, dx, dy, dz, time, dt, dt1, velmax, velmin, sampi, time1;
  float dx2, dy2, dz2, dt2, dtdx, dtdy, dtdz;
  float xo, yo, xf, yf, depth;
  float totalrec;
  int ii, jj, kk, yy, mm, nn, offset1, offset2;
  float *source = NULL, ***c = NULL, **p = NULL, **multplx = NULL;
  float tr1, tr2, tr3, tr4, fxsource, fysource;
  int wdamp = 20, ddamp = 20;
  float fidt;
  int nShots, iShot, tmpShot = 0;
  int *ixsource = NULL, *iysource = NULL, *izsource = NULL, *nRecsByLine = NULL;
  char cmd[200];
  int msgScreen; // Mensajes por pantalla
  float tmpDepth;
  int nLinesR; // Cantidad de lineas receptoras
  bool shotOk;
  unsigned char bLE, bLE2;
  char swap;
  char *usedRecsBySource = NULL;
  cDataShot *usedShots = NULL;
  FILE *fileRecsNoUsed = NULL;
  int idxShot, nUsedShots;
  // Variables de CUDA
  int cuda, numDevices;
  unsigned int volumeSize;
  float *d_p0, *d_p1, *d_p2, *d_c, *h_c;
  SimParams h_params;
  ThreadData tData[8];
  pthread_t tid[32];

  if (argc != 3) {
    printf("\n\tseis3D_2orden input cuda\n");
    printf("\n\tseis3D_2orden es un propagador de ondas 3D de 2do Orden.\n");
    printf(
        "\n\tinput\t Archivo de parametros de la propagacion. (input.in)\n\n");
    printf("\n\tcuda\t Indica si se usara CUDA (1) o no (0)\n\n");
    return 1;
  }

  // Determinacion del ordenamiento de los bytes en la arquitectura actual
  ii = cSwapBytes::isSystemBigEndian(); // (1 Big-Endian, 0 Little Endian)
  if (ii < 0) {
    printf(
        "Error: No se puede determinar el ordenamiento de los bytes en la arquitectura actual.\n");
    return 1;
  }
  bLE = (unsigned char) ii;

  printf("**********************************************************\n");
  printf("Algoritmo 3D de 2do orden de diferencias finitas que \n");
  printf("aproxima la solucion de la ecuacion escalar de onda \n");
  printf("\n");
  printf("*********************************************************\n");

  /****************************************************************
   get info. from the user
   *****************************************************************/
  printf("\nArchivo de parametros: %s\n", argv[1]);
  input = fopen(argv[1], "r");
  if (input == NULL) {
    printf("No se puede abrir el archivo %s......\n", argv[1]);
    return 1;
  }

  cuda = atoi(argv[2]);

  /****************************************************************
   read  in models
   ****************************************************************/
  itemp = fscanf(input, "%f%*[^\n]%*c", &time1); // Tiempo de simulacion (mseg)
  itemp = fscanf(input, "%f%*[^\n]%*c", &dt1); // Dt (mseg)
  itemp = fscanf(input, "%f%*[^\n]%*c", &hzval); // Frecuencia (Hz)
  itemp = fscanf(input, "%d%*[^\n]%*c", &ts); // Tipo de ondicula para los emisores

  itemp = fscanf(input, "%d%*[^\n]%*c", &idsnap); // Iteraciones entre snapshots
  itemp = fscanf(input, "%f%*[^\n]%*c", &fidt); // Tiempo del sismograma por cada iteracion (Multiplo de dt)

  // Ruta del archivo de velocidades del modelo 3D
  itemp = fscanf(input, "%[^\n]%*c", velmodel);
  itemp = fscanf(input, "%[^\n]%*c", velmodel);

  // Ruta del archivo de emisores
  itemp = fscanf(input, "%[^\n]%*c", sourceFile);
  itemp = fscanf(input, "%[^\n]%*c", sourceFile);

  // Ruta del archivo de receptores
  itemp = fscanf(input, "%[^\n]%*c", receiveFile);
  itemp = fscanf(input, "%[^\n]%*c", receiveFile);

  // Ruta del archivo relacional
  itemp = fscanf(input, "%[^\n]%*c", relationalFile);
  itemp = fscanf(input, "%[^\n]%*c", relationalFile);

  itemp = fscanf(input, "%d%*[^\n]%*c", &msgScreen); // Mensajes por pantalla
  fclose(input);

  //////////////////////////////////
  printf("Tiempo Sim: %.6f\n", time1);
  printf("Dt: %.6f\n", dt1);
  printf("Frec: %.6f\n", hzval);
  printf("Tipo de Ondicula: %d\n", ts);
  printf("Iteraciones entre snapshots: %d\n", idsnap);
  printf("Tiempo del sismograma por cada iteracion: %.6f\n", fidt);
  printf("Arc. Velocidades: %s\n", velmodel);
  printf("Arc. Emisores: %s\n", sourceFile);
  printf("Arc. Receptores: %s\n", receiveFile);
  printf("Arc. Relacional: %s\n", relationalFile);
  printf("Mensajes en pantalla cada: %d iteraciones\n", msgScreen);
  ///////////////////////////////////////

  // Validacion del Dt y el tiempo entre sismogramas
  if (fidt < dt1 || fmod(fidt, dt1) != 0.0f) {
    printf("Error: La rata de muestreo (sismograma tiene\n"
      "que ser mayor o igual y multiplo de dt (%.6f)", dt1);
    return 1;
  }

  time = 0.001f * time1; // Transformacion de milisegundos a segundos
  dt = 0.001f * dt1; // Transformacion de milisegundos a segundos
  idt = int(fidt / dt1); // Muestras por iteracion


  /*Fill matrix of velocities and density and compute dt*/
  velfile = fopen(velmodel, "rb");
  if (velfile == NULL) {
    printf("Fallo la lectura del archivo %s.......\n", velmodel);
    liberarEstructuras(ixrec, iyrec, izrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, 0, 0, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  /* Lectura del perfil de velocidades */
  stemp = fread(&bLE2, sizeof(unsigned char), 1, velfile); // Big(1)/Little(0) Endian
  swap = bLE == bLE2 ? 'n' : 'y';
  stemp = fread(&xo, sizeof(float), 1, velfile);
  stemp = fread(&yo, sizeof(float), 1, velfile);
  stemp = fread(&xf, sizeof(float), 1, velfile);
  stemp = fread(&yf, sizeof(float), 1, velfile);
  stemp = fread(&depth, sizeof(float), 1, velfile);
  stemp = fread(&dx, sizeof(float), 1, velfile);
  stemp = fread(&dy, sizeof(float), 1, velfile);
  stemp = fread(&dz, sizeof(float), 1, velfile);
  stemp = fread(&nx, sizeof(int), 1, velfile);
  stemp = fread(&ny, sizeof(int), 1, velfile);
  stemp = fread(&nz, sizeof(int), 1, velfile);

  volumeSize = nx * ny * nz;

  if (swap == 'y') {
    cSwapBytes::swapFloat4Bytes(&xo);
    cSwapBytes::swapFloat4Bytes(&yo);
    cSwapBytes::swapFloat4Bytes(&xf);
    cSwapBytes::swapFloat4Bytes(&yf);
    cSwapBytes::swapFloat4Bytes(&depth);
    cSwapBytes::swapFloat4Bytes(&dx);
    cSwapBytes::swapFloat4Bytes(&dy);
    cSwapBytes::swapFloat4Bytes(&dz);
    cSwapBytes::swapInt4Bytes(&nx);
    cSwapBytes::swapInt4Bytes(&ny);
    cSwapBytes::swapInt4Bytes(&nz);
  }

  c = (float***) malloc((nz) * sizeof(float**));

  if (c == NULL) {
    printf("Error en memoria 1\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  for (ii = 0; ii < (nz); ii++) {
    c[ii] = (float**) malloc((nx) * sizeof(float*));
    if (c[ii] == NULL) {
      printf("Error en memoria 2\n");
      return 1;
    }
    for (jj = 0; jj < (nx); jj++) {
      c[ii][jj] = (float*) malloc((ny) * sizeof(float));
      if (c[ii][jj] == NULL) {
        printf("Error en memoria 3\n");
        return 1;
      }
    }
  }

  // Primera velocidad
  stemp = fread(&c[0][0][0], sizeof(float), 1, velfile);
  if (swap == 'y') cSwapBytes::swapFloat4Bytes(&c[0][0][0]);

  if (cuda) {
    velmin = c[0][0][0];
    velmax = c[0][0][0];
    for (kk = 0; kk < (ny); kk++)
      for (jj = 0; jj < (nx); jj++)
        for (ii = !(kk == 0 && jj == 0) ? 0 : 1; ii < (nz); ii++) {
          stemp = fread(&c[ii][jj][kk], sizeof(float), 1, velfile);
          if (swap == 'y') cSwapBytes::swapFloat4Bytes(&c[ii][jj][kk]);

          if (c[ii][jj][kk] > velmax) velmax = c[ii][jj][kk];
          if (c[ii][jj][kk] < velmin) velmin = c[ii][jj][kk];
        }
  }

  else {
    velmin = c[0][0][0];
    velmax = c[0][0][0];
    for (kk = 0; kk < (ny); kk++)
      for (jj = 0; jj < (nx); jj++)
        for (ii = !(kk == 0 && jj == 0) ? 0 : 1; ii < (nz); ii++) {
          stemp = fread(&c[ii][jj][kk], sizeof(float), 1, velfile);
          if (swap == 'y') cSwapBytes::swapFloat4Bytes(&c[ii][jj][kk]);

          if (c[ii][jj][kk] > velmax) velmax = c[ii][jj][kk];
          if (c[ii][jj][kk] < velmin) velmin = c[ii][jj][kk];
        }
  }
  fclose(velfile);

  //printf("idt: %d\n", idt);
  printf("Velmin: %.6f m/s | Velmax: %.6f m/s\n", velmin, velmax);
  ///////////////////////////////////////////

  if (velmax == 0.0f && velmin == 0.0f) {
    printf("Error. El modelo tiene algunas velocidades iguales a cero.\n");
    return 1;
  }

  // Lectura del archivo de emisores
  fileOfSources = fopen(sourceFile, "r");
  if (fileOfSources == NULL) {
    printf("Fallo la lectura del archivo %s.......\n", sourceFile);
    return 1;
  }

  // Lectura del primer disparo (Solo se lee el primer disparo)
  itemp = fscanf(fileOfSources, "%f%*[^\n]%*c", &tmpDepth); // Profundidad de los emisores
  itemp = fscanf(fileOfSources, "%d%*[^\n]%*c", &nShots); // Cantidad de emisores
  itemp = fscanf(fileOfSources, "%d%*[^\n]%*c", &jj); // Cantidad de lineas
  for (ii = 0; ii < jj; ++ii) {
    itemp = fscanf(fileOfSources, "%d%*[^\n]%*c", &kk); // Cantidad de receptores por linea
  }

  itemp = fscanf(fileOfSources, "%[^\n]%*c", str); // Comentario
  itemp = fscanf(fileOfSources, "%[^\n]%*c", str); // Comentario

  ixsource = (int *) malloc(nShots * sizeof(int));
  if (ixsource == NULL) {
    printf("Error en memoria. ixsource no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  iysource = (int *) malloc(nShots * sizeof(int));
  if (iysource == NULL) {
    printf("Error en memoria. iysource no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  izsource = (int *) malloc(nShots * sizeof(int));
  if (izsource == NULL) {
    printf("Error en memoria. izsource no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  // validacion de la profundidad de los emisores
  if (tmpDepth < dz) {
    printf(
        "La profundidad de los emisores (%.6f) tiene que ser mayor que %.6f\n",
        tmpDepth, dz);
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  tmpDepth /= dz; // Discretizacion en Z

  // Lectura de los emisores
  for (ii = 0; ii < nShots; ++ii) {
    itemp = fscanf(fileOfSources, "%d", &kk);
    itemp = fscanf(fileOfSources, "%f", &fxsource);
    itemp = fscanf(fileOfSources, "%f", &fysource);

    ixsource[ii] = (int) (fxsource / dx);
    iysource[ii] = (int) (fysource / dy);
    izsource[ii] = (int) (tmpDepth);

    if (fxsource < dx) {
      printf("La ubicacion de un receptor en X tiene que ser mayor a %.6f\n",
          dx);
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
          usedShots);
      return 1;
    }

    if (fysource < dy) {
      printf("La ubicacion de un receptor en Y tiene que ser mayor a %.6f\n",
          dy);
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
          usedShots);
      return 1;
    }
  }
  fclose(fileOfSources);

  printf("Leido modelo de velocidad\n");

  tr1 = 1.0f / (dx * dx) + 1.0f / (dy * dy) + 1.0f / (dz * dz);
  tr2 = (sqrt(3.0f) * sqrt(3.0f / tr1)) / (3.0f * velmax);
  tr3 = 1.0f / (15.04f * hzval);

  if (tr2 < tr3) tr4 = tr2;
  else
    tr4 = tr3;

  if (dt > tr4) {
    printf(
        "No se cumple la condición de estabilidad. El maximo dt es %f y el utilizado es %f \n",
        tr4, dt);
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  if (dx > dy) if (dx > dz) tr1 = dx;
  else
    tr1 = dz;
  else if (dy > dz) tr1 = dy;
  else
    tr1 = dz;

  tr2 = velmin / (15.04f * hzval);

  if (tr1 > tr2) {
    printf(
        "No se cumple la condición de dispersion. El maximo dx es %f y el utilizado es %f \n",
        tr1, tr2);
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  printf("time = %lf\n", time);
  nt = int(ceil((time / dt)));
  printf("\nnt, nx, ny, nz, dx, dy, dz = %d %d %d %d %f %f %f\n", nt, nx, ny,
      nz, dx, dy, dz);
  if (nt % idt == 0) nsamp = nt / idt;
  else
    nsamp = nt / idt + 1;
  sampi = dt * float(idt * 1000);

  // Lectura del archivo de receptores
  recep = fopen(receiveFile, "r");
  if (recep == NULL) {
    printf("Error, Fallo lectura del archivo %s ......\n", receiveFile);
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  itemp = fscanf(recep, "%f%*[^\n]%*c", &tmpDepth); // Profundidad de los receptores
  itemp = fscanf(recep, "%d%*[^\n]%*c", &nrecs); // Cantidad total de receptores
  itemp = fscanf(recep, "%d%*[^\n]%*c", &nLinesR); // Cantidad de lineas

  // Creacion e inicializacion de los subconjuntos de emisores y receptores
  usedRecsBySource = new char[nrecs];
  if (usedRecsBySource == NULL) {
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    printf("Error en memoria. usedRecsBySource no puede ser creado.\n");
    return 1;
  }

  nRecsByLine = (int *) malloc(nLinesR * sizeof(int));
  if (nRecsByLine == NULL) {
    printf("Error en memoria. nRecsByLine no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  for (ii = 0; ii < nLinesR; ++ii) {
    itemp = fscanf(recep, "%d%*[^\n]%*c", &nRecsByLine[ii]); // Cantidad de receptores de la linea i-esima
    //printf("# receptores por linea %d: %d\n", ii, nRecsByLine[ii]);
  }

  itemp = fscanf(recep, "%[^\n]%*c", receiveFile); // Comentario
  //printf("%s\n", receiveFile);
  itemp = fscanf(recep, "%[^\n]%*c", receiveFile); // Comentario
  //printf("%s\n", receiveFile);

  ixrec = (int *) malloc(nrecs * sizeof(int));
  if (ixrec == NULL) {
    printf("Error en memoria. ixrec no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  izrec = (int *) malloc(nrecs * sizeof(int));
  if (izrec == NULL) {
    printf("Error en memoria. izrec no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  iyrec = (int *) malloc(nrecs * sizeof(int));
  if (iyrec == NULL) {
    printf("Error en memoria. iyrec no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  printf("Numero de celdas en la direccion X = %d\n", nx);
  printf("Numero de celdas en la direccion Y = %d\n", ny);
  printf("Cantidad total de fuentes = %d\n", nShots);
  printf("Numero de lineas receptoras = %d\n", nLinesR);
  printf("Cantidad total de receptores = %d\n", nrecs);
  printf("Numero de muestras = %d\n", nsamp);
  printf("Intervalo de muestreo = %f milisegundos\n", sampi);

  /*Lectura de receptores horizontal*/
  if (tmpDepth < dz) {
    printf(
        "La profundidad de los receptores (%.6f) tiene que ser mayor que %.6f\n",
        tmpDepth, dz);
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  tmpDepth /= dz; // Discretizacion en Z

  for (ii = 0; ii < nrecs; ii++) {
    itemp = fscanf(recep, " %d %f %f", &jj, &tr1, &tr2);
    if (tr1 < dx) {
      printf("La ubicacion de un receptor en X tiene que ser mayor a %.6f\n",
          dx);
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
          usedShots);
      return 1;
    }

    if (tr2 < dy) {
      printf("La ubicacion de un receptor en Y tiene que ser mayor a %.6f\n",
          dy);
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
          usedShots);
      return 1;
    }

    ixrec[ii] = int(tr1 / dx);
    iyrec[ii] = int(tr2 / dy);
    izrec[ii] = int(tmpDepth);
  }
  fclose(recep);
  printf("ixrec, iyrec y izrec OK ....\n");

  /***read in source function, velocity (c) and density (rho) models***/
  fsource = fopen("fsource", "w");
  if (fsource == NULL) {
    printf("Error fsource file open......\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  switch (ts) {
    case 1:
      if (!ricker(hzval, dt, fsource)) {
        printf("Error en la funcion de ricker......\n");
        liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
            source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
            usedShots);
        return 1;
      }
    break;
    case 2:
      if (!gaussian(hzval, dt, nt, fsource)) {
        printf("Error en la funcion gaussiana......\n");
        liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
            source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
            usedShots);
        return 1;
      }
    break;
  }
  fclose(fsource);

  fsource = fopen("fsource", "r");
  if (fsource == NULL) {
    printf("Error en la lectura del archivo fsource......\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }

  itemp = fscanf(fsource, "%d %d", &nsource, &ii);
  source = (float *) malloc(nsource * sizeof(float));
  if (source == NULL) {
    printf("Error en memoria. La fuente no pudo ser asignada.\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  for (ii = 0; ii < nsource; ii++)
    itemp = fscanf(fsource, "%f", &source[ii]);

  p = (float**) malloc(3 * sizeof(float*));
  if (p == NULL) {
    printf("Error en memoria 9\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 0, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  for (ii = 0; ii < 3; ii++) {
    kk = (nx) * (ny) * (nz);
    p[ii] = (float *) malloc((int) kk * sizeof(float));
    if (p[ii] == NULL) {
      printf("Error en memoria 10\n");
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 3, c, nz, nx, multplx, 0, usedRecsBySource,
          usedShots);
      return 1;
    }
  }

  multplx = (float**) malloc(nsamp * sizeof(float*));
  if (multplx == NULL) {
    printf("Error en memoria 13\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 3, c, nz, nx, multplx, 0, usedRecsBySource,
        usedShots);
    return 1;
  }
  for (ii = 0; ii < nsamp; ii++) {
    multplx[ii] = (float*) malloc(nrecs * sizeof(float));
    if (multplx[ii] == NULL) {
      printf("Error en memoria 14\n");
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 3, c, nz, nx, multplx, nsamp,
          usedRecsBySource, usedShots);
      return 1;
    }
  }

  /***close input files***/
  fclose(fsource);

  dx2 = (float) (dx * dx);
  dy2 = (float) (dy * dy);
  dz2 = (float) (dz * dz);
  dt2 = (float) (dt * dt);
  dtdx = (float) (dt / dx);
  dtdy = (float) (dt / dy);
  dtdz = (float) (dt / dz);

  /*** ask user if snapshots are to contain the entire grid ***
   *** or only the x-locations occupied by receivers ***/
  if (idsnap > 0) printf("Activada la grabacion de la pelicula\n");

  /***set misc. parameters (used in for snapshots mostly) ***/
  totalrec = (float) nt * dt * 1000.0f; //record length (msec)
  maxrec = (int) (totalrec + 0.05f); //rounded record length (msec)
  if (idsnap > 0) numsnaps = nt / idsnap; // # of snapshots
  else
    numsnaps = 0;
  printf("numsnaps=%d\n", numsnaps);

  /****************************************************************
   /     begin the main loop
   ****************************************************************/

  if (cuda) {
    /* Reservar memoria CUDA */
    h_c = (float*) malloc((int) volumeSize * sizeof(float));

    int gg = 0;
    for (ii = 0; ii < (ny); ii++) {
      for (kk = 0; kk < (nz); kk++) {
        for (jj = 0; jj < (nx); jj++) {
          h_c[gg] = c[kk][jj][ii];
          gg++;
        }
      }
    }

    h_params.dt2 = dt2;
    h_params.dx2 = dx2;
    h_params.dy2 = dy2;
    h_params.dz2 = dz2;

    gpuGetNumDevices(numDevices);
  }

  /*
   * Determinando la cantidad de tiros que necesitan ser procesados.
   *
   */

  relFile = fopen(relationalFile, "r");
  if (!relFile) {
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 3, c, nz, nx, multplx, nsamp, usedRecsBySource,
        usedShots);
    return (1);
  }
  itemp = fscanf(relFile, "%d", &nUsedShots);
  fclose(relFile);

  usedShots = new cDataShot[nUsedShots];

  /**************************************************************************/
  /*
   /*
   /*
   /* Aqui va el codigo para preparar los hilos y mandarles la informacion
   /* ademas de la ejecucion y la espera de que terminen.
   /*
   /*
   /*
   /*
   /***************************************************************************/

  for (int i = 0; i < numDevices; i++) {
    tData[i].nShots = nUsedShots / numDevices;
  }
  for (int i = 0; i < nUsedShots % numDevices; i++) {
    tData[i].nShots++;
  }
  int gpuBase = 0;
  for (int i = 0; i < numDevices; i++) {
    tData[i].device = i;
    tData[i].nsamp = nsamp;
    tData[i].nrecs = nrecs;
    tData[i].nx = nx;
    tData[i].ny = ny;
    tData[i].nz = nz;
    tData[i].nt = nt;
    tData[i].usedShots = usedShots + gpuBase; // Apuntador a clase de tipo cDataShot
    tData[i].iniShot = gpuBase;
    gpuBase += tData[i].nShots;
    tData[i].id = i;
    tData[i].idt = idt;
    tData[i].cuda = cuda;
    tData[i].ixsource = ixsource;
    tData[i].iysource = iysource;
    tData[i].izsource = izsource;
    tData[i].source = source;
    tData[i].nSource = nsource;
    tData[i].nLinesR = nLinesR;
    tData[i].ixrec = ixrec;
    tData[i].iyrec = iyrec;
    tData[i].izrec = izrec;
    tData[i].nRecsByLine = nRecsByLine;
    tData[i].volumeSize = volumeSize;
    tData[i].c = c;
    tData[i].h_c = h_c;
    tData[i].xo = xo;
    tData[i].xf = xf;
    tData[i].yo = yo;
    tData[i].yf = yf;
    tData[i].depth = depth;
    tData[i].numsnaps = numsnaps;
    tData[i].dx = dx;
    tData[i].dy = dy;
    tData[i].dz = dz;
    tData[i].h_params = h_params;
    tData[i].nSource = nsource;
    tData[i].time1 = time1;
    tData[i].ddamp = ddamp;
    tData[i].wdamp = wdamp;
    tData[i].relFile = relationalFile;
    tData[i].idsnap = idsnap;
  }
  for (int i = 0; i < numDevices; i++) {
    pthread_create(&tid[i], 0, hiloGPU, (void*) &tData[i]);
  }
  for (int i = 0; i < numDevices; i++) {
    pthread_join(tid[i], 0);
  }

  /**************************************************************************/
  /*
   /*
   /*
   /* Aquí va el código para preparar los hilos y mandarles la informacion
   /* Además de la ejecución y la espera de que terminen.
   /*
   /*
   /*
   /*
   /***************************************************************************/

  // Escritura del archivo de los sismogramas 3D
  infoSeismogramFile = fopen("info3DSeismogram.s3d", "w");
  if (infoSeismogramFile == NULL) {
    printf("Error en la escritura del archivo info3DSeismogram.s3d\n");
    liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
        source, nRecsByLine, p, 3, c, nz, nx, multplx, nsamp, usedRecsBySource,
        usedShots);
    return 1;
  }

  fprintf(infoSeismogramFile, "#########_Configuracion_##########\n");
  fprintf(infoSeismogramFile, "Total_de_emisores: %d\n", nShots);
  fprintf(infoSeismogramFile, "Total_de_receptores: %d\n", nrecs);
  fprintf(infoSeismogramFile, "Emisores_usados: %d\n", nUsedShots);
  fprintf(infoSeismogramFile, "######_Lista_de_emisores_#########\n");
  fprintf(infoSeismogramFile,
      "Emisor___Cant_Total_Lin_Usadas____Cant_Total_Rec_Usados\n");
  for (idxShot = 0; idxShot < nUsedShots; ++idxShot) {
    fprintf(infoSeismogramFile, "%d ", usedShots[idxShot].id); // Identificador del emisor
    fprintf(infoSeismogramFile, "%d ", usedShots[idxShot].nUsedRecLines); // Cantidad de lineas usadas
    fprintf(infoSeismogramFile, "%d\n", usedShots[idxShot].nUsedRecs); // Cantidad de receptores usados
  }
  fclose(infoSeismogramFile);

  if (idsnap > 0) // Si hay snapshots 3D
  {
    // Archivo con la informacion general sobre la pelicula
    sprintf(cmd, "infoMovie.m3d");
    auxfile = fopen(cmd, "w");
    if (auxfile == NULL) {
      printf("Error: No se puede crear el archivo %s\n", cmd);
      liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource,
          source, nRecsByLine, p, 3, c, nz, nx, multplx, nsamp,
          usedRecsBySource, usedShots);
      return 1;
    }

    // Sobre el mallado
    fprintf(auxfile, "xmin: %.6f xmax: %.6f\n", xo, xf); // Limites del cubo en X (mts)
    fprintf(auxfile, "ymin: %.6f ymax: %.6f\n", yo, yf); // Limites del cubo en Y (mts)
    fprintf(auxfile, "depth: %.6f\n", depth); // Profundidad (mts)
    fprintf(auxfile, "nx: %d\n", nx); // Resolucion en X
    fprintf(auxfile, "ny: %d\n", ny); // Resolucion en Y
    fprintf(auxfile, "nz: %d\n", nz); // Resolucion en Z
    fprintf(auxfile, "ncubes: %d\n", numsnaps + 1); // Cantidad de cubos
    fprintf(auxfile, "offsetX: %.6f\n", dx); // Offset en X (mts)
    fprintf(auxfile, "offsetY: %.6f\n", dy); // Offset en Y (mts)
    fprintf(auxfile, "offsetZ: %.6f\n", dz); // Offset en Z (mts)
    fprintf(auxfile, "total_#shot: %d\n", nShots); // Cantidad de emisores
    fprintf(auxfile, "Realized_#shot: %d\n", nUsedShots); // Cantidad de disparos realizados
    fprintf(auxfile, "Realized_Shot_List:");
    for (idxShot = 0; idxShot < nUsedShots; ++idxShot)
      fprintf(auxfile, " %d", usedShots[idxShot].id);
    fprintf(auxfile, "\n");
    fclose(auxfile);
  }

  liberarEstructuras(ixrec, izrec, iyrec, ixsource, iysource, izsource, source,
      nRecsByLine, p, 3, c, nz, nx, multplx, nsamp, usedRecsBySource, usedShots);
  return (0);
}

/****************************************************************
 subroutine to follow
 ****************************************************************/
bool ricker(float f, float dt, FILE *file) {
  float nw;
  int nc, ii, inw, pos = 0;
  float alpha, beta, *w, max = 0.0f;

  nw = (6.0f / f) / dt;
  //  printf("nw: %.6f\n", nw);
  inw = 2 * int(floor(nw / 2.0f)) + 1;
  nc = int(floor(float(inw) / 2.0f));

  w = (float *) malloc(inw * sizeof(float));
  if (w == NULL) {
    printf("Error en la funcion de emisores | inw: %d...\n", inw);
    return false;
  }
  for (ii = 0; ii < inw; ii++) {
    alpha = float(nc - ii) * f * dt * float(M_PI);
    beta = alpha * alpha;
    w[ii] = (1.0f - beta * 2.0f) * exp(-beta);
    if (fabs(w[ii]) > max) {
      max = fabs(w[ii]);
      pos = ii;
    }
  }
  fprintf(file, "%d %d\n", inw, pos);
  for (ii = 0; ii < inw; ii++) {
    fprintf(file, "%f\n", w[ii]);
  }
  free(w);
  return true;
}

bool gaussian(float freq, float dt, int nt, FILE *file) {
  float dtf, tmp, max, it;
  int ii, pos, ntf;
  float *cc;

  cc = (float*) malloc(nt * sizeof(float));
  if (!cc) {
    printf("Error en la funcion de emisores...\n");
    return false;
  }

  for (ii = 0; ii < nt; ii++)
    cc[ii] = 0.0f;

  max = 0.0f;
  pos = -1;
  dtf = dt;
  tmp = (float) (2.0f * M_PI * freq * dtf * 0.8f);

  ntf = (int) (2.0f / (freq * dt));

  for (ii = 0; ii < ntf; ii++) {
    it = (float) ii;
    cc[ii] = (float) (sin(tmp * it) / (exp(tmp * it * it) * 9.0f));
    if (fabs(cc[ii]) > max) {
      max = fabs(cc[ii]);
      pos = ii;
    }
  }
  fprintf(file, "%d %d\n", ntf, pos);
  for (ii = 0; ii < ntf; ii++) {
    fprintf(file, "%f\n", cc[ii]);
  }
  free(cc);
  return true;
}

void liberarEstructuras(int *ixrec, int *iyrec, int *izrec, int *ixsource,
    int *iysource, int *izsource, float *source, int *nRecsByLine, float **p,
    int np1, float ***c, int nc1, int nc2, float **multplx, int nm1,
    char *usedRecsBySource, cDataShot *usedShots) {
  int i, j;
  if (ixrec != NULL) free(ixrec);
  if (iyrec != NULL) free(iyrec);
  if (izrec != NULL) free(izrec);
  if (source != NULL) free(source);
  if (ixsource != NULL) free(ixsource);
  if (iysource != NULL) free(iysource);
  if (izsource != NULL) free(izsource);
  if (nRecsByLine != NULL) free(nRecsByLine);

  if (p != NULL) {
    for (i = 0; i < np1; ++i)
      if (p[i] != NULL) free(p[i]);
    free(p);
  }

  if (c != NULL) {
    for (i = 0; i < nc1; ++i)
      if (c[i] != NULL) {
        for (j = 0; j < nc2; ++j)
          if (c[i][j] != NULL) free(c[i][j]);
        free(c[i]);
      }
    free(c);
  }

  if (multplx != NULL) {
    for (i = 0; i < nm1; ++i)
      if (multplx[i] != NULL) free(multplx[i]);
    free(multplx);
  }

  if (usedRecsBySource != NULL) delete[] usedRecsBySource;

  if (usedShots != NULL) delete[] usedShots;
}
