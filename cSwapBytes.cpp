/******************************************************************************/
/*  TERRASA - Version 2.0
    Copyright (C) 2009 
                         

    Developed by: Jose Jaramillo (josejonasj@gmail.com)
                  Deybi Exposito (expositodeybi@gmail.com)

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

#include "cSwapBytes.h"

/*
 * Intercambia los bytes de un \c short \c int.
 */

void cSwapBytes::swapShort2Bytes(short *tni2)
{
    *tni2=(((*tni2>>8)&0xff) | ((*tni2&0xff)<<8));
}

/*
 * Intercambia los bytes de un \c unsigned \c short \c int.
 */

void cSwapBytes::swapUshort2Bytes(unsigned short *tni2)
{
    *tni2=(((*tni2>>8)&0xff) | ((*tni2&0xff)<<8));
}

/*
 * Intercambia los bytes de un \c int.
 */

void cSwapBytes::swapInt4Bytes(int *tni4)
{
    *tni4=(((*tni4>>24)&0xff) | ((*tni4&0xff)<<24) |
          ((*tni4>>8)&0xff00) | ((*tni4&0xff00)<<8));
}

/*
 * Intercambia los bytes de un \c unsigned \c int.
 */

void cSwapBytes::swapUint4Bytes(unsigned int *tni4)
{
    *tni4=(((*tni4>>24)&0xff) | ((*tni4&0xff)<<24) |
          ((*tni4>>8)&0xff00) | ((*tni4&0xff00)<<8));
}

/*
 * Intercambia los bytes de un \c long \c int.
 */

void cSwapBytes::swapLong4Bytes(long *tni4)
{
    *tni4=(((*tni4>>24)&0xff) | ((*tni4&0xff)<<24) |
          ((*tni4>>8)&0xff00) | ((*tni4&0xff00)<<8));
}

/*
 * Intercambia los bytes de un \c unsigned \c long \c int.
 */

void cSwapBytes::swapUlong4Bytes(unsigned long *tni4)
{
 *tni4=(((*tni4>>24)&0xff) | ((*tni4&0xff)<<24) |
        ((*tni4>>8)&0xff00) | ((*tni4&0xff00)<<8));
}

/*
 * Intercambia los bytes de un \c float.
 */

void cSwapBytes::swapFloat4Bytes(float *tnf4)
{
 int *tni4=(int *)tnf4;
 *tni4=(((*tni4>>24)&0xff) | ((*tni4&0xff)<<24) |
        ((*tni4>>8)&0xff00) | ((*tni4&0xff00)<<8));
}

/*
 * Intercambia los bytes de un \c double.
 */

void cSwapBytes::swapDouble8Bytes(double *tndd8)
{
  char *tnd8=(char *)tndd8;
  char tnc;

  tnc= *tnd8;
  *tnd8= *(tnd8+7);
  *(tnd8+7)=tnc;

  tnc= *(tnd8+1);
  *(tnd8+1)= *(tnd8+6);
  *(tnd8+6)=tnc;

  tnc= *(tnd8+2);
  *(tnd8+2)= *(tnd8+5);
  *(tnd8+5)=tnc;

  tnc= *(tnd8+3);
  *(tnd8+3)= *(tnd8+4);
  *(tnd8+4)=tnc;
}

int cSwapBytes::isSystemBigEndian()
{
    union
    {
        short s;
        char c[sizeof(short)];
    } un;

    un.s = 0x0102;

    if (sizeof(short) == 2)
    {
        if(un.c[0] == 1 && un.c[1] == 2)
            return 1; // Big Endian
        else if(un.c[0] == 2 && un.c[1] == 1)
            return 0; // Little Endian
        else
            return -1; // Formato desconocido
    }
    else
        return -2; // Error
}
