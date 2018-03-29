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

#ifndef CSWAPBYTES_H
#define CSWAPBYTES_H

/**
 * Contiene funciones que son de utilidad para cambiar el ordenamiento de los
 * bytes de una variable, el cual puede ser de tipo: little-endian o big-endian.
 */

namespace cSwapBytes
{

    /**
     * Intercambia los bytes de un short.
     */
    void swapShort2Bytes(short *tni2);

    /**
     * Intercambia los bytes de un \c unsigned \c short \c int.
     */
    void swapUshort2Bytes(unsigned short *tni2);

    /**
     * Intercambia los bytes de un \c int.
     */
    void swapInt4Bytes(int *tni4);

    /**
     * Intercambia los bytes de un \c unsigned \c int.
     */
    void swapUint4Bytes(unsigned int *tni4);

    /**
     * Intercambia los bytes de un \c long \c int.
     */
    void swapLong4Bytes(long *tni4);

    /**
     * Intercambia los bytes de un \c unsigned \c long \c int.
     */
    void swapUlong4Bytes(unsigned long *tni4);

    /**
     * Intercambia los bytes de un \c float.
     */
    void swapFloat4Bytes(float *tnf4);

    /**
     * Intercambia los bytes de un \c double.
     */
    void swapDouble8Bytes(double *tndd8);

    /**
     * Determina si el sistema en el cual se ejecuta es de tipo big-endian,
     * little-endian o desconocido.
     * @return Valor que indica el tipo de
     * \htmlonly m&aacute;quina, \endhtmlonly
     * \latexonly m\'aquina, \endlatexonly
     * donde: \n
     * 0 = little-endian \n
     * 1 = big-endian \n
     * -1 = desconocido \n
     * -2 = error porque las variables enteras
     * \htmlonly <em> short </em> \endhtmlonly
     * \latexonly \emph{short} \endlatexonly
     * del sistema no son de 2 bytes
     */
    int isSystemBigEndian();
}

#endif // CSWAPBYTE_H
