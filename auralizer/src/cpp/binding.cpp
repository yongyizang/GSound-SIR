#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

namespace py = pybind11;

// order 2
void SHEval2(const float fX, const float fY, const float fZ, float *pSH)
{
    float fC0, fS0, fTmpA;

    pSH[0] = 0.2820947917738781f;

    fC0 = fX;
    fS0 = fY;

    fTmpA = 0.4886025119029199f;
    pSH[1] = fTmpA * fS0;
    pSH[2] = fTmpA * fZ;
    pSH[3] = fTmpA * fC0;
}


// order 3
void SHEval3(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpC = 0.5462742152960395f;
   pSH[8] = fTmpC*fC1;
   pSH[4] = fTmpC*fS1;
}

// order 4
void SHEval4(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpC = -0.5900435899266435f;
   pSH[15] = fTmpC*fC0;
   pSH[9] = fTmpC*fS0;
}

// order 5
void SHEval5(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpC = 0.6258357354491763f;
   pSH[24] = fTmpC*fC1;
   pSH[16] = fTmpC*fS1;
}

// order 6
void SHEval6(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   pSH[30] = 1.98997487421324f*fZ*pSH[20] + -1.002853072844814f*pSH[12];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fTmpB = 2.03100960115899f*fZ*fTmpA + -0.991031208965115f*fTmpC;
   pSH[31] = fTmpB*fC0;
   pSH[29] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fTmpA = fZ*(7.190305177459987f*fZ2 + -2.396768392486662f);
   pSH[32] = fTmpA*fC1;
   pSH[28] = fTmpA*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fTmpC = -4.403144694917254f*fZ2 + 0.4892382994352505f;
   pSH[33] = fTmpC*fC0;
   pSH[27] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6258357354491763f;
   pSH[24] = fTmpA*fC1;
   pSH[16] = fTmpA*fS1;
   fTmpB = 2.075662314881041f*fZ;
   pSH[34] = fTmpB*fC1;
   pSH[26] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpC = -0.6563820568401703f;
   pSH[35] = fTmpC*fC0;
   pSH[25] = fTmpC*fS0;
}

// order 7
void SHEval7(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   pSH[30] = 1.98997487421324f*fZ*pSH[20] + -1.002853072844814f*pSH[12];
   pSH[42] = 1.993043457183567f*fZ*pSH[30] + -1.001542020962219f*pSH[20];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fTmpB = 2.03100960115899f*fZ*fTmpA + -0.991031208965115f*fTmpC;
   pSH[31] = fTmpB*fC0;
   pSH[29] = fTmpB*fS0;
   fTmpC = 2.021314989237028f*fZ*fTmpB + -0.9952267030562385f*fTmpA;
   pSH[43] = fTmpC*fC0;
   pSH[41] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fTmpA = fZ*(7.190305177459987f*fZ2 + -2.396768392486662f);
   pSH[32] = fTmpA*fC1;
   pSH[28] = fTmpA*fS1;
   fTmpB = 2.11394181566097f*fZ*fTmpA + -0.9736101204623268f*fTmpC;
   pSH[44] = fTmpB*fC1;
   pSH[40] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fTmpC = -4.403144694917254f*fZ2 + 0.4892382994352505f;
   pSH[33] = fTmpC*fC0;
   pSH[27] = fTmpC*fS0;
   fTmpA = fZ*(-10.13325785466416f*fZ2 + 2.763615778544771f);
   pSH[45] = fTmpA*fC0;
   pSH[39] = fTmpA*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6258357354491763f;
   pSH[24] = fTmpA*fC1;
   pSH[16] = fTmpA*fS1;
   fTmpB = 2.075662314881041f*fZ;
   pSH[34] = fTmpB*fC1;
   pSH[26] = fTmpB*fS1;
   fTmpC = 5.550213908015966f*fZ2 + -0.5045649007287241f;
   pSH[46] = fTmpC*fC1;
   pSH[38] = fTmpC*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.6563820568401703f;
   pSH[35] = fTmpA*fC0;
   pSH[25] = fTmpA*fS0;
   fTmpB = -2.366619162231753f*fZ;
   pSH[47] = fTmpB*fC0;
   pSH[37] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpC = 0.6831841051919144f;
   pSH[48] = fTmpC*fC1;
   pSH[36] = fTmpC*fS1;
}

// order 8
void SHEval8(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   pSH[30] = 1.98997487421324f*fZ*pSH[20] + -1.002853072844814f*pSH[12];
   pSH[42] = 1.993043457183567f*fZ*pSH[30] + -1.001542020962219f*pSH[20];
   pSH[56] = 1.994891434824135f*fZ*pSH[42] + -1.000927213921958f*pSH[30];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fTmpB = 2.03100960115899f*fZ*fTmpA + -0.991031208965115f*fTmpC;
   pSH[31] = fTmpB*fC0;
   pSH[29] = fTmpB*fS0;
   fTmpC = 2.021314989237028f*fZ*fTmpB + -0.9952267030562385f*fTmpA;
   pSH[43] = fTmpC*fC0;
   pSH[41] = fTmpC*fS0;
   fTmpA = 2.015564437074638f*fZ*fTmpC + -0.9971550440218319f*fTmpB;
   pSH[57] = fTmpA*fC0;
   pSH[55] = fTmpA*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fTmpA = fZ*(7.190305177459987f*fZ2 + -2.396768392486662f);
   pSH[32] = fTmpA*fC1;
   pSH[28] = fTmpA*fS1;
   fTmpB = 2.11394181566097f*fZ*fTmpA + -0.9736101204623268f*fTmpC;
   pSH[44] = fTmpB*fC1;
   pSH[40] = fTmpB*fS1;
   fTmpC = 2.081665999466133f*fZ*fTmpB + -0.9847319278346618f*fTmpA;
   pSH[58] = fTmpC*fC1;
   pSH[54] = fTmpC*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fTmpC = -4.403144694917254f*fZ2 + 0.4892382994352505f;
   pSH[33] = fTmpC*fC0;
   pSH[27] = fTmpC*fS0;
   fTmpA = fZ*(-10.13325785466416f*fZ2 + 2.763615778544771f);
   pSH[45] = fTmpA*fC0;
   pSH[39] = fTmpA*fS0;
   fTmpB = 2.207940216581962f*fZ*fTmpA + -0.959403223600247f*fTmpC;
   pSH[59] = fTmpB*fC0;
   pSH[53] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6258357354491763f;
   pSH[24] = fTmpA*fC1;
   pSH[16] = fTmpA*fS1;
   fTmpB = 2.075662314881041f*fZ;
   pSH[34] = fTmpB*fC1;
   pSH[26] = fTmpB*fS1;
   fTmpC = 5.550213908015966f*fZ2 + -0.5045649007287241f;
   pSH[46] = fTmpC*fC1;
   pSH[38] = fTmpC*fS1;
   fTmpA = fZ*(13.49180504672677f*fZ2 + -3.113493472321562f);
   pSH[60] = fTmpA*fC1;
   pSH[52] = fTmpA*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.6563820568401703f;
   pSH[35] = fTmpA*fC0;
   pSH[25] = fTmpA*fS0;
   fTmpB = -2.366619162231753f*fZ;
   pSH[47] = fTmpB*fC0;
   pSH[37] = fTmpB*fS0;
   fTmpC = -6.745902523363385f*fZ2 + 0.5189155787202604f;
   pSH[61] = fTmpC*fC0;
   pSH[51] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6831841051919144f;
   pSH[48] = fTmpA*fC1;
   pSH[36] = fTmpA*fS1;
   fTmpB = 2.645960661801901f*fZ;
   pSH[62] = fTmpB*fC1;
   pSH[50] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpC = -0.7071627325245963f;
   pSH[63] = fTmpC*fC0;
   pSH[49] = fTmpC*fS0;
}

// order 9
void SHEval9(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   pSH[30] = 1.98997487421324f*fZ*pSH[20] + -1.002853072844814f*pSH[12];
   pSH[42] = 1.993043457183567f*fZ*pSH[30] + -1.001542020962219f*pSH[20];
   pSH[56] = 1.994891434824135f*fZ*pSH[42] + -1.000927213921958f*pSH[30];
   pSH[72] = 1.996089927833914f*fZ*pSH[56] + -1.000600781069515f*pSH[42];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fTmpB = 2.03100960115899f*fZ*fTmpA + -0.991031208965115f*fTmpC;
   pSH[31] = fTmpB*fC0;
   pSH[29] = fTmpB*fS0;
   fTmpC = 2.021314989237028f*fZ*fTmpB + -0.9952267030562385f*fTmpA;
   pSH[43] = fTmpC*fC0;
   pSH[41] = fTmpC*fS0;
   fTmpA = 2.015564437074638f*fZ*fTmpC + -0.9971550440218319f*fTmpB;
   pSH[57] = fTmpA*fC0;
   pSH[55] = fTmpA*fS0;
   fTmpB = 2.011869540407391f*fZ*fTmpA + -0.9981668178901745f*fTmpC;
   pSH[73] = fTmpB*fC0;
   pSH[71] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fTmpA = fZ*(7.190305177459987f*fZ2 + -2.396768392486662f);
   pSH[32] = fTmpA*fC1;
   pSH[28] = fTmpA*fS1;
   fTmpB = 2.11394181566097f*fZ*fTmpA + -0.9736101204623268f*fTmpC;
   pSH[44] = fTmpB*fC1;
   pSH[40] = fTmpB*fS1;
   fTmpC = 2.081665999466133f*fZ*fTmpB + -0.9847319278346618f*fTmpA;
   pSH[58] = fTmpC*fC1;
   pSH[54] = fTmpC*fS1;
   fTmpA = 2.06155281280883f*fZ*fTmpC + -0.9903379376602873f*fTmpB;
   pSH[74] = fTmpA*fC1;
   pSH[70] = fTmpA*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fTmpC = -4.403144694917254f*fZ2 + 0.4892382994352505f;
   pSH[33] = fTmpC*fC0;
   pSH[27] = fTmpC*fS0;
   fTmpA = fZ*(-10.13325785466416f*fZ2 + 2.763615778544771f);
   pSH[45] = fTmpA*fC0;
   pSH[39] = fTmpA*fS0;
   fTmpB = 2.207940216581962f*fZ*fTmpA + -0.959403223600247f*fTmpC;
   pSH[59] = fTmpB*fC0;
   pSH[53] = fTmpB*fS0;
   fTmpC = 2.15322168769582f*fZ*fTmpB + -0.9752173865600178f*fTmpA;
   pSH[75] = fTmpC*fC0;
   pSH[69] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6258357354491763f;
   pSH[24] = fTmpA*fC1;
   pSH[16] = fTmpA*fS1;
   fTmpB = 2.075662314881041f*fZ;
   pSH[34] = fTmpB*fC1;
   pSH[26] = fTmpB*fS1;
   fTmpC = 5.550213908015966f*fZ2 + -0.5045649007287241f;
   pSH[46] = fTmpC*fC1;
   pSH[38] = fTmpC*fS1;
   fTmpA = fZ*(13.49180504672677f*fZ2 + -3.113493472321562f);
   pSH[60] = fTmpA*fC1;
   pSH[52] = fTmpA*fS1;
   fTmpB = 2.304886114323221f*fZ*fTmpA + -0.9481763873554654f*fTmpC;
   pSH[76] = fTmpB*fC1;
   pSH[68] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.6563820568401703f;
   pSH[35] = fTmpA*fC0;
   pSH[25] = fTmpA*fS0;
   fTmpB = -2.366619162231753f*fZ;
   pSH[47] = fTmpB*fC0;
   pSH[37] = fTmpB*fS0;
   fTmpC = -6.745902523363385f*fZ2 + 0.5189155787202604f;
   pSH[61] = fTmpC*fC0;
   pSH[51] = fTmpC*fS0;
   fTmpA = fZ*(-17.24955311049054f*fZ2 + 3.449910622098108f);
   pSH[77] = fTmpA*fC0;
   pSH[67] = fTmpA*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6831841051919144f;
   pSH[48] = fTmpA*fC1;
   pSH[36] = fTmpA*fS1;
   fTmpB = 2.645960661801901f*fZ;
   pSH[62] = fTmpB*fC1;
   pSH[50] = fTmpB*fS1;
   fTmpC = 7.984991490893139f*fZ2 + -0.5323327660595426f;
   pSH[78] = fTmpC*fC1;
   pSH[66] = fTmpC*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.7071627325245963f;
   pSH[63] = fTmpA*fC0;
   pSH[49] = fTmpA*fS0;
   fTmpB = -2.91570664069932f*fZ;
   pSH[79] = fTmpB*fC0;
   pSH[65] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpC = 0.72892666017483f;
   pSH[80] = fTmpC*fC1;
   pSH[64] = fTmpC*fS1;
}

// order 10
void SHEval10(const float fX, const float fY, const float fZ, float *pSH)
{
   float fC0,fC1,fS0,fS1,fTmpA,fTmpB,fTmpC;
   float fZ2 = fZ*fZ;

   pSH[0] = 0.2820947917738781f;
   pSH[2] = 0.4886025119029199f*fZ;
   pSH[6] = 0.9461746957575601f*fZ2 + -0.3153915652525201f;
   pSH[12] = fZ*(1.865881662950577f*fZ2 + -1.119528997770346f);
   pSH[20] = 1.984313483298443f*fZ*pSH[12] + -1.006230589874905f*pSH[6];
   pSH[30] = 1.98997487421324f*fZ*pSH[20] + -1.002853072844814f*pSH[12];
   pSH[42] = 1.993043457183567f*fZ*pSH[30] + -1.001542020962219f*pSH[20];
   pSH[56] = 1.994891434824135f*fZ*pSH[42] + -1.000927213921958f*pSH[30];
   pSH[72] = 1.996089927833914f*fZ*pSH[56] + -1.000600781069515f*pSH[42];
   pSH[90] = 1.996911195067937f*fZ*pSH[72] + -1.000411437993134f*pSH[56];
   fC0 = fX;
   fS0 = fY;

   fTmpA = -0.48860251190292f;
   pSH[3] = fTmpA*fC0;
   pSH[1] = fTmpA*fS0;
   fTmpB = -1.092548430592079f*fZ;
   pSH[7] = fTmpB*fC0;
   pSH[5] = fTmpB*fS0;
   fTmpC = -2.285228997322329f*fZ2 + 0.4570457994644658f;
   pSH[13] = fTmpC*fC0;
   pSH[11] = fTmpC*fS0;
   fTmpA = fZ*(-4.683325804901025f*fZ2 + 2.007139630671868f);
   pSH[21] = fTmpA*fC0;
   pSH[19] = fTmpA*fS0;
   fTmpB = 2.03100960115899f*fZ*fTmpA + -0.991031208965115f*fTmpC;
   pSH[31] = fTmpB*fC0;
   pSH[29] = fTmpB*fS0;
   fTmpC = 2.021314989237028f*fZ*fTmpB + -0.9952267030562385f*fTmpA;
   pSH[43] = fTmpC*fC0;
   pSH[41] = fTmpC*fS0;
   fTmpA = 2.015564437074638f*fZ*fTmpC + -0.9971550440218319f*fTmpB;
   pSH[57] = fTmpA*fC0;
   pSH[55] = fTmpA*fS0;
   fTmpB = 2.011869540407391f*fZ*fTmpA + -0.9981668178901745f*fTmpC;
   pSH[73] = fTmpB*fC0;
   pSH[71] = fTmpB*fS0;
   fTmpC = 2.009353129741012f*fZ*fTmpB + -0.9987492177719088f*fTmpA;
   pSH[91] = fTmpC*fC0;
   pSH[89] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.5462742152960395f;
   pSH[8] = fTmpA*fC1;
   pSH[4] = fTmpA*fS1;
   fTmpB = 1.445305721320277f*fZ;
   pSH[14] = fTmpB*fC1;
   pSH[10] = fTmpB*fS1;
   fTmpC = 3.31161143515146f*fZ2 + -0.47308734787878f;
   pSH[22] = fTmpC*fC1;
   pSH[18] = fTmpC*fS1;
   fTmpA = fZ*(7.190305177459987f*fZ2 + -2.396768392486662f);
   pSH[32] = fTmpA*fC1;
   pSH[28] = fTmpA*fS1;
   fTmpB = 2.11394181566097f*fZ*fTmpA + -0.9736101204623268f*fTmpC;
   pSH[44] = fTmpB*fC1;
   pSH[40] = fTmpB*fS1;
   fTmpC = 2.081665999466133f*fZ*fTmpB + -0.9847319278346618f*fTmpA;
   pSH[58] = fTmpC*fC1;
   pSH[54] = fTmpC*fS1;
   fTmpA = 2.06155281280883f*fZ*fTmpC + -0.9903379376602873f*fTmpB;
   pSH[74] = fTmpA*fC1;
   pSH[70] = fTmpA*fS1;
   fTmpB = 2.048122358357819f*fZ*fTmpA + -0.9934852726704042f*fTmpC;
   pSH[92] = fTmpB*fC1;
   pSH[88] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.5900435899266435f;
   pSH[15] = fTmpA*fC0;
   pSH[9] = fTmpA*fS0;
   fTmpB = -1.770130769779931f*fZ;
   pSH[23] = fTmpB*fC0;
   pSH[17] = fTmpB*fS0;
   fTmpC = -4.403144694917254f*fZ2 + 0.4892382994352505f;
   pSH[33] = fTmpC*fC0;
   pSH[27] = fTmpC*fS0;
   fTmpA = fZ*(-10.13325785466416f*fZ2 + 2.763615778544771f);
   pSH[45] = fTmpA*fC0;
   pSH[39] = fTmpA*fS0;
   fTmpB = 2.207940216581962f*fZ*fTmpA + -0.959403223600247f*fTmpC;
   pSH[59] = fTmpB*fC0;
   pSH[53] = fTmpB*fS0;
   fTmpC = 2.15322168769582f*fZ*fTmpB + -0.9752173865600178f*fTmpA;
   pSH[75] = fTmpC*fC0;
   pSH[69] = fTmpC*fS0;
   fTmpA = 2.118044171189805f*fZ*fTmpC + -0.9836628449792094f*fTmpB;
   pSH[93] = fTmpA*fC0;
   pSH[87] = fTmpA*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6258357354491763f;
   pSH[24] = fTmpA*fC1;
   pSH[16] = fTmpA*fS1;
   fTmpB = 2.075662314881041f*fZ;
   pSH[34] = fTmpB*fC1;
   pSH[26] = fTmpB*fS1;
   fTmpC = 5.550213908015966f*fZ2 + -0.5045649007287241f;
   pSH[46] = fTmpC*fC1;
   pSH[38] = fTmpC*fS1;
   fTmpA = fZ*(13.49180504672677f*fZ2 + -3.113493472321562f);
   pSH[60] = fTmpA*fC1;
   pSH[52] = fTmpA*fS1;
   fTmpB = 2.304886114323221f*fZ*fTmpA + -0.9481763873554654f*fTmpC;
   pSH[76] = fTmpB*fC1;
   pSH[68] = fTmpB*fS1;
   fTmpC = 2.229177150706235f*fZ*fTmpB + -0.9671528397231821f*fTmpA;
   pSH[94] = fTmpC*fC1;
   pSH[86] = fTmpC*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.6563820568401703f;
   pSH[35] = fTmpA*fC0;
   pSH[25] = fTmpA*fS0;
   fTmpB = -2.366619162231753f*fZ;
   pSH[47] = fTmpB*fC0;
   pSH[37] = fTmpB*fS0;
   fTmpC = -6.745902523363385f*fZ2 + 0.5189155787202604f;
   pSH[61] = fTmpC*fC0;
   pSH[51] = fTmpC*fS0;
   fTmpA = fZ*(-17.24955311049054f*fZ2 + 3.449910622098108f);
   pSH[77] = fTmpA*fC0;
   pSH[67] = fTmpA*fS0;
   fTmpB = 2.401636346922062f*fZ*fTmpA + -0.9392246042043708f*fTmpC;
   pSH[95] = fTmpB*fC0;
   pSH[85] = fTmpB*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.6831841051919144f;
   pSH[48] = fTmpA*fC1;
   pSH[36] = fTmpA*fS1;
   fTmpB = 2.645960661801901f*fZ;
   pSH[62] = fTmpB*fC1;
   pSH[50] = fTmpB*fS1;
   fTmpC = 7.984991490893139f*fZ2 + -0.5323327660595426f;
   pSH[78] = fTmpC*fC1;
   pSH[66] = fTmpC*fS1;
   fTmpA = fZ*(21.39289019090864f*fZ2 + -3.775215916042701f);
   pSH[96] = fTmpA*fC1;
   pSH[84] = fTmpA*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpA = -0.7071627325245963f;
   pSH[63] = fTmpA*fC0;
   pSH[49] = fTmpA*fS0;
   fTmpB = -2.91570664069932f*fZ;
   pSH[79] = fTmpB*fC0;
   pSH[65] = fTmpB*fS0;
   fTmpC = -9.263393182848905f*fZ2 + 0.5449054813440533f;
   pSH[97] = fTmpC*fC0;
   pSH[83] = fTmpC*fS0;
   fC1 = fX*fC0 - fY*fS0;
   fS1 = fX*fS0 + fY*fC0;

   fTmpA = 0.72892666017483f;
   pSH[80] = fTmpA*fC1;
   pSH[64] = fTmpA*fS1;
   fTmpB = 3.177317648954698f*fZ;
   pSH[98] = fTmpB*fC1;
   pSH[82] = fTmpB*fS1;
   fC0 = fX*fC1 - fY*fS1;
   fS0 = fX*fS1 + fY*fC1;

   fTmpC = -0.7489009518531884f;
   pSH[99] = fTmpC*fC0;
   pSH[81] = fTmpC*fS0;
}

class NoiseGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
    
public:
    NoiseGenerator(unsigned int seed = 42) : gen(seed), dist(-1.0f, 1.0f) {}
    
    float sample() {
        return dist(gen);
    }
};

// Simple SIMD bands structure to hold multiple frequency bands
struct SIMDBands {
    std::vector<float> bands;
    
    SIMDBands(int numBands) : bands(numBands, 0.0f) {}
    
    SIMDBands& operator+=(const SIMDBands& other) {
        for (size_t i = 0; i < bands.size(); i++) {
            bands[i] += other.bands[i];
        }
        return *this;
    }
};

// Filter coefficients for a single crossover band
struct FilterCoefficients {
    float a[6]; // Feed-forward coefficients
    float b[4]; // Feedback coefficients
};

// History state for a single filter
struct FilterState {
    std::vector<float> input;  // Input history
    std::vector<float> output; // Output history
    
    FilterState() : input(4, 0.0f), output(4, 0.0f) {}
    
    void reset() {
        std::fill(input.begin(), input.end(), 0.0f);
        std::fill(output.begin(), output.end(), 0.0f);
    }
};

class CrossoverFilter {
public:
    class History {
    public:
        History(int numBands) : states(numBands - 1) {}
        
        void reset() {
            for (auto& state : states) {
                state.reset();
            }
        }
        
        std::vector<FilterState> states;
    };
    
    CrossoverFilter(float sampleRate, const std::vector<float>& freqPoints) 
        : sampleRate(sampleRate)
        , numBands(freqPoints.size() + 1)
        , coefficients((freqPoints.size()) * 2) // Two sets per crossover point
    {
        for (size_t i = 0; i < freqPoints.size(); i++) {
            float w0 = 2.0f * M_PI * freqPoints[i] / sampleRate;
            computeButterworth2Coefficients(w0, i);
        }
    }
    
    void process(History& history, const float* input, SIMDBands* output, int numSamples) {
        for (int i = 0; i < numSamples; i++) {
            // Start with input value in all bands
            SIMDBands current(numBands);
            for (int band = 0; band < numBands; band++) {
                current.bands[band] = input[i];
            }
            
            // Apply crossover filters
            for (int crossover = 0; crossover < numBands - 1; crossover++) {
                FilterState& state = history.states[crossover];
                
                // Apply filters to bands above the crossover point
                for (int band = crossover + 1; band < numBands; band++) {
                    // Apply highpass filter
                    current.bands[band] = applyFilter(current.bands[band], 
                                                    coefficients[crossover * 2 + 1],
                                                    state);
                }
                
                // Apply lowpass filter to band at crossover point
                current.bands[crossover] = applyFilter(current.bands[crossover],
                                                     coefficients[crossover * 2],
                                                     state);
            }
            
            output[i] = current;
        }
    }
    
private:
    float sampleRate;
    int numBands;
    std::vector<FilterCoefficients> coefficients;
    
    void computeButterworth2Coefficients(float w0, int crossoverIndex) {
        // Compute lowpass coefficients
        FilterCoefficients& lowpass = coefficients[crossoverIndex * 2];
        float w0LP = 1.0f / std::tan(w0 / 2.0f);
        computeButterworth2LowPass(w0LP, lowpass);
        
        // Compute highpass coefficients  
        FilterCoefficients& highpass = coefficients[crossoverIndex * 2 + 1];
        float w0HP = std::tan(w0 / 2.0f);
        computeButterworth2HighPass(w0HP, highpass);
    }
    
    void computeButterworth2LowPass(float w0, FilterCoefficients& coeff) {
        float B1 = -2.0f * std::cos(M_PI * 3.0f / 4.0f);
        float w0squared = w0 * w0;
        float A = 1.0f + B1 * w0 + w0squared;
        
        coeff.a[0] = 1.0f / A;
        coeff.a[1] = 2.0f;
        coeff.a[2] = 1.0f;
        coeff.b[0] = 2.0f * (1.0f - w0squared) * coeff.a[0];
        coeff.b[1] = (1.0f - B1 * w0 + w0squared) * coeff.a[0];
        
        // Second stage coefficients
        coeff.a[3] = coeff.a[0];
        coeff.a[4] = coeff.a[1];
        coeff.a[5] = coeff.a[2];
        coeff.b[2] = coeff.b[0];
        coeff.b[3] = coeff.b[1];
    }
    
    void computeButterworth2HighPass(float w0, FilterCoefficients& coeff) {
        computeButterworth2LowPass(w0, coeff);
        coeff.a[1] = -coeff.a[1];
        coeff.a[4] = -coeff.a[4];
        coeff.b[0] = -coeff.b[0];
        coeff.b[2] = -coeff.b[2];
    }
    
    float applyFilter(float input, const FilterCoefficients& coeff, FilterState& state) {
        // First biquad stage
        float in1 = coeff.a[0] * input;
        float out1 = (in1 - coeff.b[0] * state.output[0]) +
                    (coeff.a[1] * state.input[0] - coeff.b[1] * state.output[1]) +
                    coeff.a[2] * state.input[1];
        
        // Update first stage history
        state.input[1] = state.input[0];
        state.input[0] = in1;
        state.output[1] = state.output[0];
        state.output[0] = out1;
        
        // Second biquad stage
        float in2 = coeff.a[3] * out1;
        float out2 = (in2 - coeff.b[2] * state.output[2]) +
                    (coeff.a[4] * state.input[2] - coeff.b[3] * state.output[3]) +
                    coeff.a[5] * state.input[3];
        
        // Update second stage history
        state.input[3] = state.input[2];
        state.input[2] = in2;
        state.output[3] = state.output[2];
        state.output[2] = out2;
        
        return out2;
    }
};

py::array_t<float> generate_ambisonic_ir(
    int order,
    py::array_t<float> listener_directions,
    py::array_t<float> intensities,
    py::array_t<float> distances,
    py::array_t<float> speeds,
    py::array_t<int> path_types,
    py::array_t<float> frequency_points,
    float sample_rate,
    bool normalize = true
) {
    // Input validation
    if (order < 1 || order > 9) {
        throw std::runtime_error("Order must be between 1 and 9");
    }
    
    // Get array dimensions
    int num_paths = listener_directions.shape(0);
    int num_bands = intensities.shape(1);
    int num_coefficients = (order + 1) * (order + 1);
    
    // Get frequency points
    auto freq_buf = frequency_points.request();
    std::vector<float> freq_points((float*)freq_buf.ptr, (float*)freq_buf.ptr + freq_buf.size);
    
    if (freq_points.size() + 1 != size_t(num_bands)) {
        throw std::runtime_error("Number of frequency points must be number of bands - 1");
    }
    
    // Calculate IR length with padding
    auto distances_buf = distances.unchecked<1>();
    auto speeds_buf = speeds.unchecked<1>();
    float max_delay = 0.0f;
    for (int i = 0; i < num_paths; i++) {
        float delay = distances_buf(i) / speeds_buf(i);
        max_delay = std::max(max_delay, delay);
    }
    
    const int filter_padding = 2048;
    int ir_length = static_cast<int>(std::ceil(max_delay * sample_rate)) + filter_padding;
    
    // Generate noise
    std::vector<float> raw_noise(ir_length);
    NoiseGenerator noise_gen(42);
    for (int i = 0; i < ir_length; i++) {
        raw_noise[i] = noise_gen.sample();
    }
    
    // Create and initialize crossover filter
    CrossoverFilter crossover(sample_rate, freq_points);
    CrossoverFilter::History noise_history(num_bands);
    
    // Create filtered noise buffer with proper initialization
    std::vector<SIMDBands> filtered_noise;
    filtered_noise.reserve(ir_length);
    for (int i = 0; i < ir_length; i++) {
        filtered_noise.emplace_back(num_bands);
    }
    
    // Filter noise through crossover
    crossover.process(noise_history, raw_noise.data(), filtered_noise.data(), ir_length);
    
    // Process paths
    auto directions_buf = listener_directions.unchecked<2>();
    auto intensities_buf = intensities.unchecked<2>();
    
    // Initialize band IRs
    std::vector<std::vector<SIMDBands>> band_irs;
    band_irs.resize(num_coefficients);
    for (int i = 0; i < num_coefficients; i++) {
        band_irs[i].resize(ir_length, SIMDBands(num_bands));
    }
    
    std::vector<float> sh_coeffs(num_coefficients);
    
    for (int path = 0; path < num_paths; path++) {
        // Calculate delay
        float delay = distances_buf(path) / speeds_buf(path);
        int delay_samples = static_cast<int>(std::floor(delay * sample_rate));
        if (delay_samples >= ir_length) continue;
        
        // Get normalized direction
        float dx = directions_buf(path, 0);
        float dy = directions_buf(path, 1);
        float dz = directions_buf(path, 2);
        
        float length = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (length > 0) {
            dx /= length;
            dy /= length;
            dz /= length;
        }
        
        // Evaluate spherical harmonics
        switch(order) {
            case 1: SHEval2(dx, dy, dz, sh_coeffs.data()); break;
            case 2: SHEval3(dx, dy, dz, sh_coeffs.data()); break;
            case 3: SHEval4(dx, dy, dz, sh_coeffs.data()); break;
            case 4: SHEval5(dx, dy, dz, sh_coeffs.data()); break;
            case 5: SHEval6(dx, dy, dz, sh_coeffs.data()); break;
            case 6: SHEval7(dx, dy, dz, sh_coeffs.data()); break;
            case 7: SHEval8(dx, dy, dz, sh_coeffs.data()); break;
            case 8: SHEval9(dx, dy, dz, sh_coeffs.data()); break;
            case 9: SHEval10(dx, dy, dz, sh_coeffs.data()); break;
        }
        
        // Add contribution to band IRs
        for (int coeff = 0; coeff < num_coefficients; coeff++) {
            for (int band = 0; band < num_bands; band++) {
                float intensity = std::sqrt(intensities_buf(path, band));
                band_irs[coeff][delay_samples].bands[band] += sh_coeffs[coeff] * intensity;
            }
        }
    }
    
    // Create output array
    std::vector<ssize_t> output_shape = {num_coefficients, ir_length};
    auto result = py::array_t<float>(output_shape);
    auto result_buf = result.mutable_unchecked<2>();
    
    // Generate final waveforms
    for (int coeff = 0; coeff < num_coefficients; coeff++) {
        for (int i = 0; i < ir_length; i++) {
            float sum = 0.0f;
            for (int band = 0; band < num_bands; band++) {
                sum += band_irs[coeff][i].bands[band] * filtered_noise[i].bands[band];
            }
            result_buf(coeff, i) = sum;
        }
    }
    
    // Normalize if requested
    if (normalize) {
        float max_sample = 0.0f;
        
        for (int coeff = 0; coeff < num_coefficients; coeff++) {
            for (int i = 0; i < ir_length; i++) {
                max_sample = std::max(max_sample, std::abs(result_buf(coeff, i)));
            }
        }
        
        if (max_sample > 0.0f) {
            float scale = 1.0f / max_sample;
            for (int coeff = 0; coeff < num_coefficients; coeff++) {
                for (int i = 0; i < ir_length; i++) {
                    result_buf(coeff, i) *= scale;
                }
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(spherical_harmonics, m) {
    m.doc() = "Spherical Harmonics processor for generating Ambisonic IR waveforms";
    
    m.def("generate_ambisonic_ir", &generate_ambisonic_ir,
          "Generate Ambisonic IR waveforms using noise-based synthesis",
          py::arg("order"),
          py::arg("listener_directions"),
          py::arg("intensities"), 
          py::arg("distances"),
          py::arg("speeds"),
          py::arg("path_types"),
          py::arg("frequency_points"),
          py::arg("sample_rate"),
          py::arg("normalize") = true);
}