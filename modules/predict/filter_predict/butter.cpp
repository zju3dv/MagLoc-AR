#include <butter.h>
#define PI 3.1415926535897932384626433832795

namespace butter {

std::vector<double> filter(std::vector<double>x, std::vector<double> coeff_b, std::vector<double> coeff_a)
{
  int len_x = x.size();
  int len_b = coeff_b.size();
  int len_a = coeff_a.size();

  std::vector<double> zi(len_b);

  std::vector<double> filter_x(len_x);

  if (len_a == 1)
  {
    for (int m = 0; m<len_x; m++)
    {
      filter_x[m] = coeff_b[0] * x[m] + zi[0];
      for (int i = 1; i<len_b; i++)
      {
        zi[i - 1] = coeff_b[i] * x[m] + zi[i];//-coeff_a[i]*filter_x[m];
      }
    }
  }
  else
  {
    for (int m = 0; m<len_x; m++)
    {
      filter_x[m] = coeff_b[0] * x[m] + zi[0];
      for (int i = 1; i<len_b; i++)
      {
        zi[i - 1] = coeff_b[i] * x[m] + zi[i] - coeff_a[i] * filter_x[m];
      }
    }
  }

  return filter_x;
}

std::vector<double> ComputeLP(int32_t FilterOrder) {
    std::vector<double> NumCoeffs(FilterOrder + 1);
    int32_t m;
    int32_t i;

    NumCoeffs[0] = 1;
    NumCoeffs[1] = FilterOrder;
    m = FilterOrder / 2;
    for (i = 2; i <= m; ++i) {
        NumCoeffs[i] = (double)(FilterOrder - i + 1) * NumCoeffs[i - 1] / i;
        NumCoeffs[FilterOrder - i] = NumCoeffs[i];
    }
    NumCoeffs[FilterOrder - 1] = FilterOrder;
    NumCoeffs[FilterOrder] = 1;

    return NumCoeffs;
}

std::vector<double> ComputeHP(int32_t FilterOrder) {
    std::vector<double> NumCoeffs;
    int32_t i;

    NumCoeffs = ComputeLP(FilterOrder);

    for (i = 0; i <= FilterOrder; ++i)
        if (i % 2) NumCoeffs[i] = -NumCoeffs[i];

    return NumCoeffs;
}

std::vector<double> TrinomialMultiply(int32_t FilterOrder, std::vector<double> &b, std::vector<double> &c) {
    int32_t i, j;
    std::vector<double> RetVal(4 * FilterOrder);

    RetVal[2] = c[0];
    RetVal[3] = c[1];
    RetVal[0] = b[0];
    RetVal[1] = b[1];

    for (i = 1; i < FilterOrder; ++i) {
        RetVal[2 * (2 * i + 1)] += c[2 * i] * RetVal[2 * (2 * i - 1)] - c[2 * i + 1] * RetVal[2 * (2 * i - 1) + 1];
        RetVal[2 * (2 * i + 1) + 1] += c[2 * i] * RetVal[2 * (2 * i - 1) + 1] + c[2 * i + 1] * RetVal[2 * (2 * i - 1)];

        for (j = 2 * i; j > 1; --j) {
            RetVal[2 * j] += b[2 * i] * RetVal[2 * (j - 1)] - b[2 * i + 1] * RetVal[2 * (j - 1) + 1] + c[2 * i] * RetVal[2 * (j - 2)] - c[2 * i + 1] * RetVal[2 * (j - 2) + 1];
            RetVal[2 * j + 1] += b[2 * i] * RetVal[2 * (j - 1) + 1] + b[2 * i + 1] * RetVal[2 * (j - 1)] + c[2 * i] * RetVal[2 * (j - 2) + 1] + c[2 * i + 1] * RetVal[2 * (j - 2)];
        }

        RetVal[2] += b[2 * i] * RetVal[0] - b[2 * i + 1] * RetVal[1] + c[2 * i];
        RetVal[3] += b[2 * i] * RetVal[1] + b[2 * i + 1] * RetVal[0] + c[2 * i + 1];
        RetVal[0] += b[2 * i];
        RetVal[1] += b[2 * i + 1];
    }
    return RetVal;
}

std::vector<double> ComputeNumCoeffs(int32_t FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> denominator) {
    std::vector<double> TCoeffs;
    std::vector<double> NumCoeffs(2 * FilterOrder + 1);
    std::vector<std::complex<double>> NormalizedKernel(2 * FilterOrder + 1);
    double Numbers[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t i;

    TCoeffs = ComputeHP(FilterOrder);

    for (i = 0; i < FilterOrder; ++i) {
        NumCoeffs[2 * i] = TCoeffs[i];
        NumCoeffs[2 * i + 1] = 0.0;
    }
    NumCoeffs[2 * FilterOrder] = TCoeffs[FilterOrder];
    double cp[2];
    //double Bw;
    double Wn;
    cp[0] = 2 * 2.0 * std::tan(PI * Lcutoff / 2.0);
    cp[1] = 2 * 2.0 * std::tan(PI * Ucutoff / 2.0);

    //Bw = cp[1] - cp[0];
    //center frequency
    Wn = sqrt(cp[0] * cp[1]);
    Wn = 2 * atan2(Wn, 4);
    const std::complex<double> result = std::complex<double>(-1, 0);

    for (int32_t k = 0; k < 2 * FilterOrder + 1; k++) {
        NormalizedKernel[k] = std::exp(-sqrt(result) * Wn * Numbers[k]);
    }
    double b = 0;
    double den = 0;
    for (int32_t d = 0; d < 2 * FilterOrder + 1; d++) {
        b += real(NormalizedKernel[d] * NumCoeffs[d]);
        den += real(NormalizedKernel[d] * denominator[d]);
    }
    for (int32_t c = 0; c < 2 * FilterOrder + 1; c++) {
        NumCoeffs[c] = (NumCoeffs[c] * den) / b;
    }

    return NumCoeffs;
}

std::vector<double> ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff) {
    int32_t k;                                    // loop variables
    double theta;                                 // PI * (Ucutoff - Lcutoff)/2.0
    double cp;                                    // cosine of phi
    double st;                                    // sine of theta
    double ct;                                    // cosine of theta
    double s2t;                                   // sine of 2*theta
    double c2t;                                   // cosine 0f 2*theta
    std::vector<double> RCoeffs(2 * FilterOrder); // z^-2 coefficients
    std::vector<double> TCoeffs(2 * FilterOrder); // z^-1 coefficients
    std::vector<double> DenomCoeffs;              // dk coefficients
    double PoleAngle;                             // pole angle
    double SinPoleAngle;                          // sine of pole angle
    double CosPoleAngle;                          // cosine of pole angle
    double a;                                     // workspace variables

    cp = std::cos(PI * (Ucutoff + Lcutoff) / 2.0);
    theta = PI * (Ucutoff - Lcutoff) / 2.0;
    st = std::sin(theta);
    ct = std::cos(theta);
    s2t = 2.0 * st * ct;       // sine of 2*theta
    c2t = 2.0 * ct * ct - 1.0; // cosine of 2*theta

    for (k = 0; k < FilterOrder; ++k) {
        PoleAngle = PI * (double)(2 * k + 1) / (double)(2 * FilterOrder);
        SinPoleAngle = std::sin(PoleAngle);
        CosPoleAngle = std::cos(PoleAngle);
        a = 1.0 + s2t * SinPoleAngle;
        RCoeffs[2 * k] = c2t / a;
        RCoeffs[2 * k + 1] = s2t * CosPoleAngle / a;
        TCoeffs[2 * k] = -2.0 * cp * (ct + st * SinPoleAngle) / a;
        TCoeffs[2 * k + 1] = -2.0 * cp * st * CosPoleAngle / a;
    }

    DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs);

    DenomCoeffs[1] = DenomCoeffs[0];
    DenomCoeffs[0] = 1.0;
    for (k = 3; k <= 2 * FilterOrder; ++k)
        DenomCoeffs[k] = DenomCoeffs[2 * k - 2];

    DenomCoeffs.resize(FilterOrder * 2 + 1);

    return DenomCoeffs;
}
} // namespace lvo
