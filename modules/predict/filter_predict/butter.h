#pragma once

//#include <lvo/common.h>
#include <complex>
#include <vector>

namespace butter {

std::vector<double> ComputeLP(int32_t FilterOrder);
std::vector<double> ComputeHP(int32_t FilterOrder);
std::vector<double> TrinomialMultiply(int32_t FilterOrder, std::vector<double> &b, std::vector<double> &c);
std::vector<double> ComputeNumCoeffs(int32_t FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> denominator);
std::vector<double> ComputeDenCoeffs(int32_t FilterOrder, double Lcutoff, double Ucutoff);
std::vector<double> filter(std::vector<double>x, std::vector<double> coeff_b, std::vector<double> coeff_a);

} // namespace lvo
