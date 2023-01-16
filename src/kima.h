#pragma once

#include "DNest4.h"
#include "distributions/distributions.h"

#include "Data.h"
#include "ConditionalPrior.h"
#include "ConditionalPrior_2.h"
#include "RVmodel.h"
#include "RVFWHMmodel.h"
#include "RV_binaries_model.h"
#include "ETmodel.h"

const double PI = M_PI;

std::string datafile;
std::vector<std::string> datafiles;


using ind = std::vector<char*>;
std::vector<std::string> indicators;


using namespace DNest4;