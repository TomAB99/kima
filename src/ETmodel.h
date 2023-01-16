#pragma once

#include <vector>
#include <memory>
#include "ConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "DNest4.h"
#include "Data.h"
#include "kepler.h"
#include "AMDstability.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "celerite/celerite.h"

/// whether the data comes from different instruments
/// (and offsets should be included in the model)
extern const bool multi_instrument;

/// include a (better) known extra Keplerian curve? (KO mode!)
extern const bool known_object;
extern const int n_known_object;

/// use a Student-t distribution for the likelihood (instead of Gaussian)
extern const bool studentt;

///type of ephemeris
extern const int ephemeris;

class ETmodel
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<ETConditionalPrior> planets =
            DNest4::RJObject<ETConditionalPrior>(5, npmax, fix, ETConditionalPrior());

        double ephem1, ephem2=0.0, ephem3=0.0;
        double nu;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // The signal
        std::vector<double> mu = // the ET model
                            std::vector<double>(ETData::get_instance().N());
        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        int is_stable() const;
        bool enforce_stability = false;
        
        double star_mass = 1.0;  // [Msun]

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        ETmodel();

        void initialise() {};

        // priors for parameters *not* belonging to the planets

        // priors for KO mode!
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Kprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};
        
        std::shared_ptr<DNest4::ContinuousDistribution> ephem1_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> ephem2_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> ephem3_prior;

        std::shared_ptr<DNest4::ContinuousDistribution> nu_prior;

        // change the name of std::make_shared :)
        /**
         * @brief Assign a prior distribution.
         * 
         * This function defines, initializes, and assigns a prior distribution.
         * Possible distributions are ...
         * 
         * For example:
         * 
         * @code{.cpp}
         *          Cprior = make_prior<Uniform>(0, 1);
         * @endcode
         * 
         * @tparam T     ContinuousDistribution
         * @tparam Args  
         * @param args   Arguments for constructor of distribution
         * @return std::shared_ptr<T> 
        */
        template <class T, class... Args>
        std::shared_ptr<T> make_prior(Args&&... args)
        {
            return std::make_shared<T>(args...);
        }

        // create an alias for ETData::get_instance()
        static ETData& get_data() { return ETData::get_instance(); }

        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

};

