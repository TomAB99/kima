#include "ETmodel.h"

using namespace std;
using namespace Eigen;
using namespace DNest4;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


/* set default priors if the user didn't change them */

void ETmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto data = get_data();
    
     if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(0.1,100.);
    
    if (ephemeris >= 4)
        throw std::logic_error("can't go higher than cubic ephemeris ");
    if (ephemeris < 1)
        throw std::logic_error("ephemeris should be at least one since eclipse needs a period");
    if (ephemeris >= 1 && !ephem1_prior){
        //ephem1_prior = make_prior<Gaussian>(0.0,10.);
        ephem1_prior =  make_prior<LogUniform>(0.0001,1000);
        printf("# No prior on Binary period specified, taken as LogUniform over 0.0001-1000\n");
    }
    if (ephemeris >= 2 && !ephem2_prior)
        ephem2_prior = make_prior<Gaussian>(0.0,pow(10,-10.));
    if (ephemeris >= 3 && !ephem3_prior)
        ephem3_prior = make_prior<Gaussian>(0.0,pow(10,-12.));
    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void ETmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();
    
    extra_sigma = Jprior->generate(rng);
    

    auto data = get_data();
    
    if (ephemeris >= 1) ephem1 = ephem1_prior->generate(rng);
    if (ephemeris >= 2) ephem2 = ephem2_prior->generate(rng);
    if (ephemeris == 3) ephem3 = ephem3_prior->generate(rng);

    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_K.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_w.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
        }
    }

    if (studentt)
        nu = nu_prior->generate(rng);

    calculate_mu();

}

/**
 * @brief Calculate the full ET model
 * 
*/
void ETmodel::calculate_mu()
{
    auto data = get_data();
    // Get the epochs from the data
    const vector<double>& epochs = data.get_epochs();

    // Update or from scratch?
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);

    // Get the components
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    // at this point, components has:
    //  if updating: only the added planets' parameters
    //  if from scratch: all the planets' parameters

    // Zero the signal
    if(!update) // not updating, means recalculate everything
    {
        mu.assign(mu.size(), data.M0_epoch);
        
        staleness = 0;
        
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += ephem1*epochs[i]+ 0.5*ephem2*ephem1*pow(epochs[i],2.0) + ephem3*pow(ephem1,2.0)*pow(epochs[i],3.0)/6.0;
        }

        if (known_object) { // KO mode!
            add_known_object();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double f, tau, ti;
    double P, K, phi, ecc, omega, Tp;
    for(size_t j=0; j<components.size(); j++)
    {
        P = components[j][0];
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];

        for(size_t i=0; i<epochs.size(); i++)
        {
            ti = data.M0_epoch + epochs[i]*ephem1;
            Tp = data.M0_epoch - (P * phi) / (2. * M_PI);
            f = nijenhuis::true_anomaly(ti, P, ecc, Tp);
            // f = brandt::true_anomaly(ti, P, ecc, Tp);
            tau = K/(24*3600) * ((1-pow(ecc,2.))/(1+ecc*cos(f)) * sin(f + omega) + ecc * sin(omega))/(pow(1 - pow(ecc*cos(omega),2.0),0.5));
            mu[i] += tau;
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void ETmodel::remove_known_object()
{
    auto data = get_data();
    auto epochs = data.get_epochs();
    double f, tau, ti, Tp;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<epochs.size(); i++)
        {
            ti = data.M0_epoch + epochs[i]*ephem1;
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            f = nijenhuis::true_anomaly(ti, KO_P[j], KO_e[j], Tp);
            tau = KO_K[j]/(24*3600) * ((1-pow(KO_e[j],2.))/(1+KO_e[j]*cos(f)) * sin(f + KO_w[j]) + KO_e[j] * sin(KO_w[j]))/(pow(1 - pow(KO_e[j]*cos(KO_w[j]),2.0),0.5));
            mu[i] -= tau;
        }
    }
}


void ETmodel::add_known_object()
{
    auto data = get_data();
    auto epochs = data.get_epochs();
    double f, tau, ti, Tp;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<epochs.size(); i++)
        {
            ti = data.M0_epoch + epochs[i]*ephem1;
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            f = nijenhuis::true_anomaly(ti, KO_P[j], KO_e[j], Tp);
            tau = KO_K[j]/(24*3600) * ((1-pow(KO_e[j],2.))/(1+KO_e[j]*cos(f)) * sin(f + KO_w[j]) + KO_e[j] * sin(KO_w[j]))/(pow(1 - pow(KO_e[j]*cos(KO_w[j]),2.0),0.5));
            mu[i] += tau;
        }
    }
}

int ETmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0)
        return 0;
    // cout << components[0].size() << endl;
    // cout << AMD::AMD_stable(components) << endl;
    return AMD::AMD_stable(components, star_mass);
}


double ETmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto data = get_data();
    const vector<double>& epochs = data.get_epochs();
    double logH = 0.;

    if(rng.rand() <= 0.75) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        
        Jprior->perturb(extra_sigma, rng);
        
        if (studentt)
            nu_prior->perturb(nu, rng);


        if (known_object)
        {
            remove_known_object();

            for (int i=0; i<n_known_object; i++){
                KO_Pprior[i]->perturb(KO_P[i], rng);
                KO_Kprior[i]->perturb(KO_K[i], rng);
                KO_eprior[i]->perturb(KO_e[i], rng);
                KO_phiprior[i]->perturb(KO_phi[i], rng);
                KO_wprior[i]->perturb(KO_w[i], rng);
            }

            add_known_object();
        }
        
    }
    else
    {
        //subtract ephemeris
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += -data.M0_epoch - ephem1*epochs[i]- 0.5*ephem2*ephem1*pow(epochs[i],2.0) - ephem3*pow(ephem1,2.0)*pow(epochs[i],3.0)/6.0;
        }
        
        // propose new ephemeris
        if (ephemeris >= 1) ephem1_prior->perturb(ephem1, rng);
        if (ephemeris >= 2) ephem2_prior->perturb(ephem2, rng);
        if (ephemeris == 3) ephem3_prior->perturb(ephem3, rng);

        //add ephemeris back in
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += data.M0_epoch + ephem1*epochs[i]+ 0.5*ephem2*ephem1*pow(epochs[i],2.0) + ephem3*pow(ephem1,2.0)*pow(epochs[i],3.0)/6.0;
        }
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6;
    cout << " ms" << std::endl;
    #endif

    return logH;
}

/**
 * Calculate the log-likelihood for the current values of the parameters.
 * 
 * @return double the log-likelihood
*/
double ETmodel::log_likelihood() const
{
    const auto data = get_data();
    int N = data.N();
    auto et = data.get_et();
    auto etsig = data.get_etsig();

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    
    double jit = extra_sigma/(24*3600);

    if (studentt){
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = etsig[i]*etsig[i] +jit*jit;

            logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(et[i] - mu[i], 2)/var/nu);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = etsig[i]*etsig[i] + jit*jit;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(et[i] - mu[i], 2)/var);
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
    {
        logL = std::numeric_limits<double>::infinity();
    }
    return logL;
}


void ETmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<extra_sigma<<'\t';


    out.precision(24);
    if (ephemeris >= 1) out << ephem1 << '\t';
    if (ephemeris >= 2) out << ephem2 << '\t';
    if (ephemeris == 3) out << ephem3 << '\t';
    out.precision(8);

    //auto data = get_data();

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
        out << '\t' << nu << '\t';

}


string ETmodel::description() const
{
    string desc;
    string sep = "   ";

    desc += "extra_sigma   ";

    if (ephemeris >= 1) desc += "ephem1" + sep;
    if (ephemeris >= 2) desc += "ephem2" + sep;
    if (ephemeris == 3) desc += "ephem3" + sep;

    //auto data = get_data();

    if(known_object) { // KO mode!
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_P" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_K" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "K" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;
    if (studentt)
        desc += "nu" + sep;

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void ETmodel::save_setup() {
    auto data = get_data();
    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "ETmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "ephemeris: " << ephemeris << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data.datafile << endl;
    fout << "skip: " << data.dataskip << endl;

    fout << "files: ";
    for (auto f: data.datafiles)
        fout << f << ",";
    fout << endl;

    fout.precision(15);
    fout << "M0_epoch: " << data.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Jprior: " << *Jprior << endl;

    if (ephemeris >= 1) fout << "ephem1_prior: " << *ephem1_prior << endl;
    if (ephemeris >= 2) fout << "ephem2_prior: " << *ephem2_prior << endl;
    if (ephemeris == 3) fout << "ephem3_prior: " << *ephem3_prior << endl;

    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "Kprior: " << *conditional->Kprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "wprior: " << *conditional->wprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *KO_Kprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

