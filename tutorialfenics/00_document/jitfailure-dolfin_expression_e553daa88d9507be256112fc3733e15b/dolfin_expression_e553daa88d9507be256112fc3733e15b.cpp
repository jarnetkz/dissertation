
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_e553daa88d9507be256112fc3733e15b : public Expression
  {
     public:
       double A;
double t;
double L;
double domain_size;
double K1;
double K2;
double D;


       dolfin_expression_e553daa88d9507be256112fc3733e15b()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = (x[0] <= domain_size) ? (A*sin(2*np.pi*t) * x[0] / (K2*D)) : (A*sin(2*np.pi*t) * (1.0 - (L - x[0]) / (K1*D)));

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "A") { A = _value; return; }          if (name == "t") { t = _value; return; }          if (name == "L") { L = _value; return; }          if (name == "domain_size") { domain_size = _value; return; }          if (name == "K1") { K1 = _value; return; }          if (name == "K2") { K2 = _value; return; }          if (name == "D") { D = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "A") return A;          if (name == "t") return t;          if (name == "L") return L;          if (name == "domain_size") return domain_size;          if (name == "K1") return K1;          if (name == "K2") return K2;          if (name == "D") return D;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_e553daa88d9507be256112fc3733e15b()
{
  return new dolfin::dolfin_expression_e553daa88d9507be256112fc3733e15b;
}

