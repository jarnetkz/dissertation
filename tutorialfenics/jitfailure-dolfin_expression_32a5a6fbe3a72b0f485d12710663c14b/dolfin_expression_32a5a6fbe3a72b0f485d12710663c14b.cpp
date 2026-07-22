
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
  class dolfin_expression_32a5a6fbe3a72b0f485d12710663c14b : public Expression
  {
     public:
       std::shared_ptr<dolfin::GenericFunction> generic_function_v;
std::shared_ptr<dolfin::GenericFunction> generic_function_D;
double L;


       dolfin_expression_32a5a6fbe3a72b0f485d12710663c14b()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          double D;
            generic_function_D->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&D), x);
          double v[2];

            generic_function_v->eval(Eigen::Map<Eigen::Matrix<double, 2, 1>>(v), x);
          values[0] = (exp(v*x[0]/D) - 1)/(exp(v*L/D) - 1);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "L") { L = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "L") return L;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {
          if (name == "v") { generic_function_v = _value; return; }          if (name == "D") { generic_function_D = _value; return; }
       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {
          if (name == "v") return generic_function_v;          if (name == "D") return generic_function_D;
       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_32a5a6fbe3a72b0f485d12710663c14b()
{
  return new dolfin::dolfin_expression_32a5a6fbe3a72b0f485d12710663c14b;
}

