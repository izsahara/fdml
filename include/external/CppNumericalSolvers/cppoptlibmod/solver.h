// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_

#include <functional>
#include <iostream>
#include <tuple>
#include <iomanip>
#include "function.h"

namespace cppoptlib::solver {
    using namespace cppoptlib::function;

    // Status of the solver state.
    enum class SolverStatus {
          NotStarted = -1,
          Continue = 0,     // Optimization should continue.
          IterationLimit,   // Maximum of allowed iterations has been reached.
          XDeltaViolation,  // Minimum change in parameter vector has been reached.
          FDeltaViolation,  // Minimum chnage in cost function has been reached.
          GradientNormViolation,  // Minimum norm in gradient vector has been reached.
          HessianConditionViolation  // Maximum condition number of hessian_t has been reached.
    };

    inline std::ostream &operator<<(std::ostream &stream, const SolverStatus &status) {
    switch (status) {
        case SolverStatus::NotStarted:
          stream << "Solver not started.";
          break;
        case SolverStatus::Continue:
          stream << "Convergence criteria not reached.";
          break;
        case SolverStatus::IterationLimit:
          stream << "Iteration limit reached.";
          break;
        case SolverStatus::XDeltaViolation:
          stream << "Change in parameter vector too small.";
          break;
        case SolverStatus::FDeltaViolation:
          stream << "Change in cost function value too small.";
          break;
        case SolverStatus::GradientNormViolation:
          stream << "Gradient vector norm too small.";
          break;
        case SolverStatus::HessianConditionViolation:
          stream << "Condition of Hessian/Covariance matrix too large.";
          break;
      }
      return stream;
    }

    // The state of the solver.
    struct SolverState {

        /*
        * num_iterations     : Maximum number of allowed iterations.
        * x_delta:           : Minimum change in parameter vector.
        * f_delta            : Minimum change in cost function.
        * gradient_norm      : Minimum norm of gradient vector.
        * condition_hessian  : Maximum condition number of hessian_t.        
        * f_delta_violations : Number of violations in cost function.
        * x_delta_violations : Number of violations in pareameter vector.
        */

        size_t num_iterations = 0;
        double x_delta = double{0};
        double f_delta = double{0};  
        double gradient_norm = double{0}; 
        double condition_hessian = double{0};
        int x_delta_violations = 0;
        int f_delta_violations = 0;
        std::vector<TVector> x_history;
        std::vector<double> f_history;


        // SolverStatus of state.
        SolverStatus status = SolverStatus::NotStarted; 
        SolverState() = default;

        // Updates state from function information.
        void update(const FunctionState prev_fstate, const FunctionState curr_fstate, const SolverState &stop_state) {
            num_iterations++;
            f_delta = fabs(curr_fstate.value - prev_fstate.value);
            x_delta = (curr_fstate.x - prev_fstate.x).template lpNorm<Eigen::Infinity>();
            gradient_norm = curr_fstate.gradient.template lpNorm<Eigen::Infinity>();
            x_history.push_back(curr_fstate.x);
            f_history.push_back(curr_fstate.value);

            if ((stop_state.num_iterations > 0) && (num_iterations == stop_state.num_iterations)) {
                status = SolverStatus::IterationLimit;
                return;
            }
            if ((stop_state.x_delta > 0) && (x_delta < stop_state.x_delta)) {
                x_delta_violations++;
                if (x_delta_violations >= stop_state.x_delta_violations) 
                {
                status = SolverStatus::XDeltaViolation;
                return;
                }
            } 
            else { x_delta_violations = 0; }
            if ((stop_state.f_delta > 0) && (f_delta < stop_state.f_delta)) {
                f_delta_violations++;
                if (f_delta_violations >= stop_state.f_delta_violations) {
                status = SolverStatus::FDeltaViolation;
                return;
                } else { f_delta_violations = 0; }
            }
            if ((stop_state.gradient_norm > 0) && (gradient_norm < stop_state.gradient_norm)) {
                status = SolverStatus::GradientNormViolation;
                return;
            }
            if (prev_fstate.order == 2) {
                if ((stop_state.condition_hessian > 0) && (condition_hessian > stop_state.condition_hessian)) {
                status = SolverStatus::HessianConditionViolation;
                return;
                }
            }
            status = SolverStatus::Continue;
        }
        };

    SolverState StoppingState() {
      SolverState state;
      state.num_iterations = 10000;
      state.x_delta = double{1e-9};
      state.x_delta_violations = 5;
      state.f_delta = double{1e-9};
      state.f_delta_violations = 5;
      state.gradient_norm = double{1e-4};
      state.condition_hessian = double{0};
      state.status = SolverStatus::NotStarted;
      return state;
    }

    // Callbacks
    using Callback = std::function<void(const FunctionState &, const SolverState &, int iter)>;

    Callback DefaultVerbose() 
    {
      return
        [](const FunctionState &fstate, const SolverState &solstate, int iter) 
        {

            std::cout << "Function-State" << "\t";
            std::cout << "  value    " << fstate.value << "\t";
            std::cout << "  x    " << fstate.x.transpose() << "\t";
            std::cout << "  gradient    " << fstate.gradient.transpose() << std::endl;
            std::cout << "Solver-State" << "\t";
            std::cout << "  iterations " << solstate.num_iterations << "\t";
            std::cout << "  x_delta " << solstate.x_delta << "\t";
            std::cout << "  f_delta " << solstate.f_delta << "\t";
            std::cout << "  gradient_norm " << solstate.gradient_norm << "\t";
            std::cout << "  condition_hessian " << solstate.condition_hessian << std::endl;
        };
    }

    Callback EmptyVerbose() { return [](const FunctionState& fstate, const SolverState& solstate, int iter) { }; }

    class Solver {
    public:
        explicit Solver(const SolverState &stopping_state = StoppingState()) : stopping_state(stopping_state), callback(EmptyVerbose()) {}

        virtual ~Solver() = default;
        virtual int order() const { return 1; }
        virtual void initialize_solver(const FunctionState & /*initial_state*/) {}
        void SetStepCallback(Callback step_callback) { callback = step_callback; }
        // Minimizes a given function and returns the function state
        virtual std::tuple<FunctionState, SolverState> minimize(const Function &function, const TVector &x0, const int& restart, int& verbosity) 
        { return this->minimize(function, function.evaluate(x0, this->order()), restart, verbosity); }
        virtual std::tuple<FunctionState, SolverState> minimize(const Function& function, const FunctionState& initial_state, const int& restart, int& verbosity)
        {
            // Solver state during the optimization.
            SolverState solver_state;
            // Function state during the optimization.
            FunctionState function_state(initial_state);

            this->initialize_solver(initial_state);
            int iter = 0;
            do {
                if (verbosity == 2) {
                    if (iter == 0) {
                    std::cout << std::setw(10) << std::left << "RESTART " 
                              << std::setw(2) << std::left << restart
                              << std::setw(17) << std::right << "FUNCTION VALUE"
                              << std::setw(17) << std::right << "DELTA_X"
                              << std::setw(18) << std::right << "DELTA_F" << std::endl;
                    }
                }                
                // Trigger a user-defined callback.
                this->callback(function_state, solver_state, iter); iter++;
                // Find next function state.
                FunctionState previous_function_state(function_state);
                function_state = this->step(function, previous_function_state, solver_state);
                // Update current solver state.
                solver_state.update(previous_function_state, function_state, stopping_state);
            } while (solver_state.status == SolverStatus::Continue);

            // Final Trigger of a user-defined callback.
            this->callback(function_state, solver_state, iter);
            if (verbosity == 1) {
                std::string header(25, '=');
                std::cout << header << " RESTART " << restart << " " << header << std::endl;
            }
            else if (verbosity == 2) 
            { std::system("cls"); }
            return { function_state, solver_state };
        }

        virtual FunctionState step(const Function &function, const FunctionState &current, const SolverState &state) = 0;

    protected:
        SolverState stopping_state;
        Callback    callback;  
    };


}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
