#ifndef PARAMETERS_H
#define PARAMETERS_H
#include "base.h"

namespace fdsm::parameters {

    /* Constraints */
    template<typename T>
    T transform_fxn(const T& value, std::string& method, bool inverse = false) { T a; return a; };

    template<>
    TVector transform_fxn(const TVector& value, std::string& method, bool inverse) {
        if (method == "logexp") {
            if (inverse) { return log(value.array()); }
            else { return exp(value.array()); }
        }
        else { return value; }
    }

    template<>
    double transform_fxn(const double& value, std::string& method, bool inverse) {
        if (method == "logexp") {
            if (inverse) { return log(value); }
            else { return exp(value); }
        }
        else { return value; }
    }


    template<typename T>
    class BaseParameter {
    protected:
        T _value;
        std::string _name;
        std::pair<T, T> _bounds;
        std::size_t _size = 1;
        bool _fixed = false;
        std::string _transform = "none";
    public:
        //const T& value = _value;
        //const std::string& name = _name;
        //std::pair<T, T> bounds;
        const bool* is_fixed = &_fixed;
    public:

        // ========== Shared Interface ========== //
        BaseParameter() = default;
        BaseParameter(const BaseParameter&) = default;
        BaseParameter(BaseParameter&&) = default;
        BaseParameter(std::string name, T value, std::string transform, std::pair<T, T> bounds) : _name(name), _value(value), _transform(transform), _bounds(bounds) {}

        virtual void fix() { _fixed = true; is_fixed = &_fixed; }
        virtual void unfix() { _fixed = false; is_fixed = &_fixed; }
        virtual void set_constraint(std::string constraint) {
            if (constraint == "none" || constraint == "logexp") {
                _transform = constraint;
            }
            else { throw std::runtime_error("Unrecognized Constraint"); }
        }

        // ========== C++ Interface ========== //

        const std::size_t& size() { return _size; }
        virtual void transform_value(bool inverse = false) {
            _value = transform_fxn(_value, _transform, inverse);
        }
        virtual void transform_value(const T& new_value, bool inverse = false) {
            _value = transform_fxn(new_value, _transform, inverse);
        }
        virtual void transform_bounds(bool inverse = false) {
            T tmp1, tmp2;
            tmp1 = transform_fxn(_bounds.first, _transform, inverse);
            tmp2 = transform_fxn(_bounds.second, _transform, inverse);
            _bounds = std::make_pair(tmp1, tmp2);
        }
        virtual void transform_bounds(const std::pair<T, T>& new_bounds, bool inverse = false) {
            T tmp1, tmp2;
            tmp1 = transform_fxn(new_bounds.first, _transform, inverse);
            tmp2 = transform_fxn(new_bounds.second, _transform, inverse);
            _bounds = std::make_pair(tmp1, tmp2);
        }

        BaseParameter& operator=(const T& oValue) {
            if (_fixed) { throw std::runtime_error("Error fixed value.."); }
            _value = oValue;
            return *this;
        }
        BaseParameter& operator=(const BaseParameter& oParam) {
            _size = oParam._size;
            _name = oParam._name;
            _value = oParam._value;
            _fixed = oParam._fixed;
            _transform = oParam._transform;
            _bounds = oParam._bounds;
            return *this;
        }
        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const BaseParameter<U>& param);

        // Setters
        virtual void set_value(const T& new_value) {
            if (_fixed) { throw std::runtime_error("Error fixed value.."); }
            _value = new_value;
        };
        virtual void set_name(const std::string& new_name) { _name = new_name; };
        virtual void set_transform(const std::string& new_transform) { _transform = new_transform; };
        virtual void set_bounds(const std::pair<T, T>& new_bounds) = 0;


        virtual T value() const { return _value; }
        virtual const std::string name() const { return _name; }

        // Getters
        virtual const T get_value() const { return _value; };
        virtual const std::string get_name() const { return _name; }
        virtual const std::string get_transform() const { return _transform; }
        virtual const std::pair<T, T> get_bounds() const { return _bounds; }
        virtual const bool fixed() const { return _fixed; }

    };

    template<typename T>
    std::ostream& operator<<(std::ostream& stream, const BaseParameter<T>& param) {
        return stream << param.value;
    }


    template <typename T>
    class Parameter;

    template<>
    class Parameter<double> : public BaseParameter<double>
    {
        
    public:        

        Parameter() : BaseParameter() {}
        Parameter(std::string name, double value)  {
            _name = name;
            _value = value;
            _transform = "none";
            _bounds = std::make_pair(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
        }
        Parameter(std::string name, double value, std::pair<double, double> bounds) {
            _name = name;
            _value = value;
            _transform = "none";
            _bounds = bounds;
        }

        Parameter(std::string name, double value, std::string transform) {
            _name = name;
            _value = value;
            _transform = transform;
            _bounds = std::make_pair(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
        }
        Parameter(std::string name, double value, std::string transform, std::pair<double, double> bounds) : BaseParameter(name, value, transform, bounds) {}
     
        void set_bounds(const std::pair<double, double>& new_bounds) override {
            if (_bounds.first > _bounds.second) { throw std::runtime_error("Lower Bound > Upper Bound"); }
            _bounds = new_bounds;
        }

        // C++ Interface
        Parameter& operator=(const double& oValue)
        {
            if (_fixed) { throw std::runtime_error("Error fixed value.."); }
            _value = oValue;
            return *this;
        }
        
    };

    template<>
    class Parameter<TVector> : public BaseParameter<TVector>
    {
    protected:

    public:
        Parameter() : BaseParameter() {}
        Parameter(std::string name, TVector value) {
            _name = name;
            _value = value;
            _transform = "none";
            _size = value.size();
            TVector lower_bound(_size);
            TVector upper_bound(_size);
            for (int i = 0; i < _size; ++i) { lower_bound[i] = -std::numeric_limits<double>::infinity(); }
            for (int i = 0; i < _size; ++i) { upper_bound[i] = std::numeric_limits<double>::infinity(); }
            _bounds = std::make_pair(lower_bound, upper_bound);
        }
        Parameter(std::string name, TVector value, std::string transform) {
            _name = name;
            _value = value;
            _transform = transform;
            _size = value.size();
            TVector lower_bound(_size);
            TVector upper_bound(_size);
            for (int i = 0; i < _size; ++i) { lower_bound[i] = -std::numeric_limits<double>::infinity(); }
            for (int i = 0; i < _size; ++i) { upper_bound[i] = std::numeric_limits<double>::infinity(); }
            _bounds = std::make_pair(lower_bound, upper_bound);
        }
        Parameter(std::string name, TVector value, std::pair<TVector, TVector> bounds) {
            _name = name;
            _value = value;
            _transform = "none";
            _size = value.size();
            _bounds = bounds;
        }
        Parameter(std::string name, TVector value, std::string transform, std::pair<TVector, TVector> bounds) : BaseParameter(name, value, transform, bounds) { _size = value.size(); }
        
        void set_bounds(const std::pair<TVector, TVector>& new_bounds) override {
            if ((_bounds.first.array() > _bounds.second.array()).any())
            { throw std::runtime_error("Lower Bound > Upper Bound"); }
            _bounds = new_bounds;
        }

        // C++ Interface
        Parameter& operator=(const TVector& oValue)
        {
            if (_fixed) { throw std::runtime_error("Error fixed value.."); }
            _value = oValue;
            _size = _value.rows();
            return *this;
        }
        double& operator[](const Eigen::Index& idx) { return _value[idx]; }

    };
}
#endif