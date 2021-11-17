#ifndef RELIABILITY_H
#define RELIABILITY_H

#include "./base_models.h"

namespace fdml::reliability {
	using namespace fdml::base_models;

	class AKMCS {

	public:
		AKMCS(const GPR& model) : model(model) {}


	public:
		GPR& model;

	};
}




#endif