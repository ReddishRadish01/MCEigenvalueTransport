#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>
#include <cmath>
#include <iomanip>

#include "Constants.cuh"
#include "RNG.cuh"
#include "Neutron.cuh"


enum ReactionType {
	scatter,
	capture,
	fission
};

// 
struct ReflectiveSlab {
	double D_x, D_y, D_z;			// meters
	double MacroXS_c;	// cm^{-1}
	double MacroXS_s;	// cm^{-1}
	double MacroXS_f;	// cm^{-1}
	double nu;

	ReflectiveSlab()
		:D_x(0), D_y(0), D_z(0), MacroXS_c(0), MacroXS_s(0), MacroXS_f(0), nu(0)
	{
	}

	ReflectiveSlab(double D_x, double D_y, double D_z, double MacroXS_c, double MacroXS_s, double MacroXS_f, double nu)
		:D_x(D_x), D_y(D_y), D_z(D_z), MacroXS_c(MacroXS_c), MacroXS_s(MacroXS_s), MacroXS_f(MacroXS_f), nu(nu)
	{
	}

	__host__ __device__
	inline ReactionType getInteractionType(Neutron incidentNeutron, GnuAMCM& RNG) {
		double cumulativeXS = this->MacroXS_c + this->MacroXS_s + this->MacroXS_f;
		double rngNo = RNG.uniform(0.0, cumulativeXS);
		if (rngNo < this->MacroXS_c) { return ReactionType::capture; }
		else if (rngNo < this->MacroXS_c + MacroXS_s) { return ReactionType::scatter; }
		else { return ReactionType::fission; }
	}

	// returns [cm^{-1}]
	__host__ __device__
	double inline getInteracitonXS(Neutron incidentNeutron, GnuAMCM& RNG) {
		ReactionType rType = this->getInteractionType(incidentNeutron, RNG);
		if (rType == ReactionType::capture) { return this->MacroXS_c; }
		else { return this->MacroXS_s; }
	}

	__host__ __device__
	bool inline outOfRange(Neutron incidentNeutron) const {
		double posX = incidentNeutron.pos.x;
		double posY = incidentNeutron.pos.y;
		double posZ = incidentNeutron.pos.z;

		if (posX < 0 || posX > D_x || posY < 0 || posY > D_y || posZ < 0 || posZ > D_z) { return true; }
		else { return false; }
	}

	__host__ __device__ 
	bool inline outOfRange(vec3 pos) const {
		if (pos.x < 0 || pos.x > D_x || pos.y < 0 || pos.y > D_y || pos.z < 0 || pos.z > D_z) { return true; }
		else { return false; }
	}

	// returns [m].
	__host__ __device__
	inline double averageDistance(Neutron incidentNeutron, GnuAMCM& RNG) {
		double distance = -1.0 / (this->MacroXS_c + this->MacroXS_s + this->MacroXS_f) / 100 * log(RNG.uniform_open(0.0, 1.0));
		return distance;
	}

	__host__ __device__ double DTC(Neutron incidentNeutron, GnuAMCM& RNG);

	__host__ __device__ double DTS(Neutron& incidentNeutron, vec3& dirNormal);

	__host__ __device__ void reflection(Neutron& incidentNeutron, double DTC, double DTS, vec3 dirNormal, GnuAMCM& RNG, int idx);

	__host__ __device__ void elasticScattering(Neutron& incidentNeutron, GnuAMCM& RNG);


	__host__ __device__ void absorption(Neutron& incidentNeutron);

	//__device__ void fission(Neutron& incidentNeutron, NeutronDistribution* Neutrons, GnuAMCM& RNG, double* k, bool passFlag, int* fisNum);
	__device__ void fission(Neutron& incidentNeutron, NeutronDistribution* Neutrons, GnuAMCM& RNG, double* k, bool passFlag);

	__host__ __device__ double calculate_k(NeutronDistribution& Neutrons, int& previousCount);
};