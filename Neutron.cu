//#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <math.h>


#include "RNG.cuh"
#include "Neutron.cuh"
#include "Constants.cuh"




__host__ __device__ vec3 vec3::operator-(const vec3 vec) const {
	return { x - vec.x, y - vec.y, z - vec.z };
}
__host__ __device__ vec3 vec3::operator+(const vec3 vec) const {
	return { x + vec.x, y + vec.y, z + vec.z };
}
__host__ __device__ vec3 vec3::operator*(const double coeff) const {
	return { x * coeff, y * coeff, z * coeff };
}

__host__ __device__ vec3 vec3::operator/(const double coeff) const {
	return { x / coeff, y / coeff, z / coeff };
}
__host__ __device__ vec3 vec3::cross(const vec3 vec) const {
	return {
		y * vec.z - z * vec.y,
		z * vec.x - x * vec.z,
		x * vec.y - y * vec.x
	};
}
__host__ __device__ double vec3::dot(const vec3 vec) const {
	return x * vec.x + y * vec.y + z * vec.z;
}

__host__ __device__ double vec3::magnitude() const {
	return sqrt(x * x + y * y + z * z);
}

__host__ __device__ vec3 vec3::normalize() const {
	return {
		x / magnitude(),
		y / magnitude(),
		z / magnitude()
	};
}


__host__ __device__ vec3 vec3::randomUnit(GnuAMCM& RNG) {	// static
	// wrong!!!!
	//vec3 randUnitVec = { static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()) };
	//vec3 randUnitVec = { localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0) };

	double phi = RNG.uniform(0, 1) * 2 * Constants::PI;
	double theta = acos(2 * RNG.uniform(0, 1) - 1);
	Spherical sphericalDir(theta, phi, AngleType::Radian);
	return sphericalDir.convToVec3();
}


__host__ __device__  Spherical vec3::convToSpherical() const {
	double theta, phi;
	if (this->magnitude() == 1) {
		theta = acos(z);
		phi = atan2(y, x);
	}
	else {
		vec3 normalizedVec = this->normalize();
		theta = acos(normalizedVec.z);
		phi = atan2(normalizedVec.y, normalizedVec.x);
	}

	return { theta, phi, AngleType::Radian, 0 };
}

__host__ __device__ double Neutron::Velocity() const {
	return sqrt(2 * energy * Constants::ElectronC / (Constants::M_Neutron * Constants::amuToKilogram));
	// Constants namespace's atom mass always have gram/mol (i.e. amu) - convert it to kg
}

__host__ __device__ vec3 Neutron::VelocityVec() const {
	return dirVec * this->Velocity();
}

__host__ __device__ void Neutron::Nullify() {
	this->pos = { 0.0, 0.0, 0.0 };
	this->dirVec = { 0.0, 0.0, 0.0 };
	this->energy = 0.0;
	this->status = false;

}

__host__ __device__ void Neutron::reInitialize(vec3 pos, vec3 dirVec, double energy) {
	this->pos = pos;
	this->dirVec = dirVec;
	this->energy = energy;
	this->status = true;
}

__host__ __device__ bool Neutron::isNullified() const {
	if (this->status) { return true; }
	else { return false; }
}

__host__ __device__ void Neutron::updateWithLength(double length) { 
	this->pos.x += length * this->dirVec.x;
	this->pos.y += length * this->dirVec.y;
	this->pos.z += length * this->dirVec.z;
}

__host__ __device__ vec3 Spherical::convToVec3() const{
	double x = r * sin(this->theta) * cos(this->phi);
	double y = r * sin(this->theta) * sin(this->phi);
	double z = r * cos(this->theta);

	return { x, y, z };
}



__host__ __device__ void NeutronDistribution::setNeutrons(Spherical dir, double energy) {
	GnuAMCM RNG(this->seedNo);
	for (int i = 0; i < int(this->neutronSize); i++) {
		this->neutrons[i].pos.x = 0.0;
		this->neutrons[i].pos.y = 0.0;
		this->neutrons[i].pos.z = 0.0;
		//this->neutrons[i].m_pos.y = RNG.uniform(-1.0, 1.0);
		//this->neutrons[i].m_pos.z = RNG.uniform(-1.0, 1.0);

		this->neutrons[i].dirVec = dir.convToVec3();
		this->neutrons[i].energy = energy;
	}
}

__host__ __device__ 
void NeutronDistribution::setUniformNeutrons(double D_x, double D_y, double D_z) {
	GnuAMCM RNG(this->seedNo);
	for (int i = 0; i < int(this->neutronSize); i++) {
		this->neutrons[i].pos.x = RNG.uniform_open(0, D_x);
		this->neutrons[i].pos.y = RNG.uniform_open(0, D_y);
		this->neutrons[i].pos.z = RNG.uniform_open(0, D_z);
		this->neutrons[i].dirVec = vec3::randomUnit(RNG);
		//this->neutrons[i].energy = 0.0;
	}
	
}

__host__ void NeutronDistribution::updateAddedNeutronStatus() {
	this->addedNeutronSize = this->addedNeutronIndex;

}


__host__ NeutronThrustDevice NeutronThrustHost::HtoD(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons) {
	return NeutronThrustDevice{ thrust::raw_pointer_cast(d_Neutrons.data()), thrust::raw_pointer_cast(d_addedNeutrons.data()),
		(unsigned int)(d_Neutrons.size()), (unsigned int)(d_addedNeutrons.size()), seedNo };
}
	
__host__ void NeutronThrustHost::DtoH(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons) {
	cudaDeviceSynchronize();

	
}
	
__host__ void NeutronThrustHost::setUniformNeutron(double D_x, double D_y, double D_z) {
	
}


__host__ __device__ void NeutronThrustManager::mergeCheck(NeutronDistribution& Neutrons) {
	if (Neutrons.addedNeutronSize > 0.9 * Neutrons.allocatableNeutrons) {
		NeutronThrustManager::MergeNeutron(Neutrons);
	}
	else { return; }
}

__host__ void NeutronThrustManager::MergeNeutron(NeutronDistribution& Neutrons) {
	std::vector<Neutron> neutronManager;
	neutronManager.reserve(Neutrons.neutronSize + Neutrons.addedNeutronSize);

	for (unsigned int i = 0; i < Neutrons.neutronSize; i++) {
		if (!Neutrons.neutrons[i].isNullified()) {
			neutronManager.push_back(Neutrons.neutrons[i]);
			Neutrons.neutrons[i].Nullify();
		}
	}

	for (unsigned int i = 0; i < Neutrons.addedNeutronSize; i++) {
		if (!Neutrons.addedNeutrons[i].isNullified()) {
			neutronManager.push_back(Neutrons.addedNeutrons[i]);
			Neutrons.addedNeutrons[i].Nullify();
		}
	}

	int managerSize = neutronManager.size();

	for (int i = 0; i < managerSize; i++) {
		Neutrons.neutrons[i] = neutronManager[i];
	}
}

/*
__host__ void NeutronThrustManager::MergeNeutron() {
	auto e1 = thrust::remove_if(d_neutrons.begin(), d_neutrons.end(), [] __device__(Neutron const& n) { return n.isNullified(); });
	d_neutrons.erase(e1, d_neutrons.end());

	auto e2 = thrust::remove_if(d_addedNeutrons.begin(), d_addedNeutrons.end(), [] __device__(Neutron const& n) { return n.isNullified(); });
	d_addedNeutrons.erase(e2, d_addedNeutrons.end());

	size_t neutronN = d_neutrons.size();
	size_t addedNeutronN = d_addedNeutrons.size();

	d_neutrons.reserve(neutronN + addedNeutronN);
	thrust::copy(d_addedNeutrons.begin(), d_addedNeutrons.end(), std::back_inserter(d_neutrons));
	d_addedNeutrons.clear();
}
*/