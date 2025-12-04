#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <math.h>
#include <cmath>
#include <stdlib.h>

#include "thrustHeader.cuh"
#include "RNG.cuh"
#include "Constants.cuh"



enum AngleType {
	Degree,
	Radian
};

struct Spherical;

struct vec3 {
	double x, y, z;

	__host__ __device__ vec3()
		: x(), y(), z()
	{}

	__host__ __device__ vec3(double xx, double yy, double zz)
		: x(xx), y(yy), z(zz)
	{}


	__host__ __device__ vec3 operator-(const vec3 vec) const;
	__host__ __device__ vec3 operator+(const vec3 vec) const;
	__host__ __device__ vec3 operator*(const double coeff) const;
	__host__ __device__ vec3 operator/(const double coeff) const;

	__host__ __device__ vec3 cross(const vec3 vec) const;
	__host__ __device__ double dot(const vec3 vec) const;

	__host__ __device__ double magnitude() const;

	__host__ __device__ vec3 normalize() const;
	
	// this is static - there's no need to make a void function for randomizing vectors.
	__host__ __device__ static vec3 randomUnit(GnuAMCM& RNG);

	/*__host__ __device__ static vec3 randomUnit(GnuAMCM& RNG) {	// static

		//vec3 randUnitVec = { static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()) };
		//vec3 randUnitVec = { localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0) };

		double phi = RNG.uniform(0, 1) * 2 * Constants::PI;
		double theta = acos(2 * RNG.uniform(0, 1) - 1);
		Spherical sphericalDir(theta, phi, AngleType::Radian);
		return sphericalDir.convToVec3();
	}
	*/

	__host__ __device__ Spherical convToSpherical() const;
};

// always in radian, but accepts degree type,
// seems like we are using physics/engineering context spherical coordinate
// theta is the polar angle, and phi is azimuthal angle
struct Spherical {
	double theta, phi;	// always in radian
	double r;
	AngleType angleType;
	
	
	__host__ __device__ Spherical()
		: theta(0.0), phi(0.0), r(0.0), angleType(AngleType::Radian)
	{}

	__host__ __device__ Spherical(double theta, double phi, AngleType angleType, double r = 1.0)
		: theta(theta), phi(phi), angleType(angleType), r(r)
	{
		if (angleType == AngleType::Degree) {
			// Degree to radian
			this->theta = this->theta * atan(1.0) * 4.0 / 180.0;
			this->phi = this->phi * atan(1.0) * 4.0 / 180.0;
			//this->theta = this->theta * Constants::PI / 180.0;
			//this->phi = this->phi * Constants::PI / 180.0;
			this->angleType = AngleType::Radian;
		}
	}
	
	__host__ __device__ vec3 convToVec3() const;
};


struct Neutron {
	vec3 pos;			// METERS !!!
	vec3 dirVec;		// UNIT VECTOR
	double energy;	// eV !!!
	bool status;
	bool passFlag;

	__host__ __device__ Neutron()
		: pos({ 0.0, 0.0, 0.0 }), dirVec({ 0.0, 0.0, 0.0 }), energy(0.0), status(false), passFlag(true) {}

	__host__ __device__ Neutron(vec3 pos, vec3 dirVec, double energy)
		: pos(pos), dirVec(dirVec), energy(energy), status(true), passFlag(false)
	{}

	__host__ __device__ Neutron(vec3 pos, Spherical sphercial, double energy) 
		: pos(pos), dirVec(sphercial.convToVec3()), energy(energy), status(true), passFlag(false)
	{

	}

	__host__ __device__ double Velocity() const;
	__host__ __device__ vec3 VelocityVec() const;

	__host__ __device__ void Nullify();
	__host__ __device__ bool isNullified() const;
	__host__ __device__ void reInitialize(vec3 pos, vec3 dir, double energy);

	__host__ __device__ void updateWithLength(double length);
	
	__host__ inline void printInfo() {
		std::cout << "(" << this->pos.x << ", " << this->pos.y << ", " << this->pos.z << "),  ";
		std::cout << "(" << this->dirVec.x << ", " << this->dirVec.y << ", " << this->dirVec.z << ") , status: ";
		if (this->status) { std::cout << " true.\n"; }
		else { std::cout << " false.\n"; }
	}

};


struct NeutronDistribution {
	Neutron* neutrons;
	Neutron* addedNeutrons;
	int neutronSize;
	int allocatableNeutronNum;
	int addedNeutronSize;
	int addedNeutronIndex;
	unsigned long long seedNo;

	__host__ __device__ NeutronDistribution( unsigned int initialNeutronNum, unsigned long long seedNo)
		: neutrons(new Neutron[initialNeutronNum]), addedNeutrons(new Neutron[initialNeutronNum]), neutronSize(initialNeutronNum), allocatableNeutronNum(initialNeutronNum),
		addedNeutronSize(0), addedNeutronIndex(0), seedNo(seedNo)
	{}

	
	__host__ __device__ ~NeutronDistribution() {
		// Note this delete[] operation will also deallocate the device side shits:
		// you must nullptr a temporary objects containing device pointers.
		// this will be manually done at the end of the main file
		//delete[] neutrons;
		//delete[] addedNeutrons;
	}

	__host__ __device__ inline int getTotalNeutronNum() {
		return this->addedNeutronSize + this->neutronSize;
	}
	

	__host__ __device__ void setNeutrons(Spherical Dir, double energy);
	__host__ __device__ void setUniformNeutrons(double D_x, double D_y, double D_z);
	//__host__ __device__ void updateAddedNeutronStatus();
};

struct NeutronThrustDevice {
	Neutron* neutrons;
	Neutron* addedNeutrons;
	unsigned int neutronNumber;
	unsigned int addedNeutronNumber;
	unsigned long long seedNo;

	__host__ __device__ NeutronThrustDevice(Neutron* neutrons, Neutron* addedNeutrons, unsigned int neutronNumber, unsigned int addedNeutronNumber, unsigned long long seedNo)
		:neutrons(neutrons), addedNeutrons(addedNeutrons), neutronNumber(neutronNumber), addedNeutronNumber(addedNeutronNumber), seedNo(seedNo)
	{}
};


struct NeutronThrustHost {
	thrust::host_vector<Neutron> neutrons;
	thrust::host_vector<Neutron> addedNeutrons;
	unsigned long long seedNo;

	__host__ NeutronThrustHost(int initialSize, unsigned long long seedNo)
		: neutrons(initialSize), addedNeutrons(initialSize), seedNo(seedNo)                                                                                                                                                                                                      
	{
		// You dont have to initialize the neutrons in here. the above neutrons(initialSize) does the work for you.
		// if you really want to do, do: 
		//neutrons.resize(initialSize);
		//addedNeutrons.resize(initialSize); 
	}

	__host__ NeutronThrustDevice HtoD(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutron);

	__host__ void DtoH(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons);

	__host__ void setUniformNeutron(double D_x, double D_y, double D_z);

};

struct NeutronThrustManager {
	thrust::device_vector<Neutron> d_neutrons;
	thrust::device_vector<Neutron> d_addedNeutrons;
	unsigned long long seedNo;
	

	// guess its kinda redundnat?
	NeutronThrustManager(thrust::device_vector<Neutron>& d_neutrons, thrust::device_vector<Neutron>& d_addedNeutrons, unsigned long long seedNo)
		: d_neutrons(d_neutrons), d_addedNeutrons(d_addedNeutrons), seedNo(seedNo)
	{
	}


	// host fucking specific
	__host__ __device__ static void mergeCheck(NeutronDistribution& Neutrons);
	__host__ static void MergeNeutron(NeutronDistribution& Neutrons);
	
};