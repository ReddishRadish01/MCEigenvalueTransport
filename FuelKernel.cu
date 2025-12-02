#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <math.h>


#include "Constants.cuh"
#include "RNG.cuh"
#include "Neutron.cuh"
#include "FuelKernel.cuh"

// returns [m]
__host__ __device__
double ReflectiveSlab::DTC(Neutron incidentNeutron, GnuAMCM& RNG) {
	double distance = this->averageDistance(incidentNeutron, RNG);
	return distance;
}

// returns [m], and Surface normal of headed direction <- REQUIRED!!! I used reference passing for Colliding surface normal
__host__ __device__
double ReflectiveSlab::DTS(Neutron& incidentNeutron, vec3& dirNormal) {
	// fml i need to find the way to verify how this shit is going to work man
	//if (incidentNeutron.isNullified()) return 0;
	//else {  }

	// we need to decide the equation for the surface: left or right? use dot product signs.
	// assume plane x = 0 has normal pointing to -x direction (i.e. [-1, 0, 0]) and x = D_x has normal of [1, 0, 0]
	vec3 x_0 = { -1, 0, 0 };
	vec3 x_Dx = { 1, 0, 0 };


	// Noted N_ prefix to show that the vector is a surface normal - e.g. N_yz_Pos : "N"ormal vector, of "yz" plane, facing "Pos"itive
	// This can be removed later? for performace
	vec3 N_yz_Pos = { 1, 0, 0 };
	vec3 N_yz_Neg = { -1, 0, 0 };
	vec3 N_xy_Pos = { 0, 0, 1 };
	vec3 N_xy_Neg = { 0, 0, -1 };
	vec3 N_xz_Pos = { 0, 1, 0 };
	vec3 N_xz_Neg = { 0, -1, 0 };

	// distance to each plane - we decide which face (pos or neg) the particle is heading, and assign the length)
	// tX: 
	double tX = 0.0; vec3 tX_Dir = N_yz_Pos;
	if (incidentNeutron.dirVec.x > 0.0) { tX = (this->D_x - incidentNeutron.pos.x) / incidentNeutron.dirVec.x; }
	else if (incidentNeutron.dirVec.x < 0.0) { tX = -incidentNeutron.pos.x / incidentNeutron.dirVec.x; tX_Dir = N_yz_Neg; }

	double tY = 0.0; vec3 tY_Dir = N_xz_Pos;
	if (incidentNeutron.dirVec.y > 0.0) { tY = (this->D_y - incidentNeutron.pos.y) / incidentNeutron.dirVec.y; }
	else if (incidentNeutron.dirVec.y < 0.0) { tY = -incidentNeutron.pos.y / incidentNeutron.dirVec.y; tY_Dir = N_xz_Neg; }
	
	double tZ = 0.0; vec3 tZ_Dir = N_xy_Pos;
	if (incidentNeutron.dirVec.x > 0.0) { tZ = (this->D_z - incidentNeutron.pos.z) / incidentNeutron.dirVec.z; }
	else if (incidentNeutron.dirVec.x < 0.0) { tZ = -incidentNeutron.pos.z / incidentNeutron.dirVec.z; tZ_Dir = N_xy_Neg; }


	// tHit : Distance to the collision (DTC)
	// No need for sqrt() shit - our direction vector is already normalized
	double tHit = 1.8E+108; // very close to the maximium value of double, IEEE 754
	if (tX > 0.0 && tX < tHit) {
		tHit = tX;
		dirNormal = tX_Dir;
	}
	if (tY > 0.0 && tY < tHit) {
		tHit = tY;
		dirNormal = tY_Dir;
	}
	if (tZ > 0.0 && tZ < tHit) {
		tHit = tZ;
		dirNormal = tZ_Dir;
	}
	return tHit;
}


__host__ __device__
void ReflectiveSlab::reflection(Neutron& incidentNeutron, double DTC, double DTS, vec3 dirNormal, GnuAMCM& RNG, int& counter) {
	if (counter > 10) {
		incidentNeutron.Nullify();
	}
	// when given: d=directional vector of particle, n=surface normal
	// reflected direction \vec{r} = d - 2 (d \cdot n) n    <---- meh this fucking thing ahhh so hard to intuitively think about - its my skill issue
	vec3 collisionPos = incidentNeutron.pos + incidentNeutron.dirVec * DTS;
	vec3 reflectVec = incidentNeutron.dirVec - dirNormal * (2 * incidentNeutron.dirVec.dot(dirNormal));

	vec3 afterCollisionPos = collisionPos + reflectVec * (DTC - DTS);
	if (this->outOfRange(afterCollisionPos)) {
		// move to DTC
		incidentNeutron.pos = collisionPos;
		incidentNeutron.dirVec = reflectVec;
		vec3 reflectSurfaceNormal(0, 0, 0);

		// calculate DTC and DTS for updatd Neutron
		double _DTC = this->DTC(incidentNeutron, RNG);
		double _DTS = this->DTS(incidentNeutron, reflectSurfaceNormal);
		counter++;
		// recursion - carry out reflection in updated Neutron
		this->reflection(incidentNeutron, _DTC, _DTS, reflectSurfaceNormal, RNG, counter);
	}
	else {
		incidentNeutron.pos = collisionPos + reflectVec * (DTC - DTS);
		incidentNeutron.dirVec = reflectVec;
	}

}

__host__ __device__
void ReflectiveSlab::elasticScattering(Neutron& incidentNeutron, GnuAMCM& RNG) {
	incidentNeutron.dirVec = vec3::randomUnit(RNG);
}

__host__ __device__
void ReflectiveSlab::absorption(Neutron& incidentNeutron) {
	incidentNeutron.Nullify();
}

__device__
void ReflectiveSlab::fission(Neutron& incidentNeutron, NeutronDistribution* Neutrons, GnuAMCM& RNG, int* k) {
	int fissionNum = int(this->nu / *k + RNG.uniform(0.0, 1.0));
	// assign as fission neutron;
	incidentNeutron.dirVec = vec3::randomUnit(RNG);
	// this is problematic? im not sure
	for (int i = 0; i < fissionNum - 1; i++) {
		Neutrons->addedNeutrons[Neutrons->addedNeutronIndex].dirVec = vec3::randomUnit(RNG);
		atomicAdd(&(Neutrons->addedNeutronIndex), 1);
		// this fucking atomicAdd is __device__ or __global__ propreitary functions - no hosts
	}
}

__host__ __device__ double ReflectiveSlab::calculate_k(NeutronDistribution& Neutrons, int& previousNum) {
	double k = Neutrons.neutronSize + Neutrons.addedNeutronSize / double(previousNum);
	previousNum = Neutrons.neutronSize + Neutrons.addedNeutronSize;
	return k;
}
