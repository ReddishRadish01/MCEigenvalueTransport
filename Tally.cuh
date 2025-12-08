#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <iomanip>


#include "RNG.cuh"
#include "Neutron.cuh"
#include "FuelKernel.cuh"


// this might be host specific - maybe it's static functions can be device called tho
// 이걸 네임스페이스로 넣어야 하나 하 모르겟노 ㅅㅂ
struct Tally {
	Tally() {}

	~Tally() {}

	__host__
	static void fluxTally2D_host(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide);

	__device__
	static void fluxTally2D_device(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide);

	__host__
	static void fluxTally3D_host(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide);

	__device__
	static void fluxTally3D_device(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide);
};