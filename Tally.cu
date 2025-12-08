#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <iomanip>

#include "RNG.cuh"
#include "Neutron.cuh"
#include "FuelKernel.cuh"
#include "Tally.cuh"

//#define TALLYDEBUG

__host__
void Tally::fluxTally2D_host(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide, int loopIdx) {
	std::cout << "Flux tally for " << loopIdx << "th Cycle\n";
	int totalNeutronNum = Neutrons.neutronSize + Neutrons.addedNeutronSize;
	double widthPerCell = Slab3D.D_x / numRegionsPerSide;
	int lengthCellCount = static_cast<int>(Slab3D.D_y / widthPerCell);
	std::vector<std::vector<int>> populationArr(numRegionsPerSide, std::vector<int>(lengthCellCount, 0));
	std::cout << Neutrons.neutrons[2].pos.x << " x pos\n";
	for (int i = 0; i < Neutrons.allocatableNeutronNum; i++) {
		if (!Neutrons.neutrons[i].isNullified()) {
			int widthIndex = static_cast<int>(Neutrons.neutrons[i].pos.x / widthPerCell);
			int lengthIndex = static_cast<int>(Neutrons.neutrons[i].pos.y / widthPerCell);
			// 이거 밑에 이건 좀 나중에도 보고 배워라 이래놓으니까 벡터 인덱스 에러가 나지 시 ㅂ
			//if (widthIndex >= 0 && widthIndex <= numRegionsPerSide && lengthIndex >= 0 && lengthIndex <= lengthCellCount) {
			if (widthIndex >= 0 && widthIndex < numRegionsPerSide && lengthIndex >= 0 && lengthIndex < lengthCellCount) {
				populationArr[widthIndex][lengthIndex]++;
			}
#ifdef TALLYDEBUG
			else { 
				std::cout << "Neutron " << i << " excluded becasue pos was shit: width and lenght index was: " << widthIndex << " " << lengthIndex << "    ";
				Neutrons.neutrons[i].printInfo();
				
			}
#endif
		}
		if (!Neutrons.addedNeutrons[i].isNullified()) {
			int widthIndex = static_cast<int>(Neutrons.addedNeutrons[i].pos.x / widthPerCell);
			int lengthIndex = static_cast<int>(Neutrons.addedNeutrons[i].pos.y / widthPerCell);
			if (widthIndex >= 0 && widthIndex < numRegionsPerSide && lengthIndex >= 0 && lengthIndex < lengthCellCount) {
				populationArr[widthIndex][lengthIndex]++;
			}
#ifdef TALLYDEBUG
			else {
				std::cout << "addedNeutron " << i << " excluded becasue pos was shit: width and lenght index was: " << widthIndex << " " << lengthIndex << "    ";
				Neutrons.addedNeutrons[i].printInfo();
			}
#endif
		}
	}

	std::ofstream outFile;
	std::string filename = "output/Loop_" + std::to_string(loopIdx) + "_fluxTally.txt";
	outFile.open(filename);
	
	for (int i = 0; i < lengthCellCount; i++) {
		for (int j = 0; j < numRegionsPerSide; j++) {
			outFile << populationArr[j][i] << "  ";
		}
		outFile << "\n";
	}

	outFile.close();
}

__host__ 
void Tally::fluxTally2D_host_HeightSpecific(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide, double height) {

}

__device__
void Tally::fluxTally2D_device(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide) {
	
}

__host__
void Tally::fluxTally3D_host(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide) {
	
}

__device__
void Tally::fluxTally3D_device(NeutronDistribution Neutrons, ReflectiveSlab Slab3D, int numRegionsPerSide) {
	
}