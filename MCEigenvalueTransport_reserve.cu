#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


//#include "DebugFlag.cuh"

#include "RNG.cuh"
#include "Neutron.cuh"
#include "FuelKernel.cuh"

//#define DEBUG
//#define NUMNEUTRONSPEC


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

  // ...


__global__ void ReadCounters(NeutronDistribution* Neutrons, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = Neutrons->neutronSize;
        out[1] = Neutrons->addedNeutronSize;
    }
}


__global__ void SingleCycle_Neutron(ReflectiveSlab* Slab3D, NeutronDistribution* Neutrons, unsigned long long* seedNo, double* k_mult, int* fissionSIG, int* fisNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }
    GnuAMCM RNG(seedNo[idx]);

    

    if (!Neutrons->neutrons[idx].isNullified()) {
        //if (Neutrons->neutrons[idx].status == true) { printf("neutron %d idx active\n", idx); }
        // run MC for neutrons
        double mainNeutron_DTC = Slab3D->DTC(Neutrons->neutrons[idx], RNG);
        vec3 dirNormal{};
        double mainNeutron_DTS = Slab3D->DTS(Neutrons->neutrons[idx], dirNormal);

        //printf("pos: idx %d : %f, %f, %f,\t", idx, Neutrons->neutrons[idx].pos.x, Neutrons->neutrons[idx].pos.y, Neutrons->neutrons[idx].pos.z);
        //printf("DTC: idx %d : %f, DTS: %f\n", idx, mainNeutron_DTC, mainNeutron_DTS);

        //if (idx == 3) { atomicAdd(&(Neutrons->neutronSize), -1); }

        if (mainNeutron_DTS > mainNeutron_DTC) { // reaction!!!
            //if (true) {
            // step the neutron forward
            Neutrons->neutrons[idx].updateWithLength(mainNeutron_DTC);
            // now the reaction

            ReactionType RType = Slab3D->getInteractionType(Neutrons->neutrons[idx], RNG);

            //if (true) {
            if (RType == ReactionType::capture) {
                #ifdef DEBUG 
                    printf("neutron capture on idx %d\n", idx); 
                #endif
                Slab3D->absorption(Neutrons->neutrons[idx]);
                atomicAdd(&(Neutrons->neutronSize), -1);
            }
            else if (RType == ReactionType::scatter) {
                //printf("scatter on idx %d\n", idx);
                Neutrons->neutrons[idx].dirVec = vec3::randomUnit(RNG);
            }
            else {
                #ifdef DEBUG 
                    printf("neutron fission on idx %d\n", idx);
                #endif
                
                Slab3D->fission(Neutrons->neutrons[idx], Neutrons, RNG, k_mult, false, fisNum);
            }

        }
        else {
            Slab3D->reflection(Neutrons->neutrons[idx], mainNeutron_DTC, mainNeutron_DTS, dirNormal, RNG, idx);
            // 이새끼를 recursion으로 풀려고 하니까 ㅈㄹ났던거임 - 함수안에 for loop으로 바꿈
        }
    }

#ifdef DEBUG
    Neutrons->neutrons[idx].printInfo_Kernel(idx);
#endif

    seedNo[idx] = RNG.gen();
}

__global__ void printNeutronsInKernel(NeutronDistribution* Neutrons) {
    int idx = threadIdx.x + blockIdx.x + blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }
    Neutrons->neutrons[idx].printInfo_Kernel(idx);
}

__global__ void printAddedNeutronsInKernel(NeutronDistribution* Neutrons) {
    int idx = threadIdx.x + blockIdx.x + blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }
    Neutrons->addedNeutrons[idx].printInfo_Kernel(idx);
}


__global__ void addedNeutronPassResetter(NeutronDistribution* Neutrons) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > Neutrons->allocatableNeutronNum) { return; }

    if (!Neutrons->addedNeutrons[idx].isNullified()) {
        Neutrons->addedNeutrons[idx].passFlag = false;
    }
}

__global__ void SingleCycle_addedNeutron(ReflectiveSlab* Slab3D, NeutronDistribution* Neutrons, unsigned long long* seedNo, double* k_mult, int* fissionSIG, int* fisNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }
    GnuAMCM RNG(seedNo[idx]);

    

    if (!Neutrons->addedNeutrons[idx].isNullified()) {
        if (Neutrons->addedNeutrons[idx].passFlag) {
#ifdef DEBUG
            printf("idx %d passed because it passFlag was true\n", idx);
#endif
            Neutrons->addedNeutrons[idx].passFlag = false;
            return;
        }
        //if (true) {
        //printf("addedNeutron is working!\n");
        double addedNeutron_DTC = Slab3D->DTC(Neutrons->addedNeutrons[idx], RNG);
        vec3 dirNormal{};
        double addedNeutron_DTS = Slab3D->DTS(Neutrons->addedNeutrons[idx], dirNormal);

        if (addedNeutron_DTS > addedNeutron_DTC) {
            Neutrons->addedNeutrons[idx].updateWithLength(addedNeutron_DTC);
            ReactionType addedNeutronRType = Slab3D->getInteractionType(Neutrons->addedNeutrons[idx], RNG);
            if (addedNeutronRType == ReactionType::capture) {
                //if (true) {
                Slab3D->absorption(Neutrons->addedNeutrons[idx]);
#ifdef DEBUG
                printf(" capture in addedNeutron idx %d\n", idx);
#endif
                atomicAdd(&(Neutrons->addedNeutronSize), -1);
            }
            else if (addedNeutronRType == ReactionType::scatter) {
                Neutrons->addedNeutrons[idx].dirVec = vec3::randomUnit(RNG);
            }
            else {
#ifdef DEBUG
                printf("addedNeutron fission on idx %d\n", idx);
#endif
                *fissionSIG = 1;
                Slab3D->fission(Neutrons->addedNeutrons[idx], Neutrons, RNG, k_mult, true, fisNum);
            }
        }
        else {
            Slab3D->reflection(Neutrons->addedNeutrons[idx], addedNeutron_DTC, addedNeutron_DTS, dirNormal, RNG, idx);
        }

    }
#ifdef DEBUG
    Neutrons->addedNeutrons[idx].printInfo_Kernel(idx);
#endif
    seedNo[idx] = RNG.gen();
}



/*
// for now lets design our program to 
__global__ void SingleCycle(ReflectiveSlab* Slab3D, NeutronDistribution* Neutrons, unsigned long long* seedNo, double* k_mult, int* fissionSIG) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }
    GnuAMCM RNG(seedNo[idx]);

    if (!Neutrons->addedNeutrons[idx].isNullified()) {
        // for freshly made neutorns - check it
        if (Neutrons->addedNeutrons[idx].passFlag) {
            Neutrons->addedNeutrons[idx].passFlag = false;
            return;
        }

        //if (true) {
        //printf("addedNeutron is working!\n");
        double addedNeutron_DTC = Slab3D->DTC(Neutrons->addedNeutrons[idx], RNG);
        vec3 dirNormal{};
        double addedNeutron_DTS = Slab3D->DTS(Neutrons->addedNeutrons[idx], dirNormal);

        if (addedNeutron_DTS > addedNeutron_DTC) {
            Neutrons->addedNeutrons[idx].updateWithLength(addedNeutron_DTC);
            ReactionType addedNeutronRType = Slab3D->getInteractionType(Neutrons->addedNeutrons[idx], RNG);
            if (addedNeutronRType == ReactionType::capture) {
                //if (true) {
                Slab3D->absorption(Neutrons->addedNeutrons[idx]);
               // printf(" absorption in addedNeutron idx %d\n", idx);
                atomicAdd(&(Neutrons->addedNeutronSize), -1);
            }
            else if (addedNeutronRType == ReactionType::scatter) {
                Neutrons->addedNeutrons[idx].dirVec = vec3::randomUnit(RNG);
            }
            else {
                *fissionSIG = 1;
                Slab3D->fission(Neutrons->addedNeutrons[idx], Neutrons, RNG, k_mult, false);
            }
        }
        else {
            int counter = 0;
            Slab3D->reflection(Neutrons->addedNeutrons[idx], addedNeutron_DTC, addedNeutron_DTS, dirNormal, RNG, counter);
        }

    }

    if (!Neutrons->neutrons[idx].isNullified()) {
        //if (Neutrons->neutrons[idx].status == true) { printf("neutron %d idx active\n", idx); }
        // run MC for neutrons
        double mainNeutron_DTC = Slab3D->DTC(Neutrons->neutrons[idx], RNG);
        vec3 dirNormal{};
        double mainNeutron_DTS = Slab3D->DTS(Neutrons->neutrons[idx], dirNormal);

        //printf("pos: idx %d : %f, %f, %f,\t", idx, Neutrons->neutrons[idx].pos.x, Neutrons->neutrons[idx].pos.y, Neutrons->neutrons[idx].pos.z);
        //printf("DTC: idx %d : %f, DTS: %f\n", idx, mainNeutron_DTC, mainNeutron_DTS);

        //if (idx == 3) { atomicAdd(&(Neutrons->neutronSize), -1); }

        if (mainNeutron_DTS > mainNeutron_DTC) { // reaction!!!
            //if (true) {
            Neutrons->neutrons[idx].updateWithLength(mainNeutron_DTC);
            // now the reaction

            ReactionType RType = Slab3D->getInteractionType(Neutrons->neutrons[idx], RNG);

            //if (true) {
            if (RType == ReactionType::capture) {
                //printf("capture on idx %d\n", idx);
                Slab3D->absorption(Neutrons->neutrons[idx]);
                //printf("Im working!\n");
                atomicAdd(&(Neutrons->neutronSize), -1);
            }
            else if (RType == ReactionType::scatter) {
                //printf("scatter on idx %d\n", idx);
                Neutrons->neutrons[idx].dirVec = vec3::randomUnit(RNG);
            }
            else {
                //printf("fission on idx %d\n", idx);
                Slab3D->fission(Neutrons->neutrons[idx], Neutrons, RNG, k_mult, true);
            }

        }
        else {
            int counter = 0;
            Slab3D->reflection(Neutrons->neutrons[idx], mainNeutron_DTC, mainNeutron_DTS, dirNormal, RNG, counter);
            // 이새끼를 recursion으로 풀려고 하니까 ㅈㄹ났던거임 - 함수안에 for loop으로 바꿈
        }
    }


    seedNo[idx] = RNG.gen();

}
*/


int main() {
    int initialNeutronNum = 500000;  
    //int excessNumNeutron = initialNumNeutrons * 1.5;
    unsigned long long seedNo = 92235942397;
    int numCycle = 510;
    int initialOffset = 20;
    int threadPerBlock = 32;
    int blockPerDim = (initialNeutronNum + threadPerBlock - 1) / threadPerBlock;

     /*
      *  ___ _  _  ___     ___ ___ _____ _   _ ___ 
      * | _ \ \| |/ __|   / __| __|_   _| | | | _ \
      * |   / .` | (_ |   \__ \ _|  | | | |_| |  _/
      * |_|_\_|\_|\___|   |___/___| |_|  \___/|_|  
      *                                            
      */
    GnuAMCM RNG(seedNo);
    unsigned long long* h_SeedArr = new unsigned long long[initialNeutronNum];
    for (int i = 0; i < initialNeutronNum; i++) {
        /*
        for (int j = 0; j < 3999; j++) {
            RNG.gen_static();
        }
        h_SeedArr[i] = RNG.gen();
        */
        h_SeedArr[i] = (RNG.gen() + 2 * i) & (0xFFFFFFFFFFFFULL);
    }

    unsigned long long* d_SeedArr = nullptr; 
    cudaMalloc(&d_SeedArr, initialNeutronNum * sizeof(unsigned long long));
    cudaMemcpy(d_SeedArr, h_SeedArr, initialNeutronNum * sizeof(unsigned long long), cudaMemcpyHostToDevice);

     /*
      *  _  _          _                    ___      _             
      * | \| |___ _  _| |_ _ _ ___ _ _     / __| ___| |_ _  _ _ __ 
      * | .` / -_) || |  _| '_/ _ \ ' \    \__ \/ -_)  _| || | '_ \
      * |_|\_\___|\_,_|\__|_| \___/_||_|   |___/\___|\__|\_,_| .__/
      *                                                      |_|   
      */
    // build host struct
    NeutronDistribution h_Neutrons(initialNeutronNum, RNG.gen());
    h_Neutrons.setUniformNeutrons(0.02, 0.02, 0.02);
    
    // build device struct pointer. allocate only the space for it.
    NeutronDistribution* d_Neutrons = nullptr;
    cudaMalloc(&d_Neutrons, sizeof(NeutronDistribution));

    // This is redundant - I'll explain here.
    //cudaMemcpy(d_Neutrons, &h_Neutrons, sizeof(NeutronDistribution), cudaMemcpyHostToDevice);
    // From this Memcpy, d_Neutrons only have info of numNeutrons, seedNo, and only the address of the first array element (i.e. &neutrons[0]).
    // Rest of the array (neutrons[1], neutrons[2]) are lost - cannot be deferred from device perspective.
    // Thus, we need to declare device side arrays that contains the data of h_Neutron.neutrons.
    // Other than that, we are going to Memcpy the tmp_Neutrons to d_Neutrons anyways.
    

    // allocate device buffer arrays - this will be used to feed the device 'actual' neutron arrays.
    Neutron* d_bufferNeutrons = nullptr;
    Neutron* d_bufferAddedNeutrons = nullptr;

    // allocate and copy the data from the 'actual' host neutron arrays.
    cudaMalloc(&d_bufferNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron));
    cudaMemcpy(d_bufferNeutrons, h_Neutrons.neutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bufferAddedNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron));
    cudaMemcpy(d_bufferAddedNeutrons, h_Neutrons.addedNeutrons, h_Neutrons.neutronSize * sizeof(Neutron), cudaMemcpyHostToDevice);
    
    
    // build a temporary host-side copy of object, where the device array buffer will reside in.
    // For the neutron array, it only contains the first Neutron element address of host-side array.
    // i.e. tmp_Neutrons is a shallow copy of h_Neutrons.
    NeutronDistribution tmp_Neutrons = h_Neutrons;

    // Here, we feed this object a 'device-side neutron address'.
    // this is again a shallow copy - tmp_Neutrons.neutrons and d_bufferNeutrons points to same address.
    tmp_Neutrons.neutrons = d_bufferNeutrons;
    tmp_Neutrons.addedNeutrons = d_bufferAddedNeutrons;
    // now this temporary object's member neutrons* have device array pointers.

    // copy the temporary object containing device array buffer to actual device struct
    cudaMemcpy(d_Neutrons, &tmp_Neutrons, sizeof(NeutronDistribution), cudaMemcpyHostToDevice);
    // now d_Neutrons have full info of h_Neutrons.    
    
    // make this temporary object's neutron* member to not refer the device pointer.
    //tmp_Neutrons.neutrons = nullptr;
    //tmp_Neutrons.addedNeutrons = nullptr;
    // This is very important, since as this object goes out of the scope, it will call a destructor.
    // When the destructor is called (i.e. delete[] neutrons (a type Neutron*)), it will deallocate the device pointer.
    // This is bad - host objects aren't meant to deallocate device object/pointers. It's done with cudaFree.
    // If you do it, it will caused undefined behavior - usually crash, error on delete_scalar.cpp

    // 25.12.2: new error: if you want to copy back the neutrons, or do any operation related to it, it will 




    // setup the 2 banks: 
    
    // Some fucking values to pass around - like multiplication factor (k) or other shits nigga i wanna fuck myself
    int h_mergeSignal = 0;
    int* d_mergeSignal = nullptr;
    cudaMalloc(&d_mergeSignal, sizeof(int));
    cudaMemcpy(d_mergeSignal, &h_mergeSignal, sizeof(int), cudaMemcpyHostToDevice);
    
    int h_fissionSIG = 0;
    int* d_fissionSIG = nullptr;
    cudaMalloc(&d_fissionSIG, sizeof(int));
    cudaMemcpy(d_fissionSIG, &h_fissionSIG, sizeof(int), cudaMemcpyHostToDevice);

     /*
      *  ___         _     ___      _             
      * | __|  _ ___| |   / __| ___| |_ _  _ _ __ 
      * | _| || / -_) |   \__ \/ -_)  _| || | '_ \
      * |_| \_,_\___|_|   |___/\___|\__|\_,_| .__/
      *                                     |_|   
      */

    // Currently its a 3d CUBE, 
    // parametrs:             x,    y,    z,    XS_c,XS_s,XS_f,nu
    ReflectiveSlab h_CubeSlab(0.02, 0.02, 0.02, 0.2, 1.5, 0.4, 2.0); 
    ReflectiveSlab* d_CubeSlab = nullptr;
    cudaMalloc(&d_CubeSlab, sizeof(ReflectiveSlab));
    cudaMemcpy(d_CubeSlab, &h_CubeSlab, sizeof(ReflectiveSlab), cudaMemcpyHostToDevice);


    NeutronDistribution h_NeutronsReceiver = h_Neutrons;
    NeutronDistribution meta(h_NeutronsReceiver.allocatableNeutronNum, RNG.gen());
    int h_fissionSourceNum = initialNeutronNum;
    int* d_fissionSourceNum = nullptr;
    cudaMalloc(&d_fissionSourceNum, sizeof(int));
    cudaMemcpy(d_fissionSourceNum, &h_fissionSourceNum, sizeof(int), cudaMemcpyHostToDevice);

    double h_multK = 1.2;
    double* d_multK = nullptr;
    cudaMalloc(&d_multK, sizeof(double));
    cudaMemcpy(d_multK, &h_multK, sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> k_Tally;
    //numCycle = 110;


    for (int i = 0; i < numCycle; i++) {
        int previousFissionSourceNum = h_fissionSourceNum;
        int previousGenNeutronNum = h_NeutronsReceiver.neutronSize + h_NeutronsReceiver.addedNeutronSize;
#ifdef NUMNEUTRONSPEC
        std::cout << "on the beginnig of the loop, previousGenNumNeutron: "
        std::cout << previousGenNeutronNum;
        std::cout << " --- each: NeutronSize: " << h_NeutronsReceiver.neutronSize << ", addedNeutronSize: " << h_NeutronsReceiver.addedNeutronSize << ", addindex: " << h_NeutronsReceiver.addedNeutronIndex << "\n";
#endif
        //SingleCycle << <blockPerDim, threadPerBlock >> > (d_CubeSlab, d_Neutrons, d_SeedArr, d_multK, d_fissionSIG);
#ifdef DEBUG
        std::cout << "AddedNeutron's info:\n";
#endif
        SingleCycle_addedNeutron << <blockPerDim, threadPerBlock >> > (d_CubeSlab, d_Neutrons, d_SeedArr, d_multK, d_fissionSIG, d_fissionSourceNum);
        CUDA_CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
        std::cout << "\nNeutron's Info:\n";
#endif
        addedNeutronPassResetter << <blockPerDim, threadPerBlock >> > (d_Neutrons);
        CUDA_CHECK(cudaDeviceSynchronize());
        

        SingleCycle_Neutron << <blockPerDim, threadPerBlock >> > (d_CubeSlab, d_Neutrons, d_SeedArr, d_multK, d_fissionSIG, d_fissionSourceNum);
        CUDA_CHECK(cudaDeviceSynchronize());


        cudaMemcpy(&h_mergeSignal, d_mergeSignal, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_fissionSIG, d_fissionSIG, sizeof(int), cudaMemcpyDeviceToHost);
        //if (h_fissionSIG == 1) {
        //    std::cout << "\n FISSION!\n";
        //}
        


        //std::cout << "copying device structs to host...\n";
        cudaMemcpy(&meta, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost);
        h_NeutronsReceiver.neutronSize = meta.neutronSize;
        h_NeutronsReceiver.addedNeutronSize = meta.addedNeutronSize;
        h_NeutronsReceiver.addedNeutronIndex = meta.addedNeutronIndex;

        //h_Neutrons.neutronSize = h_NeutronsReceiver.neutronSize;
        //cudaMemcpy(h_NeutronsReceiver.neutrons, d_bufferNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_NeutronsReceiver.addedNeutrons, d_bufferAddedNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyDeviceToHost);
        //std::cout << "copying ended!\n";

        //cudaMemcpy(&h_Neutrons, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost); // will this just copy the plain values ?
        




        

        if (static_cast<double>(h_NeutronsReceiver.addedNeutronIndex) > 0.8 * static_cast<double>(h_Neutrons.allocatableNeutronNum)) {
            std::cout << "addIndex almost full - sorting and merging neutrons...\n";

            

            //cudaMemcpy(&h_NeutronsReceiver, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost);

            
            
            /*
            CUDA_CHECK(cudaMemcpy(&meta, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost));

            h_NeutronsReceiver.neutronSize = meta.neutronSize;
            h_NeutronsReceiver.addedNeutronSize = meta.addedNeutronSize;
            h_NeutronsReceiver.addedNeutronIndex = meta.addedNeutronIndex;
            */
#ifdef NUMNEUTRONSPEC
            std::cout << "Data from meta struct: " << h_NeutronsReceiver.neutronSize << ", " << h_NeutronsReceiver.addedNeutronSize << ", " << h_NeutronsReceiver.addedNeutronIndex << "\n";
#endif
            // first copy the device array to host
            cudaMemcpy(h_NeutronsReceiver.neutrons, d_bufferNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_NeutronsReceiver.addedNeutrons, d_bufferAddedNeutrons, h_Neutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyDeviceToHost);

            //cudaMemcpy(&h_NeutronsReceiver, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost);


            std::vector<Neutron> NeutronContainer;

            
            for (int j = 0; j < h_NeutronsReceiver.allocatableNeutronNum; j++) {
#ifdef DEBUG
                std::cout << "idx" << j << ", info: ";
                h_NeutronsReceiver.neutrons[j].printInfo();
#endif
                if (!h_NeutronsReceiver.neutrons[j].isNullified()) {
                    h_NeutronsReceiver.neutrons[j].passFlag = false;
                    NeutronContainer.push_back(h_NeutronsReceiver.neutrons[j]);
                    
                }
            }
            for (int j = 0; j < h_NeutronsReceiver.allocatableNeutronNum; j++) {
#ifdef DEBUG
                std::cout << "idx" << j << ", info: ";
                h_NeutronsReceiver.addedNeutrons[j].printInfo();
#endif
                if (!h_NeutronsReceiver.addedNeutrons[j].isNullified()) {
                    h_NeutronsReceiver.addedNeutrons[j].passFlag = false;
                    NeutronContainer.push_back(h_NeutronsReceiver.addedNeutrons[j]);

                }
            }

            

            // Allocate a new struct for merging
            NeutronDistribution h_MergedNeutrons(h_Neutrons.allocatableNeutronNum, 1);

            // push scalar values
            // note: there can be 2 cases: if total number of neutrons exceed the main array - fill the addedNeutron array
            // else, only fill the main array.
            if (NeutronContainer.size() < h_Neutrons.allocatableNeutronNum) {
                h_Neutrons.neutronSize = NeutronContainer.size(); // assume neutron will be full
                h_Neutrons.addedNeutronSize = 0;
                h_Neutrons.addedNeutronIndex = 0;

                h_MergedNeutrons.neutronSize = NeutronContainer.size(); // assume neutron will be full
                h_MergedNeutrons.addedNeutronSize = 0;
                h_MergedNeutrons.addedNeutronIndex = 0;
                std::cout << "Neutron vector size: " << NeutronContainer.size() << "\n";

                std::cout << "Assigning the neutrons to the host merge array:\n";
                for (int j = 0; j < NeutronContainer.size(); j++) { // error in this line
                    h_MergedNeutrons.neutrons[j] = NeutronContainer[j];
                }
            }
            else {

                h_Neutrons.neutronSize = h_Neutrons.allocatableNeutronNum; // assume neutron will be full
                h_Neutrons.addedNeutronSize = NeutronContainer.size() - h_Neutrons.allocatableNeutronNum;
                h_Neutrons.addedNeutronIndex = h_Neutrons.addedNeutronSize;

                h_MergedNeutrons.neutronSize = h_Neutrons.allocatableNeutronNum; // assume neutron will be full
                h_MergedNeutrons.addedNeutronSize = NeutronContainer.size() - h_Neutrons.allocatableNeutronNum;
                h_MergedNeutrons.addedNeutronIndex = h_MergedNeutrons.addedNeutronSize;
                std::cout << "Neutron vector size: " << NeutronContainer.size() << "\n";

                std::cout << "Assigning the neutrons to the host merge array:\n";
                for (int j = 0; j < h_Neutrons.allocatableNeutronNum; j++) { // error in this line
                    h_MergedNeutrons.neutrons[j] = NeutronContainer[j];
#ifdef DEBUG
                    std::cout << "neutrons idx " << j << " : ";
                    h_MergedNeutrons.neutrons[j].printInfo();
#endif
                }
                for (int j = 0; j < h_Neutrons.addedNeutronSize; j++) {
                    h_MergedNeutrons.addedNeutrons[j] = NeutronContainer[h_Neutrons.neutronSize + j];
#ifdef DEBUG
                    std::cout << "addedNeutrons idx " << j << " : ";
                    h_MergedNeutrons.addedNeutrons[j].printInfo();
#endif
                }
            }

            
            

            
            cudaMemcpy(d_bufferNeutrons, h_MergedNeutrons.neutrons, h_MergedNeutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bufferAddedNeutrons, h_MergedNeutrons.addedNeutrons, h_MergedNeutrons.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyHostToDevice);

            h_MergedNeutrons.neutrons = d_bufferNeutrons;
            h_MergedNeutrons.addedNeutrons = d_bufferAddedNeutrons;


            cudaMemcpy(d_Neutrons, &h_MergedNeutrons, sizeof(NeutronDistribution), cudaMemcpyHostToDevice);
            // this is enough -> tmp_Neutrons already have address for d_bufferNeutrons arrays, and it points to the d_Neutorns anyways.

            std::cout << "Neutron size after merging: for main neutron array: " << h_NeutronsReceiver.neutronSize << ", for added: " << h_Neutrons.addedNeutronSize << "\n";
            

        }

        cudaMemcpy(&h_fissionSourceNum, d_fissionSourceNum, sizeof(int), cudaMemcpyDeviceToHost);

        int nextFissionSourceNum = h_fissionSourceNum;

        int currentGenNeutronNum = h_NeutronsReceiver.neutronSize + h_NeutronsReceiver.addedNeutronSize;

        //h_multK *= static_cast<double>(currentGenNeutronNum) / static_cast<double>(previousGenNeutronNum);

        h_multK = static_cast<double>(nextFissionSourceNum) / static_cast<double>(previousFissionSourceNum);

        //h_multK = static_cast<double>(currentGenNeutronNum) / static_cast<double>(previousGenNeutronNum);

        cudaMemcpy(d_multK, &h_multK, sizeof(double), cudaMemcpyHostToDevice);

        std::cout << "Loop " << i;
#ifdef NUMNEUTRONSPEC
        std::cout << ": after Kernel";
        std::cout << ", : neutron: " << h_NeutronsReceiver.neutronSize << " , addedNeutron: " << h_NeutronsReceiver.addedNeutronSize << " , addIndex: " << h_NeutronsReceiver.addedNeutronIndex << "\n";
#endif
        std::cout << " total Neutron : " << currentGenNeutronNum << ", multiplication factor k: " << std::fixed << std::setprecision(6) << h_multK << "\n";
        std::cout << "fission neutron number was: " << nextFissionSourceNum << "\n";
        if (i >= initialOffset) {
            k_Tally.push_back(h_multK);
        }
    }

    double avg = 0.0;
    double stdev = 0.0;
    for (int i = 0; i < numCycle-initialOffset; i++) {
        avg += k_Tally[i];
    }
    avg /= (numCycle - initialOffset);

    for (int i = 0; i < numCycle-initialOffset; i++) {
        stdev += std::pow((k_Tally[i] - avg), 2);
    }
    stdev /= (numCycle - initialOffset);

    std::cout << "\nFor total of ~" << initialNeutronNum << " neutrons, result of K-Tally over " << numCycle - initialOffset << " cycles:\n";
    std::cout << "k = " << avg << " pm " << stdev << "\n";




    delete[] h_SeedArr;
    

    tmp_Neutrons.neutrons = nullptr;
    tmp_Neutrons.addedNeutrons = nullptr;
    cudaFree(d_SeedArr);
    cudaFree(d_Neutrons);
    cudaFree(d_bufferAddedNeutrons);
    cudaFree(d_bufferNeutrons);
    cudaFree(d_CubeSlab);
    cudaFree(d_fissionSIG);

    
    cudaDeviceReset();
}