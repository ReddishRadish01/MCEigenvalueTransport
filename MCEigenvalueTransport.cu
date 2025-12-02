#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#include "RNG.cuh"
#include "Neutron.cuh"
#include "FuelKernel.cuh"

 /*
  *  ___     ___      _ _ _         _       __  __                            _             _       __  __         _    _          
  * |_ _|   | _ )_  _(_) | |_      /_\     |  \/  |___ _ __  ___ _ _ _  _    | |   ___ __ _| |__   |  \/  |__ _ __| |_ (_)_ _  ___ 
  *  | |    | _ \ || | | |  _|    / _ \    | |\/| / -_) '  \/ _ \ '_| || |   | |__/ -_) _` | / /   | |\/| / _` / _| ' \| | ' \/ -_)
  * |___|   |___/\_,_|_|_|\__|   /_/ \_\   |_|  |_\___|_|_|_\___/_|  \_, |   |____\___\__,_|_\_\   |_|  |_\__,_\__|_||_|_|_||_\___|
  *                                                                  |__/                                                          
  *
  *  ___                _               _   _    _        ___         _            _   _           ___             _                ___ ___ _   _ 
  * | _ \_  _ _ _  _ _ (_)_ _  __ _    | |_| |_ (_)___   |_ _|_ _  __| |_ __ _ _ _| |_| |_  _     / __|_ _ __ _ __| |_  ___ ___    / __| _ \ | | |
  * |   / || | ' \| ' \| | ' \/ _` |   |  _| ' \| (_-<    | || ' \(_-<  _/ _` | ' \  _| | || |   | (__| '_/ _` (_-< ' \/ -_)_-<   | (_ |  _/ |_| |
  * |_|_\\_,_|_||_|_||_|_|_||_\__, |    \__|_||_|_/__/   |___|_||_/__/\__\__,_|_||_\__|_|\_, |    \___|_| \__,_/__/_||_\___/__/    \___|_|  \___/ 
  *                           |___/                                                      |__/                                                     
  */



// for now lets design our program to 
__global__ void SingleCycle(ReflectiveSlab* Slab3D, NeutronDistribution* Neutrons, unsigned long long* seedNo, int* mergeSignal, int* k_mult) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutrons) { return; }
    GnuAMCM RNG(seedNo[idx]);

    Neutrons->addedNeutronIndex = 1231;
    
    if (idx == 0) { printf("Hello world!\n"); }
            
    // run MC for neutrons
    double DTC = Slab3D->DTC(Neutrons->neutrons[idx], RNG);
    vec3 dirNormal{};
    double DTS = Slab3D->DTS(Neutrons->neutrons[idx], dirNormal);

    printf("%f, %f, %f\n", Neutrons->neutrons[idx].pos.x, Neutrons->neutrons[idx].pos.y, Neutrons->neutrons[idx].pos.z);
    printf("%f\n", DTC); 
    // 아니 DTS DTC 프린트 할려고하면 염병을떠노 하

    //if (idx == 3) { atomicAdd(&(Neutrons->neutronSize), -1); }

    ReactionType RType = Slab3D->getInteractionType(Neutrons->neutrons[idx], RNG);
    if (idx == 3) {
    //if (RType == ReactionType::absorption) {
        Slab3D->absorption(Neutrons->neutrons[idx]);
        atomicAdd(&(Neutrons->neutronSize), -1);
    }
    else if (RType == ReactionType::scatter) {
        Neutrons->neutrons[idx].dirVec = vec3::randomUnit(RNG);
    }
    else {
        Slab3D->fission(Neutrons->neutrons[idx], Neutrons, RNG, k_mult);
    }

    // run MC for addedNeutrons
    if (idx <= Neutrons->addedNeutronSize) {
        double DTC = Slab3D->DTC(Neutrons->addedNeutrons[idx], RNG);
        vec3 dirNormal{};
        double DTS = Slab3D->DTS(Neutrons->addedNeutrons[idx], dirNormal);
        
        ReactionType RType = Slab3D->getInteractionType(Neutrons->addedNeutrons[idx], RNG);
        if (RType == ReactionType::absorption) {
            Slab3D->absorption(Neutrons->addedNeutrons[idx]);
            atomicAdd(&(Neutrons->addedNeutronSize), -1);
        }
        else if (RType == ReactionType::scatter) {
            Neutrons->addedNeutrons[idx].dirVec = vec3::randomUnit(RNG);
        }   
        else {
            Slab3D->fission(Neutrons->addedNeutrons[idx], Neutrons, RNG, k_mult);
        }


    }
    else { return; }
    
    
    if (idx == 2) {
        printf("%d\n", Neutrons->addedNeutronIndex);
    }

    seedNo[idx] = RNG.gen();
    // below can be accessed in host side - copy only the values to Host, decide whether it needs merge or not.
    // If merge is needed, then we will copy the whole struct to host, do some merging, and copy back to device
    // note: we have to check the addeNeutronIndex: not the addedNeutron size, in order for us to determine whether to merge or not.
    //if (Neutrons->addedNeutronIndex > static_cast<int>(0.8 * Neutrons->allocatableNeutrons)) { *mergeSignal = 1; }
}


int main() {
    int initialNumNeutrons = 100000;  
    int excessNumNeutron = initialNumNeutrons * 1.5;
    unsigned long long seedNo = 9223594239;
    int numCycle = 100;
    int threadPerBlock = 32;
    int blockPerDim = (excessNumNeutron + threadPerBlock - 1) / threadPerBlock;

     /*
      *  ___ _  _  ___     ___ ___ _____ _   _ ___ 
      * | _ \ \| |/ __|   / __| __|_   _| | | | _ \
      * |   / .` | (_ |   \__ \ _|  | | | |_| |  _/
      * |_|_\_|\_|\___|   |___/___| |_|  \___/|_|  
      *                                            
      */
    GnuAMCM RNG(seedNo);
    unsigned long long* h_SeedArr = new unsigned long long[excessNumNeutron];
    for (int i = 0; i < excessNumNeutron; i++) {
        /*
        for (int j = 0; j < 3999; j++) {
            RNG.gen_static();
        }
        h_SeedArr[i] = RNG.gen();
        */
        h_SeedArr[i] = (RNG.gen() + 2 * i) & (0xFFFFFFFFFFFFULL);
    }

    unsigned long long* d_SeedArr = nullptr; 
    cudaMalloc(&d_SeedArr, excessNumNeutron * sizeof(unsigned long long));
    cudaMemcpy(d_SeedArr, h_SeedArr, excessNumNeutron * sizeof(unsigned long long), cudaMemcpyHostToDevice);

     /*
      *  _  _          _                    ___      _             
      * | \| |___ _  _| |_ _ _ ___ _ _     / __| ___| |_ _  _ _ __ 
      * | .` / -_) || |  _| '_/ _ \ ' \    \__ \/ -_)  _| || | '_ \
      * |_|\_\___|\_,_|\__|_| \___/_||_|   |___/\___|\__|\_,_| .__/
      *                                                      |_|   
      */
    // build host struct
    NeutronDistribution h_Neutrons(excessNumNeutron, initialNumNeutrons, RNG.gen());
    h_Neutrons.setUniformNeutrons(0.02, 0.02, 0.02);
    
    // build device struct pointer. allocate only the space for it.
    NeutronDistribution* d_Neutrons = nullptr;
    cudaMalloc(&d_Neutrons, sizeof(NeutronDistribution));

    // This is redundant - I'll explain here.
    cudaMemcpy(d_Neutrons, &h_Neutrons, sizeof(NeutronDistribution), cudaMemcpyHostToDevice);
    // From this Memcpy, d_Neutrons only have info of numNeutrons, seedNo, and only the address of the first array element (i.e. &neutrons[0]).
    // Rest of the array (neutrons[1], neutrons[2]) are lost - cannot be deferred from device perspective.
    // Thus, we need to declare device side arrays that contains the data of h_Neutron.neutrons.
    // Other than that, we are going to Memcpy the tmp_Neutrons to d_Neutrons anyways.
    

    // allocate device buffer arrays - this will be used to feed the device 'actual' neutron arrays.
    Neutron* d_bufferNeutrons = nullptr;
    Neutron* d_bufferAddedNeutrons = nullptr;

    // allocate and copy the data from the 'actual' host neutron arrays.
    cudaMalloc(&d_bufferNeutrons, h_Neutrons.allocatableNeutrons * sizeof(Neutron));
    cudaMemcpy(d_bufferNeutrons, h_Neutrons.neutrons, h_Neutrons.allocatableNeutrons * sizeof(Neutron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bufferAddedNeutrons, h_Neutrons.allocatableNeutrons * sizeof(Neutron));
    cudaMemcpy(d_bufferAddedNeutrons, h_Neutrons.addedNeutrons, h_Neutrons.allocatableNeutrons * sizeof(Neutron), cudaMemcpyHostToDevice);
    
    
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
    tmp_Neutrons.neutrons = nullptr;
    tmp_Neutrons.addedNeutrons = nullptr;
    // This is very important, since as this object goes out of the scope, it will call a destructor.
    // When the destructor is called (i.e. delete[] neutrons (a type Neutron*)), it will deallocate the device pointer.
    // This is bad - host objects aren't meant to deallocate device object/pointers. It's done with cudaFree.
    // If you do it, it will caused undefined behavior - usually crash, error on delete_scalar.cpp




    // setup the 2 banks: 



    // Some fucking values to pass around - like multiplication factor (k) or other shits nigga i wanna fuck myself
    int h_mergeSignal = 0;
    int* d_mergeSignal = nullptr;
    cudaMalloc(&d_mergeSignal, sizeof(int));
    cudaMemcpy(d_mergeSignal, &h_mergeSignal, sizeof(int), cudaMemcpyDeviceToHost);
    

    double h_multK = 1.0;
    int* d_multK = nullptr;
    cudaMalloc(&d_multK, sizeof(double));
    cudaMemcpy(d_multK, &h_multK, sizeof(double), cudaMemcpyDeviceToHost);

     /*
      *  ___         _     ___      _             
      * | __|  _ ___| |   / __| ___| |_ _  _ _ __ 
      * | _| || / -_) |   \__ \/ -_)  _| || | '_ \
      * |_| \_,_\___|_|   |___/\___|\__|\_,_| .__/
      *                                     |_|   
      */

    // Currently its a 3d CUBE, 
    ReflectiveSlab h_CubeSlab(0.02, 0.02, 0.02, 0.2, 1.5, 0.3, 2.0);
    ReflectiveSlab* d_CubeSlab = nullptr;
    cudaMalloc(&d_CubeSlab, sizeof(ReflectiveSlab));
    cudaMemcpy(d_CubeSlab, &h_CubeSlab, sizeof(ReflectiveSlab), cudaMemcpyDeviceToHost);


    std::vector<double> DTCArray(excessNumNeutron, 0.0);

    numCycle = 1;
    for (int i = 0; i < numCycle; i++) {

        //
        // SingleCycle << <blockPerDim, threadPerBlock >> > (d_CubeSlab, d_Neutrons, d_SeedArr, d_mergeSignal, d_multK);

        for (int j = 0; j < 10; j++) {


            double DTC = h_CubeSlab.DTC(h_Neutrons.neutrons[j], RNG);
            vec3 dirNormal{};
            double DTS = h_CubeSlab.DTS(h_Neutrons.neutrons[j], dirNormal);

            std::cout << h_Neutrons.neutrons[j].pos.x << " " << h_Neutrons.neutrons[j].pos.y << " " << h_Neutrons.neutrons[j].pos.z << ", dir: ";
            std::cout << h_Neutrons.neutrons[j].dirVec.x << " " << h_Neutrons.neutrons[j].dirVec.y << " " << h_Neutrons.neutrons[j].dirVec.z;
            std::cout << " " << DTC << "\n";

        }
        cudaMemcpy(&h_mergeSignal, d_mergeSignal, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(&h_Neutrons, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost); // will this just copy the plain values ?
        std::cout << "neutron: " << h_Neutrons.neutronSize << " , addedNeutron: " << h_Neutrons.addedNeutronSize << " , addIndex: " << h_Neutrons.addedNeutronIndex << "\n";
        
        if (h_mergeSignal == 1) {
            NeutronDistribution h_NeutronsReceiver = h_Neutrons;
            cudaMemcpy(&h_NeutronsReceiver, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost);
            h_Neutrons.neutronSize = h_NeutronsReceiver.neutronSize;
            h_Neutrons.addedNeutronSize = h_NeutronsReceiver.addedNeutronSize;

            cudaMemcpy(h_Neutrons.neutrons, d_bufferNeutrons, excessNumNeutron * sizeof(Neutron), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Neutrons.addedNeutrons, d_bufferAddedNeutrons, excessNumNeutron * sizeof(Neutron), cudaMemcpyDeviceToHost);
            std::cout << h_Neutrons.addedNeutronSize << "\n";
        }
    }

    

    cudaFree(d_SeedArr);
    cudaFree(d_Neutrons);
    

}