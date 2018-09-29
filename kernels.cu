#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.h"

#define EPSILON 0.0001

__global__ void update_iaf_psc_delta_data(struct nest::iaf_psc_delta_data_ data, 
    struct nest::spike_event_data_ spike_event_data_dev,
    int size, //number of nodes in the system
    int from, int to, int time_resolution_ms, int *r_arr_test){

  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int tid = blockId * blockDim.x + threadIdx.x;
  //kernel related initializations
  int my_spike_index = 0;


  //NEST code
  if(tid < size - 3) { //-3 will need to be updated
  
    for(int lag = from ; lag < to ; lag++){
      //for(int lag = 0 ; lag < 1 ; lag++){
      if ( data.r_arr[tid] == 0 ){
        // neuron not refractory
        data.y3_arr[tid] = data.P30_arr[tid]*(data.y0_arr[tid] + data.I_e_arr[tid]) +
          data.P33_arr[tid] * data.y3_arr[tid] + 
          data.spikes_arr[tid + lag * data.size];

        // if we have accumulated spikes from refractory period, 
        // add and reset accumulator
        if ( data.with_refr_input_arr[tid] && fabs(data.refr_spikes_buffer_arr[tid]) < EPSILON){
          data.y3_arr[tid] += data.refr_spikes_buffer_arr[tid];
          data.refr_spikes_buffer_arr[tid] = 0.0;
        }

        // lower bound of membrane potential
        data.y3_arr[tid] = ( data.V_min_arr[tid] - data.y3_arr[tid] > -1 * EPSILON ? 
            data.V_min_arr[tid] : data.y3_arr[tid]); 	 
      }
        else { // neuron is absolute refractory
          // read spikes from buffer and accumulate them, discounting
          // for decay until end of refractory period
          if ( data.with_refr_input_arr[tid] )
            data.refr_spikes_buffer_arr[tid] += 
              data.spikes_arr[tid + lag * data.size] * 
              std::exp(-data.r_arr[tid] * 
                  time_resolution_ms / data.tau_m_arr[tid]);

          //ENGIN next else block supposed to clear the entry in the buffer by
          //doing a read however, since we do not have the buffer structure in
          //GPU, I will set the member of the array directly to zero. As there
          //is another read in the above if statement there will be no else in
          //the new version
          /*else
            data.spikes_arr[tid + lag * data.size];  // clear buffer entry, ignore spike*/

          data.spikes_arr[tid + lag * data.size] = 0.0;

          --data.r_arr[tid];
        }

        // threshold crossing
        if((data.y3_arr[tid] - data.V_th_arr[tid]) > -1 * EPSILON){ 
          data.r_arr[tid] = data.RefractoryCounts_arr[tid];
          data.y3_arr[tid] = data.V_reset_arr[tid];

          // EX: must compute spike time
          //set_spiketime(Time::step(origin.get_steps()+lag+1));

          //SpikeEvent se;
          //network()->send(*this, se, lag);

          //GPU CODE
          spike_event_data_dev.lag_arr[tid + size * my_spike_index] = lag;
          my_spike_index++;

        }

        // set new input current
        data.y0_arr[tid] = data.currents_arr[tid + lag * data.size];

        // voltage logging
        //data.logger_.record_data(origin.get_steps()+lag);

    }
  }
}

namespace nest {

  CUDAController::CUDAController(){
    log_file.open("/mnt/lustre_server/users/engin/work/NEST/nest_test/kernels_log",
        std::ofstream::out | std::ofstream::app);
  }

  void CUDAController::get_outbox(int **senders, int **lags, int size){
    cudaMemcpy((void *) *lags, (const void *)spike_event_data_dev.lag_arr, 
        sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaMemset((void *) spike_event_data_dev.lag_arr, -1, sizeof(int) * size);

  }

  void CUDAController::check_outbox(int size){

    printf("OUTBOX SIZE : %d\n", size);
    int *tmp = (int *)malloc(sizeof(int) * size);

    cudaMemcpy((void *)tmp, (const void *)spike_event_data_dev.sender_idx_arr, sizeof(int) * size, 
        cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < size ; i++){
      printf("Sender idx : %d\n", tmp[i]);
    }
  }

  std::string CUDAController::iaf_psc_delta_kw(int from, int to, int time_resolution_ms, int network_size){
    cudaError_t err;

    log_file << "iaf_psc_delta kernel wrapper called" << std::endl;


    int threads_per_block = 256;
    int num_blocks = (network_size + threads_per_block - 1) / threads_per_block;

    dim3 dim_grid(num_blocks, 1);

    log_file << "Num Blocks " << num_blocks << std::endl;	
    log_file.flush();
    update_iaf_psc_delta_data<<<dim_grid, threads_per_block>>>(
        iaf_psc_delta_data_dev,
        spike_event_data_dev,
        network_size, //iaf_psc_delta_data_dev.size,
        from, to, time_resolution_ms, iaf_psc_delta_data_dev.r_arr);


    err = cudaGetLastError();
    if(err != cudaSuccess){
      log_file << "Error 2 : " << cudaGetErrorString(err) << std::endl;
      return cudaGetErrorString(err);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess){
      log_file << "Error 3 : " << cudaGetErrorString(err) << std::endl;
      return cudaGetErrorString(err);
    }

    log_file.close();
    return "NO ERR";
  }


  void CUDAController::allocate_and_offload(void **dest, void *source, size_t size){

    cudaError_t error;

    cudaMalloc((void **) dest, size);
    if((error = cudaGetLastError()) != cudaSuccess){
      log_file << "Error in cudaMalloc" << std::endl;
    }
    else {
      ;//log_file << "cudaMalloc successful for " << size << " bytes" <<  std::endl;
    }

    cudaMemcpy(*dest, source, size, cudaMemcpyHostToDevice);
    if((error = cudaGetLastError()) != cudaSuccess){
      log_file << "Error in cudaMemcpy" << std::endl;
    }
    else {
      ;//log_file << "cudaMemcpy successful for " << size << " bytes" <<  std::endl;
    }

  }

  void CUDAController::offload_spikes(double *spikes, int size){

    cudaMemcpy((void *) iaf_psc_delta_data_dev.spikes_arr, (const void *) spikes, 
        sizeof(double) * size, cudaMemcpyHostToDevice);

  }	

  void CUDAController::offload_data_wrapper(struct nest::iaf_psc_delta_data_ iaf_psc_delta_data, 
      double *rate_arr, /*spike det.?*/
      struct nest::spike_event_data_ spike_event_data, int iaf_psc_delta_size, 
      int rate_arr_size,
      int spike_event_size, int spikes_arr_size, int currents_arr_size){

    //this struct will make passing lots of pointers to device kernel easier
    //struct nest::iaf_psc_delta_data_ iaf_psc_delta_data_dev;


    //TODO write CUDA error handling macro
    iaf_psc_delta_data_dev.size = iaf_psc_delta_size;
    printf("IAF PSC Delta size : %d\n", iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.tau_m_arr, iaf_psc_delta_data.tau_m_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.c_m_arr, iaf_psc_delta_data.c_m_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.t_ref_arr, iaf_psc_delta_data.t_ref_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.E_L_arr, iaf_psc_delta_data.E_L_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.I_e_arr, iaf_psc_delta_data.I_e_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.V_th_arr, iaf_psc_delta_data.V_th_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.V_min_arr, iaf_psc_delta_data.V_min_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.V_reset_arr, iaf_psc_delta_data.V_reset_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.with_refr_input_arr, 
        iaf_psc_delta_data.with_refr_input_arr,
        sizeof(bool) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.y0_arr, iaf_psc_delta_data.y0_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.y3_arr, iaf_psc_delta_data.y3_arr,
        sizeof(double) * iaf_psc_delta_size);

    int i;

    allocate_and_offload((void **)&iaf_psc_delta_data_dev.r_arr, iaf_psc_delta_data.r_arr,
        sizeof(int) * iaf_psc_delta_size);
    /*		cudaMalloc((void **) &iaf_psc_delta_data_dev.r_arr, sizeof(int) * iaf_psc_delta_size);
                    cudaMemcpy((void *) iaf_psc_delta_data_dev.r_arr, 
                    (const void *) iaf_psc_delta_data.r_arr,
                    sizeof(int) * iaf_psc_delta_size,
                    cudaMemcpyHostToDevice);
     */

    allocate_and_offload((void **)&iaf_psc_delta_data_dev.refr_spikes_buffer_arr, 
        iaf_psc_delta_data.refr_spikes_buffer_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.spikes_arr, iaf_psc_delta_data.spikes_arr,
        sizeof(double) * spikes_arr_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.currents_arr, iaf_psc_delta_data.currents_arr,
        sizeof(double) * currents_arr_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.P30_arr, iaf_psc_delta_data.P30_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.P33_arr, iaf_psc_delta_data.P33_arr,
        sizeof(double) * iaf_psc_delta_size);
    allocate_and_offload((void **)&iaf_psc_delta_data_dev.RefractoryCounts_arr, 
        iaf_psc_delta_data.RefractoryCounts_arr,
        sizeof(int) * iaf_psc_delta_size);
    //iaf_psc_delta offload ends here

    //spike event offload starts here


    cudaMalloc((void **) &spike_event_data_dev.lag_arr, sizeof(int) * spikes_arr_size);
    cudaMemset((void *) spike_event_data_dev.lag_arr, -1, sizeof(int) * spikes_arr_size);
    //log_file.close();
  }

  std::string kernel_wrapper_2(int *array, int size){
    cudaError_t err;
    int *device_array;

    std::ofstream log_file;
    log_file.open("/mnt/lustre_server/users/engin/work/NEST/nest_test/kernel_log", 
        std::ofstream::out | std::ofstream::app);


    cudaMalloc((void **)&device_array, size * sizeof(int));
    err = cudaGetLastError();

    if(err != cudaSuccess){
      //printf("Error 1 : %s\n", cudaGetErrorString(err));
      log_file << "Error 1 : " << cudaGetErrorString(err) << std::endl;
      return cudaGetErrorString(err);
    }

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    dim3 dim_grid(num_blocks, 1);
    printf("Calling the kernel\n");


    err = cudaGetLastError();
    if(err != cudaSuccess){
      //printf("Error 2 : %s\n", cudaGetErrorString(err));
      log_file << "Error 2 : " << cudaGetErrorString(err) << std::endl;
      return cudaGetErrorString(err);
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess){
      //printf("Error 2 : %s\n", cudaGetErrorString(err));
      log_file << "Error 3 : " << cudaGetErrorString(err) << std::endl;
      return cudaGetErrorString(err);
    }

    //log_file << "After synch, starting copy\n";
    cudaMemcpy((void *)array, (void *)device_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    log_file.close();
    return "NO ERR";
  }
}
