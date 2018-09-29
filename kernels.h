#include <fstream>

#include "gpu_commons.h"

namespace nest {


  class CUDAController{
    public:
      //variables
      CUDAController();
      //member functions
      void offload_data_wrapper(struct nest::iaf_psc_delta_data_ iaf_psc_delta_data, 
          double *rate_arr, /*spike det.?*/
          struct nest::spike_event_data_ spike_event_data, int iaf_psc_delta_size,
          int rate_arr_size, int spike_event_size, int spikes_arr_size, 
          int currents_arr_size);

      std::string iaf_psc_delta_kw(int from, int to, int time_resolution_ms, int network_size);
      void check_outbox(int size);
      void get_outbox(int **, int **, int);
      void offload_spikes(double *, int);
    private:
      //variables
      struct iaf_psc_delta_data_ iaf_psc_delta_data_dev;
      struct spike_event_data_ spike_event_data_dev;
      std::ofstream log_file;
      //member functions
      void allocate_and_offload(void **dest, void *source, size_t size);

  };

}
