namespace nest{

  struct edge_data_{
    int *source_idx_arr;
    int *target_idx_arr;

    int *weight_arr;
    int *delay_arr;
  };

  struct spike_event_data_ {
    //these class parameters are directly inherited from event class
    //spikeevent class does not have any parameters specific to itself
    int *sender_idx_arr;
    int *receiver_idx_arr;

    //these port numbers are probbaly unnecessary for now
    //but may be necessary for event recording purposes
    int *sender_port_arr; //change this to int for perfroamcen
    int *receiver_port_arr;

    int *d_arr; //delays

    double *offset_arr;
    double *weight_arr;

    //GPU specific attribute that will keep the lag component of a
    //spike this is probably the only thing necessary along with the
    //GID of the node sending the spike
    int *lag_arr;
    long *steps_arr;
  };

  struct iaf_psc_delta_data_ {

    //Parameter Arrays
    double *tau_m_arr;
    double *c_m_arr;
    double *t_ref_arr;
    double *E_L_arr;
    double *I_e_arr;
    double *V_th_arr;
    double *V_min_arr;
    double *V_reset_arr;
    bool *with_refr_input_arr;

    //State Arrays
    double *y0_arr;
    double *y3_arr;
    int *r_arr;
    double *refr_spikes_buffer_arr;

    //Buffer structures

    //NOTE: I used 1D allocation as that might be asier for GPU
    //we can take a look at its implications later on
    double *spikes_arr;
    double *currents_arr;

    //Variable arrays
    double *P30_arr;
    double *P33_arr;
    int *RefractoryCounts_arr; //I dont like capital R


    //additional variables for controlling iaf_psc_delta data
    int size;

    int spikes_size; //for our case thise can represent both
    //incoming and outgoing
    int currents_size;

  };

}
