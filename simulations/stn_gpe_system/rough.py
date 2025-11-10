axs[0, 0].plot(t_chunk_PD, V_stn_time_all_PD[t_low_PD:t_high_PD, 15, 15], color=colors['normal'])
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("V (mV)")


spike_array_stn_PD = np.array(spike_monitor_stn_PD[t_low_PD:t_high_PD])
num_neurons_stn_PD = spike_array_stn_PD.shape[1]
time_steps_stn_PD= spike_array_stn_PD.shape[0]
t_raster_PD = np.linspace(0, time_steps_stn_PD * h_PD / 1000, time_steps_stn_PD)
for n in range(num_neurons_stn_PD):
    axs[0, 1].scatter(t_raster_PD, (n + 1) * spike_array_stn_PD[:, n], color=colors['normal'], s=0.5)
axs[0, 1].set_ylim(0.5, num_neurons_stn_PD + 0.5)
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Neuron")


im1 = axs[0, 2].pcolormesh(t_spec_stn_PD, f_stn_PD, 10 * np.log10(Sxx_stn_PD), 
                          cmap=cmap_PD, shading='gouraud')
axs[0, 2].set_ylabel('Frequency (Hz)')
axs[0, 2].set_xlabel('Time (s)')
axs[0, 2].set_ylim(0, 40)
