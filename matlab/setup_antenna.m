function l = setup_antenna(centerFreq)
    if nargin < 1 || isempty(centerFreq)
        centerFreq = 2e9;
    end
    s = qd_simulation_parameters;
    s.center_frequency = centerFreq;  
    l = qd_layout(s);
    l.no_tx = 1;
    antenna = qd_arrayant('3gpp-3d', 1, 1, s.center_frequency);
    l.tx_array(1,1) = antenna;
    l.tx_array(1,1).rotate_pattern(-90, 'y', 1, 1);
    l.rx_array = qd_arrayant('omni');
end
