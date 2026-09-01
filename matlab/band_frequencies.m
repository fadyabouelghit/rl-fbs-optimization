function freqs = band_frequencies()
%BAND_FREQUENCIES Central source of antenna carrier frequencies.
%
% Every transmitter class (FBS, MBS, femto) shares the same band-id space:
%   band id 0 -> coverage band (antenna row 1)
%   band id 1 -> capacity band (antenna row 2)
%
% SINREvaluation accumulates same-band interference by band id, so every BS
% at a given band id must transmit on the same physical carrier or the
% interference accounting is incoherent. Edit this function in one place
% and every entry point picks up the change automatically:
%
%   - ppo_world_setup.m           (FBS + MBS antenna templates)
%
% Output struct fields:
%   coverage : centre frequency [Hz] for band-id 0 antennas (row 1)
%   capacity : centre frequency [Hz] for band-id 1 antennas (row 2)
%
% To configure a real dual-band split (e.g. 2 GHz / 2.6 GHz), set
% capacity = 2.6e9 below and rerun -- nothing else needs to change.

    freqs = struct( ...
        'coverage', 2e9, ...
        'capacity', 2.6e9);
end
