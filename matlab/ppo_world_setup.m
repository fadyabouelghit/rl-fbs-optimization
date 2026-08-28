function info = ppo_world_setup(worldJson, worldId)
%PPO_WORLD_SETUP Build and persist the PPO world MATLAB-side.
%
% Called by the Python PPO pipeline (ppo/matlab_bridge.py). Keeps the heavy
% objects (antenna templates, precomputed MBS map cache) inside MATLAB
% appdata so per-step SINR calls (ppo_sinr_eval) only marshal scalars across
% the Python <-> MATLAB boundary. The legacy Python path round-tripped the
% full map cache on every step.
%
% worldJson (JSON string) fields:
%   width, height       : world extent [m]
%   num_mbs             : requested MBS count (used by the hex generator)
%   mbs_locations       : num_mbs x 2 [x y] array, or [] for hex placement
%   isd, margin         : hex generator parameters
%   mbs_height, mbs_power, ue_height
%   scenario, map_mode  : QuaDRiGa scenario / power_map mode
%   cache_dir           : on-disk cache folder for precompute_mbs_power_maps
%   band_mode           : 'legacy' (single band, original train_ppo.py world)
%                         or 'multi' (coverage+capacity, GA-parity world)
%
% The world build mirrors optimize_base_station_ga.m exactly: pack_mbs_params
% -> x<->y row swap -> precompute_mbs_power_maps, so cached maps are shared
% with the GA harness for identical geometries.
%
% Returns a lightweight info struct (true MBS coordinates, band count, ...).

    if nargin < 2 || isempty(worldId); worldId = 'default'; end
    cfg = jsondecode(worldJson);

    bandMode = char(cfg.band_mode);
    isMulti = strcmp(bandMode, 'multi');

    if isMulti
        freqs = band_frequencies();
        fbsAntenna = [setup_antenna(freqs.coverage), setup_antenna(freqs.capacity)];
        mbsAntenna = [setup_antenna(freqs.coverage), setup_antenna(freqs.capacity)];
    else
        % Legacy single-band: identical antenna setup to the original
        % train_ppo.py (setup_antenna default carrier).
        fbsAntenna = setup_antenna();
        mbsAntenna = setup_antenna();
    end

    % MBS sites: explicit coordinates override the hex generator.
    if isfield(cfg, 'mbs_locations') && ~isempty(cfg.mbs_locations)
        locs = cfg.mbs_locations;
        if isvector(locs); locs = reshape(locs, 1, []); end
        xs = locs(:, 1).';
        ys = locs(:, 2).';
    else
        [xs, ys] = generate_hex_sites(cfg.width, cfg.height, cfg.isd, ...
            cfg.margin, cfg.num_mbs);
    end

    [mbs_params, antennaObjectMbs, containsMbs, numBS] = ...
        pack_mbs_params(xs, ys, cfg.mbs_height, cfg.mbs_power, mbsAntenna);

    % Legacy x<->y row swap -- must mirror optimize_base_station_ga.m so the
    % cache keys and downstream consumers stay in the same (swapped) frame.
    tempForX = mbs_params(1, :);
    mbs_params(1, :) = mbs_params(2, :);
    mbs_params(2, :) = tempForX;

    subset = struct('xmin', 0, 'xmax', cfg.width, 'ymin', 0, 'ymax', cfg.height);
    mbsCache = precompute_mbs_power_maps(mbs_params, antennaObjectMbs, subset, ...
        char(cfg.scenario), char(cfg.map_mode), cfg.ue_height, char(cfg.cache_dir));

    world = struct();
    world.bandMode = bandMode;
    world.isMulti = isMulti;
    world.fbsAntenna = fbsAntenna;
    world.antennaObjectMbs = antennaObjectMbs;
    world.mbsCache = mbsCache;
    world.containsMbs = logical(containsMbs);
    world.numMbs = numBS;
    world.W = cfg.width;
    world.H = cfg.height;
    % Swapped-frame rows; un-swapped at the SINREvaluation call site exactly
    % like evaluatePopulation (arg mbs_x <- row 2 = true x).
    world.mbs_row1 = mbs_params(1, :);   % true y
    world.mbs_row2 = mbs_params(2, :);   % true x
    world.mbs_height = mbs_params(3, :);
    world.mbs_power = mbs_params(4, :);

    setappdata(0, ['PPO_WORLD_' worldId], world);

    info = struct( ...
        'world_id', worldId, ...
        'mbs_x', xs(:).', ...
        'mbs_y', ys(:).', ...
        'num_mbs', numBS, ...
        'contains_mbs', double(containsMbs), ...
        'n_bands', numel(fbsAntenna), ...
        'band_mode', bandMode);
end
