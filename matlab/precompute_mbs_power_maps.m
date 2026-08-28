function cache = precompute_mbs_power_maps(mbs_params, antennaObjectMbs, subset, scenario, mode, ue_height, cacheDir, returnCell)
% Precompute (or load) per-MBS, per-band power maps and store them in a cache struct array.
% Inputs:
%   mbs_params        : 4×N [x; y; z; power]
%   antennaObjectMbs  : nBands×N array of antenna templates (one per band per MBS)
%   subset            : struct with fields xmin, xmax, ymin, ymax
%   scenario          : e.g., '3GPP_38.901_UMa_LOS'
%   mode              : e.g., 'quick'
%   ue_height         : e.g., 1.5
%   cacheDir          : folder to store .mat cache files (optional but recommended)
%
% Output:
%   cache(b, i) : struct with fields
%       .key, .x, .y, .z, .power, .scenario, .subset (struct),
%       .ue_height, .mode, .band, .center_freq,
%       .x_coords, .y_coords, .map (single)
%
% Notes:
%   - Ensures all sites/bands share identical x_coords/y_coords; warns if not.
%   - Uses on-disk caching keyed by MD5 of the config (incl. band) to avoid recomputing.

    arguments
        mbs_params (4,:) double
        antennaObjectMbs (:,:)    % nBands × N
        subset struct
        scenario char
        mode char
        ue_height double = 1.5
        cacheDir char = ''
        returnCell logical = false
    end

    N = size(mbs_params, 2);
    nBands = size(antennaObjectMbs, 1);
    assert(size(antennaObjectMbs, 2) == N, ...
        'antennaObjectMbs must be nBands × N (got %d × %d, N=%d).', ...
        size(antennaObjectMbs,1), size(antennaObjectMbs,2), N);

    if ~isempty(cacheDir) && ~exist(cacheDir,'dir')
        mkdir(cacheDir);
    end
    emptyEntry = struct('key',[], 'x',[], 'y',[], 'z',[], 'power',[], ...
                        'scenario',[], 'subset',[], 'ue_height',[], 'mode',[], ...
                        'band',[], 'center_freq',[], ...
                        'x_coords',[], 'y_coords',[], 'map',[]);
    cache = repmat(emptyEntry, nBands, N);

    commonGrid = [];  % to verify identical coordinate grids across sites

    for i = 1:N
        x  = mbs_params(1,i);
        y  = mbs_params(2,i);
        z  = mbs_params(3,i);
        pw = mbs_params(4,i);

        for b = 1:nBands
            ant = antennaObjectMbs(b, i);
            center_freq = NaN;
            try
                if isprop(ant, 'center_frequency') || isfield(ant, 'center_frequency')
                    center_freq = ant.center_frequency;
                end
            catch
                center_freq = NaN;
            end

            key = make_mbs_key(x, y, z, pw, scenario, mode, ue_height, subset, b, center_freq);

            map_file = '';
            if ~isempty(cacheDir)
                map_file = fullfile(cacheDir, [key '.mat']);
            end

            if ~isempty(map_file) && exist(map_file,'file')
                S = load(map_file, 'map', 'x_coords', 'y_coords', 'meta');
                map      = S.map;
                x_coords = S.x_coords;
                y_coords = S.y_coords;
            else
                ant.tx_position(:,1) = [x; y; z];

                [cellMaps, x_coords, y_coords] = ant.power_map( ...
                    scenario, mode, 1, ...
                    subset.xmin, subset.xmax, subset.ymin, subset.ymax, ...
                    ue_height, pw);

                map = sum(cat(3, cellMaps{:}), 3);
                map = single(map);

                if ~isempty(map_file)
                    meta = struct('x',x,'y',y,'z',z,'power',pw, ...
                                  'scenario',scenario,'mode',mode, ...
                                  'ue_height',ue_height,'subset',subset, ...
                                  'band',b,'center_freq',center_freq);
                    try
                        save(map_file, 'map', 'x_coords', 'y_coords', 'meta', '-v7.3');
                    catch
                        save(map_file, 'map', 'x_coords', 'y_coords', 'meta');
                    end
                end
            end

            cache(b, i).key         = key;
            cache(b, i).x           = x;
            cache(b, i).y           = y;
            cache(b, i).z           = z;
            cache(b, i).power       = pw;
            cache(b, i).scenario    = scenario;
            cache(b, i).subset      = subset;
            cache(b, i).ue_height   = ue_height;
            cache(b, i).mode        = mode;
            cache(b, i).band        = b;
            cache(b, i).center_freq = center_freq;
            cache(b, i).x_coords    = x_coords;
            cache(b, i).y_coords    = y_coords;
            cache(b, i).map         = map;

            thisGrid = [numel(x_coords), numel(y_coords), x_coords(1), x_coords(end), y_coords(1), y_coords(end)];
            if isempty(commonGrid)
                commonGrid = thisGrid;
            else
                if any(abs(thisGrid - commonGrid) > 1e-9)
                    warning('MBS #%d band %d grid differs from others. Power maps may not align exactly.', i, b);
                end
            end
        end
    end

    if returnCell
        cache = arrayfun(@(s) s, cache, 'UniformOutput', false);
    end
end

function [map, x_coords, y_coords] = lookup_mbs_power_map(cache, b, i)
% Retrieve cached map for MBS index i at band b
    map      = cache(b, i).map;
    x_coords = cache(b, i).x_coords;
    y_coords = cache(b, i).y_coords;
end

function key = make_mbs_key(x,y,z,pw,scenario,mode,ue_height,subset,band,center_freq)
% Deterministic key from full configuration (robust to float formatting)
    payload = struct('x',x,'y',y,'z',z,'pw',pw, ...
                     'scenario',string(scenario), 'mode',string(mode), ...
                     'ue',ue_height, ...
                     'xmin',subset.xmin,'xmax',subset.xmax,'ymin',subset.ymin,'ymax',subset.ymax, ...
                     'band',band, 'fc',center_freq);
    s = jsonencode(payload);
    key = md5_of_string(s);
end

function h = md5_of_string(s)
% MD5 as hex using Java (available in MATLAB)
    md = java.security.MessageDigest.getInstance('MD5');
    md.update(uint8(s));
    d = typecast(md.digest, 'uint8');
    h = lower(reshape(dec2hex(d)',1,[]));
end
