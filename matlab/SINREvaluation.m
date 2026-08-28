function [user_positions, df_users_table, total_connected_users, total_transmitted_pwr, avg_rate_connected_bpsHz, fbs_connected_users, mbs_connected_users, sum_rate_connected_bpsHz, mbs_coverage_connected, mbs_capacity_connected] = SINREvaluation(antenna_object, power_status, tx_x, tx_y, tx_height, no_fbs, tx_power, mbs_x, mbs_y, mbs_height, mbs_power, subset_x_min, subset_x_max, subset_y_min, subset_y_max, num_users, threshold, containsMbs, antennaObjectMbs, mbsCache, bsBandIds, mbsSlotMap)
% SINREVALUATION Computes the SINR values for users based on FBS and optional MBS parameters.
%
%   INPUTS:
%     antenna_object     : FBS antenna object(s), scalar or 1×no_fbs array
%     power_status       : Activation vector for each FBS (1 = active, 0 = inactive)
%     tx_x, tx_y         : X and Y coordinates of each FBS
%     tx_height          : Heights of each FBS
%     no_fbs             : Number of Flying Base Stations (FBSs)
%     tx_power           : Transmission power for each FBS
%     mbs_x, mbs_y       : Coordinates of MBSs (optional, retained for API
%                          continuity; map sampling uses mbsCache).
%     mbs_height         : Heights of MBSs (optional)
%     mbs_power          : Transmission power for each MBS
%     subset_x_min/max,
%     subset_y_min/max   : Bounds for the region of interest
%     num_users          : Number of users
%     threshold          : SINR threshold in dB
%     containsMbs        : Binary flag indicating MBS presence
%     antennaObjectMbs   : Array of MBS antenna objects
%     mbsCache           : Power map cache (nBands × num_mbs struct/cell array)
%     bsBandIds          : 1×(no_fbs + numSlots) band ids (0/1). Interference
%                          is only accumulated from same-band BSs. numSlots is
%                          the number of MBS-band slot rows (see mbsSlotMap).
%     mbsSlotMap         : Optional numSlots×3 matrix describing the MBS-band
%                          slot expansion: each row is [mbsIdx, bandIdx, active].
%                            mbsIdx  : column index into mbsCache (1..num_mbs)
%                            bandIdx : row index into mbsCache (1-based band id)
%                            active  : 1 to sample the cached map, 0 to emit
%                                      a zero column (slot disabled but slot
%                                      exists for stable column layout).
%                          When omitted, falls back to one slot per MBS using
%                          the legacy bsBandIds(no_fbs+1:end) band ids.
%
%   OUTPUTS:
%     user_positions         : Matrix of user coordinates
%     df_users_table         : Table with power values and SINRs per BS
%     total_connected_users  : Total number of connected users (SINR ≥ threshold)
%     total_transmitted_pwr  : Sum of transmission power of all active FBSs
%     ...
%     mbs_coverage_connected : Connected users served by an MBS coverage-band
%                              slot (mbsSlotMap bandIdx 1)
%     mbs_capacity_connected : Connected users served by an MBS capacity-band
%                              slot (mbsSlotMap bandIdx 2)
%                              Invariant: fbs + coverage + capacity == total.

    if iscell(mbsCache)
        if isempty(mbsCache)
            mbsCache = struct([]);
        else
            mbsCache = reshape([mbsCache{:}], size(mbsCache));
        end
    end

    num_fbs_scalar = round(double(no_fbs));
    num_mbs_scalar = 0;
    if containsMbs
        num_mbs_scalar = size(antennaObjectMbs, 2);
    end

    if nargin < 22 || isempty(mbsSlotMap)
        % Legacy fallback: one slot per MBS, band id taken from the trailing
        % portion of bsBandIds. Caller is responsible for keeping bsBandIds
        % length consistent (no_fbs + num_mbs_scalar).
        if num_mbs_scalar > 0
            legacyBands = ones(1, num_mbs_scalar);
            if nargin >= 21 && ~isempty(bsBandIds) && numel(bsBandIds) >= num_fbs_scalar + num_mbs_scalar
                legacyBands = double(bsBandIds(num_fbs_scalar + (1:num_mbs_scalar)));
            end
            mbsSlotMap = [ (1:num_mbs_scalar).', legacyBands(:) + 1, ones(num_mbs_scalar, 1) ];
        else
            mbsSlotMap = zeros(0, 3);
        end
    else
        assert(size(mbsSlotMap, 2) == 3, ...
            'mbsSlotMap must have 3 columns [mbsIdx, bandIdx, active].');
    end

    numSlots = size(mbsSlotMap, 1);
    total_bs = num_fbs_scalar + numSlots;
    if nargin < 21 || isempty(bsBandIds)
        % Derive bsBandIds from the slot map (cache row index -> band id is
        % bandIdx - 1). FBS portion defaults to coverage (0).
        mbsBandIdsFromSlots = mbsSlotMap(:, 2).' - 1;
        bsBandIds = [zeros(1, num_fbs_scalar), mbsBandIdsFromSlots];
    else
        bsBandIds = reshape(double(bsBandIds), 1, []);
    end
    assert(numel(bsBandIds) == total_bs, ...
        'bsBandIds must have %d entries (no_fbs + numSlots).', total_bs);

    user_positions = generate_user_positions(subset_x_min, subset_x_max, subset_y_min, subset_y_max, num_users);

    [~, df_users_table, avg_rate_connected_bpsHz, sum_rate_connected_bpsHz] = calculate_power_iterator(antenna_object, no_fbs, power_status, tx_y, tx_x, tx_height, tx_power, ...
        mbs_x, mbs_y, mbs_height, mbs_power, user_positions, ...
        subset_x_min, subset_x_max, subset_y_min, subset_y_max, threshold, containsMbs, antennaObjectMbs, mbsCache, bsBandIds, mbsSlotMap);

    total_connected_users = sum(df_users_table.is_connected);
    fbs_connected_users = sum(df_users_table.is_connected & df_users_table.FBS_connection_index >= 1 & df_users_table.FBS_connection_index <= no_fbs);
    mbs_connected_users = total_connected_users - fbs_connected_users;

    % Per-tier MBS split: map each MBS-connected user's serving slot column
    % through mbsSlotMap (bandIdx 1 = coverage, 2 = capacity).
    mbsConnMask = df_users_table.is_connected & df_users_table.FBS_connection_index > num_fbs_scalar;
    connSlotIdx = df_users_table.FBS_connection_index(mbsConnMask) - num_fbs_scalar;
    if isempty(connSlotIdx)
        mbs_coverage_connected = 0;
        mbs_capacity_connected = 0;
    else
        slotBandIdx = mbsSlotMap(connSlotIdx, 2);
        mbs_coverage_connected = sum(slotBandIdx == 1);
        mbs_capacity_connected = sum(slotBandIdx == 2);
    end
    if fbs_connected_users + mbs_coverage_connected + mbs_capacity_connected ~= total_connected_users
        warning('SINREvaluation:tierSplitMismatch', ...
            'Tier split fbs=%d + coverage=%d + capacity=%d ~= total connected %d.', ...
            fbs_connected_users, mbs_coverage_connected, mbs_capacity_connected, total_connected_users);
    end

    % total_transmitted_pwr = sum(tx_power .* power_status);
    total_transmitted_pwr = sum(tx_power);
end

function user_positions = generate_user_positions(x_min, x_max, y_min, y_max, num_users)
% GENERATE_USER_POSITIONS Generates uniformly distributed user positions.
%
%   INPUTS:
%     x_min, x_max     : Range of x-coordinates
%     y_min, y_max     : Range of y-coordinates
%     num_users        : Number of users
%
%   OUTPUT:
%     user_positions   : num_users × 2 matrix of (x, y) user positions

    % Adjust zero bounds for indexing
    x_min = max(1, x_min);
    y_min = max(1, y_min);

    % User map is ALWAYS the same (seed 0). Two ways to draw it:
    %   legacy   : reset the GLOBAL stream (historical behavior; reproduces
    %              old logged runs bit-for-bit, but leaks determinism into
    %              the GA operators). Selected by optimizeBaseStation when
    %              params.randomizeGA = false.
    %   isolated : dedicated local stream; global RNG untouched, so GA
    %              operators stay truly random (params.randomizeGA = true,
    %              or SINREvaluation called outside the GA loop).
    if isappdata(0, 'GA_LEGACY_USER_RNG') && getappdata(0, 'GA_LEGACY_USER_RNG')
        rng(0);
        user_positions = [randi([x_min, x_max], num_users, 1), ...
                          randi([y_min, y_max], num_users, 1)];
    else
        s = RandStream('mt19937ar', 'Seed', 0);
        user_positions = [randi(s, [x_min, x_max], num_users, 1), ...
                          randi(s, [y_min, y_max], num_users, 1)];
    end
end

function user_positions = generate_user_positions_clustered(x_min, x_max, y_min, y_max, num_users, separation, std1, std2)
% GENERATE_USER_POSITIONS_CLUSTERED Creates two user clusters within defined bounds.
%
%   INPUTS:
%     x_min, x_max, y_min, y_max : Bounds of area
%     num_users                  : Number of users
%     separation, std1, std2     : Cluster separation and standard deviations
%
%   OUTPUT:
%     user_positions             : Clustered user positions matrix

    adjusted_x_min = x_min + 1;
    adjusted_x_max = x_max - 1;
    adjusted_y_min = y_min + 1;
    adjusted_y_max = y_max - 1;

    mean1 = [adjusted_x_min + separation/2, adjusted_y_min + separation/2];
    mean2 = [adjusted_x_max - separation/2, adjusted_y_max - separation/2];

    % Fixed user map; legacy vs isolated stream — see generate_user_positions.
    if isappdata(0, 'GA_LEGACY_USER_RNG') && getappdata(0, 'GA_LEGACY_USER_RNG')
        rng(0);
        cluster1 = mean1 + std1 * randn(num_users/2, 2);
        cluster2 = mean2 + std2 * randn(num_users/2, 2);
    else
        s = RandStream('mt19937ar', 'Seed', 0);
        cluster1 = mean1 + std1 * randn(s, num_users/2, 2);
        cluster2 = mean2 + std2 * randn(s, num_users/2, 2);
    end

    user_positions = round([cluster1; cluster2]);
    user_positions(:, 1) = max(min(user_positions(:, 1), adjusted_x_max), adjusted_x_min);
    user_positions(:, 2) = max(min(user_positions(:, 2), adjusted_y_max), adjusted_y_min);
end

function [power_map, x_coords, y_coords] = calculate_power(antenna_object, tx_x, tx_y, tx_height, tx_power, subset_x_min, subset_x_max, subset_y_min, subset_y_max)
% CALCULATE_POWER Computes received power map for a transmitter using Quadriga.
%
%   OUTPUTS:
%     power_map : Grid of received power values (mW)
%     x_coords, y_coords : Coordinates associated with power_map

    antenna_object.tx_position(:,1) = [tx_x; tx_y; tx_height];

    [map, x_coords, y_coords] = antenna_object.power_map('3GPP_38.901_UMa_LOS', 'quick', 1, ...
        subset_x_min, subset_x_max, subset_y_min, subset_y_max, 1.5, tx_power);

    power_map = sum(cat(3, map{:}), 3)'; % Total received power in mW
end

function [df_users, df_users_table, avg_rate_connected_bpsHz, sum_rate_connected_bpsHz] = calculate_power_iterator(antenna_object, no_fbs, power_status, tx_y, tx_x, tx_height, tx_power, ...
    mbs_x, mbs_y, mbs_height, mbs_power, user_positions, ...
    subset_x_min, subset_x_max, subset_y_min, subset_y_max, ...
    threshold, containsMbs, antennaObjectMbs, mbsCache, bsBandIds, mbsSlotMap)
% CALCULATE_POWER_ITERATOR Evaluates SINR at each user for all BSs.
%
%   OUTPUTS:
%     df_users       : Raw matrix of power received at each user from each BS
%     df_users_table : Structured table of power and SINR values

    num_users = size(user_positions, 1);
    numSlots = size(mbsSlotMap, 1);
    df_users = zeros(num_users, no_fbs + numSlots);
    noise_power = 1e-11;

    % FBS loop
    for fbs_id = 1:no_fbs
        if power_status(fbs_id)
            if numel(antenna_object) >= fbs_id
                antThis = antenna_object(fbs_id);
            else
                antThis = antenna_object(1);
            end
            [P,x_coords,y_coords] = calculate_power(antThis, tx_x(fbs_id), tx_y(fbs_id), tx_height(fbs_id), ...
                                tx_power(fbs_id), subset_x_min, subset_x_max, subset_y_min, subset_y_max);
        else
            P = zeros(subset_x_max + 1, subset_y_max + 1);
        end

        df_users(:,fbs_id) = sample_nearest(P, user_positions);
    end

    % MBS slot loop. Each slot is one (mbsIdx, bandIdx, active) triple.
    % A base MBS contributes two slots (coverage + capacity); a femto/fixed
    % BS contributes one slot at its pinned band. Inactive slots emit a zero
    % column so the column layout stays stable across chromosome flips.
    if containsMbs && numSlots > 0
        num_mbs = size(antennaObjectMbs, 2);
        assert(num_mbs == size(mbsCache, 2), ...
            'num_mbs (%d) must equal cached sites (%d).', num_mbs, size(mbsCache, 2));
        nBandsCached = size(mbsCache, 1);
        refMap = mbsCache(1, 1).map;
        for ii = 1:numel(mbsCache)
            assert(isequal(size(mbsCache(ii).map), size(refMap)), ...
                'Cached MBS maps must be same size.');
        end

        for slot_idx = 1:numSlots
            mbsIdx  = mbsSlotMap(slot_idx, 1);
            bandIdx = mbsSlotMap(slot_idx, 2);
            active  = mbsSlotMap(slot_idx, 3) >= 0.5;
            col     = no_fbs + slot_idx;
            if ~active
                continue;  % df_users column already initialized to zero
            end
            assert(mbsIdx >= 1 && mbsIdx <= num_mbs, ...
                'Slot %d references mbsIdx %d outside [1, %d].', slot_idx, mbsIdx, num_mbs);
            assert(bandIdx >= 1 && bandIdx <= nBandsCached, ...
                'Slot %d band index %d out of range (cache has %d bands).', ...
                slot_idx, bandIdx, nBandsCached);
            P_mbs = mbsCache(bandIdx, mbsIdx).map;
            df_users(:, col) = sample_nearest(P_mbs, user_positions);
        end
    end

    df_users_table = convert_to_dataframe_style(df_users);

    % SINR evaluation
    max_sinr = -inf(num_users, 1);
    max_fbs_index = zeros(num_users, 1);

    bsIterator = size(df_users, 2);
    assert(numel(bsBandIds) == bsIterator, ...
        'bsBandIds (%d) must match number of BS power columns (%d).', numel(bsBandIds), bsIterator);

    for bs_id = 1:bsIterator
        sinr_column = zeros(num_users, 1);
        sameBandMask = (bsBandIds == bsBandIds(bs_id));
        parfor user_id = 1:num_users
            signal = df_users(user_id, bs_id);
            interference = sum(df_users(user_id, sameBandMask)) - signal;
            sinr = signal / (interference + noise_power);
            sinr_db = 10 * log10(sinr);
            sinr_column(user_id) = sinr_db;

            eta = log2(1 + sinr);

            if sinr_db >= threshold && sinr_db > max_sinr(user_id)
                max_sinr(user_id) = sinr_db;
                max_fbs_index(user_id) = bs_id;
            end
        end
        df_users_table.(['SINR_FBS_' num2str(bs_id)]) = sinr_column;
    end

    df_users_table.is_connected = max_sinr >= threshold;
    df_users_table.FBS_connection_index = max_fbs_index;
    % stats = summarize_bs_connections(df_users_table);
    [avg_rate_connected_bpsHz, total_connected_copy, sum_rate_connected] = global_avg_rate_connected(df_users_table);
    sum_rate_connected_bpsHz = sum_rate_connected;
end

function df_users_table = convert_to_dataframe_style(df_users)
% CONVERT_TO_DATAFRAME_STYLE Converts power matrix into a table with labeled columns.
%
%   INPUT:
%     df_users         : num_users × num_base_stations matrix
%   OUTPUT:
%     df_users_table   : MATLAB table with named columns for each FBS

    [~, no_fbs] = size(df_users);
    col_names = arrayfun(@(i) sprintf('power_FBS_%d', i), 1:no_fbs, 'UniformOutput', false);
    df_users_table = array2table(df_users, 'VariableNames', col_names);
end

function user_powers = sample_nearest(P, user_positions)
    % P: (ny x nx), ny = 3001, nx = 3001, covering x,y in [0,3000]
    % user_positions: (N x 2) [x y], within [0,3000]
    % Returns: (N x 1) power values from P at nearest grid point

    ix = round(user_positions(:,1)) + 1;  % columns (x)
    iy = round(user_positions(:,2)) + 1;  % rows (y)

    % Clamp to valid indices (handles rare boundary rounding)
    [nx, ny] = size(P);
    ix = min(max(ix, 1), nx);
    iy = min(max(iy, 1), ny);

    % Linear indexing
    user_powers = P(sub2ind([nx, ny], ix, iy));
end

% obtain the average sum rate per connected cluster
function per_bs_stats = summarize_bs_connections(df_users_table)
% SUMMARIZE_BS_CONNECTIONS
% For a user-level table with columns:
%   power_FBS_i, SINR_FBS_i (dB), is_connected (0/1), FBS_connection_index
% returns a table with per-BS connected counts and average spectral efficiency.
%
% Output columns:
%   bs_index, connected_count, avg_rate_bpsHz
%
% Notes:
% - SINR columns are assumed in dB and converted to linear before log2(1+SINR).
% - avg_rate_bpsHz is NaN if no users are connected to that BS.
%
% Example:
%   stats = summarize_bs_connections(df_users_table);

    % ---- Validate required columns ----
    reqCols = {'is_connected','FBS_connection_index'};
    for c = reqCols
        if ~ismember(c{1}, df_users_table.Properties.VariableNames)
            error('Input table missing required column: %s', c{1});
        end
    end

    % ---- Discover number of BS from SINR column names ----
    allVars = string(df_users_table.Properties.VariableNames);
    sinrCols = allVars(startsWith(allVars,"SINR_FBS_"));
    if isempty(sinrCols)
        error('No SINR_FBS_* columns found.');
    end

    % Extract BS indices from names like "SINR_FBS_3"
    bs_idx = sort(unique(str2double(extractAfter(sinrCols, "SINR_FBS_")),'stable'));
    n_bs   = numel(bs_idx);

    % Basic columns
    is_conn = logical(df_users_table.is_connected);                 % ensure logical
    conn_ix = df_users_table.FBS_connection_index;                  % numeric indices

    connected_count  = zeros(n_bs,1);
    avg_rate_bpsHz   = nan(n_bs,1);                                 % NaN if none connected

    % ---- Compute per-BS stats ----
    for k = 1:n_bs
        b = bs_idx(k);

        % Users connected to BS b
        mask_b = is_conn & (conn_ix == b);
        connected_count(k) = sum(mask_b);

        % Corresponding SINR (dB) column for BS b
        colname = "SINR_FBS_" + string(b);
        if ~ismember(colname, allVars)
            % If column missing, leave NaN average for this BS
            continue;
        end

        sinr_db_vec = df_users_table.(colname);
        % Convert dB -> linear then compute spectral efficiency
        sinr_lin_vec = 10.^(sinr_db_vec/10);
        rates_bpsHz  = log2(1 + sinr_lin_vec);

        if connected_count(k) > 0
            % Sum over connected users, divide by number connected
            avg_rate_bpsHz(k) = mean(rates_bpsHz(mask_b), 'omitnan');
        end
    end

    per_bs_stats = table(bs_idx(:), connected_count, avg_rate_bpsHz, ...
        'VariableNames', {'bs_index','connected_count','avg_rate_bpsHz'});
end

% obtain the average sum rate over the global connected user base 
function [avg_rate_connected_bpsHz, total_connected, sum_rate_connected] = global_avg_rate_connected(df_users_table)
% GLOBAL_AVG_RATE_CONNECTED
% Compute the average spectral efficiency across all connected users:
%   avg_rate = (sum over connected users of log2(1 + SINR_linear_of_connected_BS)) / (#connected users)
%
% Assumptions:
% - Table has columns: SINR_FBS_i (dB), is_connected (0/1), FBS_connection_index.
% - Users connected to BS b must have a corresponding column "SINR_FBS_b".
%
% Outputs:
%   avg_rate_connected_bpsHz : scalar average rate over connected users (bps/Hz)
%   total_connected          : number of connected users
%   sum_rate_connected       : sum of per-user rates over connected users (bps/Hz)

    % ---- Validate required columns ----
    reqCols = {'is_connected','FBS_connection_index'};
    for c = reqCols
        if ~ismember(c{1}, df_users_table.Properties.VariableNames)
            error('Input table missing required column: %s', c{1});
        end
    end

    % ---- Discover SINR columns and BS indices ----
    allVars = string(df_users_table.Properties.VariableNames);
    sinrCols = allVars(startsWith(allVars, "SINR_FBS_"));
    if isempty(sinrCols)
        error('No SINR_FBS_* columns found.');
    end
    bs_idx = sort(unique(str2double(extractAfter(sinrCols, "SINR_FBS_")),'stable'));
    n_bs   = numel(bs_idx);

    % ---- Basic columns ----
    is_conn = logical(df_users_table.is_connected);
    conn_ix = df_users_table.FBS_connection_index;  % values are BS indices (e.g., 1..N, or any integer set)

    % ---- Early exit: no connected users ----
    total_connected = sum(is_conn);
    if total_connected == 0
        avg_rate_connected_bpsHz = NaN;
        sum_rate_connected       = 0;
        return;
    end

    % ---- Build SINR (dB) matrix [nUsers x n_bs] with columns ordered by bs_idx ----
    nUsers = height(df_users_table);
    SINR_dB_mat = nan(nUsers, n_bs);
    for k = 1:n_bs
        colname = "SINR_FBS_" + string(bs_idx(k));
        if ismember(colname, allVars)
            SINR_dB_mat(:, k) = df_users_table.(colname);
        else
            % leave NaNs if a column is missing
        end
    end

    % ---- For connected users, select the SINR column matching their BS ----
    % Map each connection index value -> column position in SINR_dB_mat
    [tf_map, col_of_bs] = ismember(conn_ix, bs_idx);  % col_of_bs is 0 if no match

    % Mask for valid rows (connected AND we have that BS's SINR column)
    valid_rows = is_conn & tf_map & (col_of_bs > 0);

    if ~any(valid_rows)
        % No valid rows to aggregate (e.g., missing SINR column for all connected)
        avg_rate_connected_bpsHz = NaN;
        sum_rate_connected       = 0;
        total_connected          = sum(is_conn);  % keeps original count
        return;
    end

    % Row and column indices for linear indexing
    rows = find(valid_rows);
    cols = col_of_bs(valid_rows);

    % Pull the SINR(dB) for each user from their connected BS column
    lin_idx = sub2ind(size(SINR_dB_mat), rows, cols);
    sinr_db_selected = SINR_dB_mat(lin_idx);

    % Convert dB -> linear, compute per-user rates
    sinr_lin_selected = 10.^(sinr_db_selected/10);
    rates_bpsHz = log2(1 + sinr_lin_selected);

    % Sum and average over connected users.
    % Note: If some connected users are missing SINR (NaN), omit them from sum/avg.
    sum_rate_connected = sum(rates_bpsHz, 'omitnan');

    % Define denominator as the number of connected users with valid SINR rates
    denom = sum(~isnan(rates_bpsHz));
    if denom == 0
        avg_rate_connected_bpsHz = NaN;
    else
        avg_rate_connected_bpsHz = sum_rate_connected / denom;
    end

    % Optionally, if you prefer to divide by ALL connected users (including those with missing SINR),
    % replace denom with total_connected. The current choice uses only valid rates.
end
