function out = ppo_sinr_eval(tx, powerStatus, fbsBandFlags, mbsCapacityGenes, ...
        numUsers, threshold, collectUsers, worldId)
%PPO_SINR_EVAL One SINR evaluation against a world persisted by ppo_world_setup.
%
%   tx               : num_fbs x 4 matrix [x, y, z, power] per FBS
%   powerStatus      : 1 x num_fbs 0/1 activation vector
%   fbsBandFlags     : 1 x num_fbs 0/1 band ids ('multi' mode; [] in legacy)
%   mbsCapacityGenes : 1 x num_mbs 0/1 capacity toggles ('multi'; [] in legacy)
%   numUsers         : number of UEs to draw (fixed seed inside SINREvaluation)
%   threshold        : SINR connection threshold [dB]
%   collectUsers     : 1 to also return the user-position matrix
%   worldId          : key used at ppo_world_setup time (default 'default')
%
% In 'multi' mode: per-FBS antenna selection by band flag, base-MBS slot
% expansion (coverage always on, capacity gated by the gene), and same-band
% interference ids. In 'legacy' mode it issues the original train_ppo.py call
% (no band arguments, scalar antenna template) so historical agents replay
% bit-identically.
%
% Returns a scalar struct of metrics -- cheap to marshal back to Python.

    if nargin < 8 || isempty(worldId); worldId = 'default'; end
    world = getappdata(0, ['PPO_WORLD_' worldId]);
    assert(~isempty(world), ...
        'ppo_sinr_eval: world "%s" not initialized; call ppo_world_setup first.', worldId);

    tx = double(tx);
    if isvector(tx); tx = reshape(tx, 1, []); end
    assert(size(tx, 2) == 4, 'tx must be num_fbs x 4 [x y z power].');
    nFbs = size(tx, 1);
    x = tx(:, 1).'; y = tx(:, 2).'; z = tx(:, 3).'; p = tx(:, 4).';
    powerStatus = double(reshape(powerStatus, 1, []));

    if world.isMulti
        flags = double(reshape(fbsBandFlags, 1, []));
        if isempty(flags); flags = zeros(1, nFbs); end
        genes = double(reshape(mbsCapacityGenes, 1, []));
        if isempty(genes); genes = zeros(1, world.numMbs); end

        % Per-FBS antenna selection by band flag.
        fbsAntennaEval = repmat(world.fbsAntenna(1), 1, nFbs);
        capMask = flags >= 0.5;
        if any(capMask)
            fbsAntennaEval(capMask) = world.fbsAntenna(2);
        end

        % Base-MBS slot expansion (build_mbs_slots with every site a base
        % MBS): coverage slot always on, capacity slot gated by the gene.
        numMbs = world.numMbs;
        slotMap = zeros(2 * numMbs, 3);
        slotBands = zeros(1, 2 * numMbs);
        for j = 1:numMbs
            slotMap(2*j-1, :) = [j, 1, 1];
            slotMap(2*j, :)   = [j, 2, double(genes(j) >= 0.5)];
            slotBands(2*j-1)  = 0;
            slotBands(2*j)    = 1;
        end
        bsBandIds = [flags, slotBands];

        [user_positions, ~, tot, pwr, avgRate, fbsConn, mbsConn, sumRate, covConn, capConn] = ...
            SINREvaluation(fbsAntennaEval, powerStatus, x, y, z, nFbs, p, ...
                world.mbs_row2, world.mbs_row1, world.mbs_height, world.mbs_power, ...
                0, world.W, 0, world.H, numUsers, threshold, ...
                world.containsMbs, world.antennaObjectMbs, world.mbsCache, ...
                bsBandIds, slotMap);
    else
        % Legacy single-band world. Note: SINREvaluation's no-band-args
        % fallback defaults MBS slots to the capacity band (bandIdx 2), which
        % asserts on a 1-band cache — so the original train_ppo.py call
        % pattern no longer runs on this branch. Passing explicit ids
        % reproduces the true pre-multi-band physics: every transmitter on
        % one shared band (all mutual interference), one always-on slot per
        % MBS sampled from the single cached map.
        numMbs = world.numMbs;
        slotMap = [(1:numMbs).', ones(numMbs, 1), ones(numMbs, 1)];
        bsBandIds = zeros(1, nFbs + numMbs);
        [user_positions, ~, tot, pwr, avgRate, fbsConn, mbsConn, sumRate, covConn, capConn] = ...
            SINREvaluation(world.fbsAntenna(1), powerStatus, x, y, z, nFbs, p, ...
                world.mbs_row2, world.mbs_row1, world.mbs_height, world.mbs_power, ...
                0, world.W, 0, world.H, numUsers, threshold, ...
                world.containsMbs, world.antennaObjectMbs, world.mbsCache, ...
                bsBandIds, slotMap);
    end

    out = struct( ...
        'total_connected', double(tot), ...
        'fbs_connected', double(fbsConn), ...
        'mbs_connected', double(mbsConn), ...
        'mbs_coverage_connected', double(covConn), ...
        'mbs_capacity_connected', double(capConn), ...
        'total_power', double(pwr), ...
        'avg_rate', double(avgRate), ...
        'sum_rate', double(sumRate));
    if collectUsers
        out.user_positions = user_positions;
    end
end
