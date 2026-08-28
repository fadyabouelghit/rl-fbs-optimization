function [xs, ys] = generate_hex_sites(W, H, ISD, margin, numMbs)
% Generate a hexagonal grid then select exactly numMbs sites
% W, H: grid dimensions
% ISD : inter-site distance
% margin: guard distance from edges
% numMbs: desired number of MBSs

if nargin < 4, margin = 0; end

dx = ISD;                 
dy = ISD * sqrt(3)/2;     

% Full hex grid
x0 = margin; x_max = W - margin;
y0 = margin; y_max = H - margin;

ys_all = (y0:dy:y_max).';
xs_all = [];
ys_out = [];
for r = 1:numel(ys_all)
    y = ys_all(r);
    if mod(r,2)==1
        xrow = (x0:dx:x_max);
    else
        xrow = (x0 + dx/2):dx:x_max;
    end
    xs_all = [xs_all; xrow(:)];
    ys_out = [ys_out; y*ones(numel(xrow),1)];
end

% Clip to area
in = xs_all>=margin & xs_all<=W-margin & ys_out>=margin & ys_out<=H-margin;
xs_all = xs_all(in);  
ys_all = ys_out(in);

% Now we have potentially more than numMbs
N = numel(xs_all);

if numMbs > N
    error('Requested %d MBSs but only %d hex sites fit in area', numMbs, N);
end

% Strategy: pick the ones closest to the center (so you get a compact cluster)
center = [W/2, H/2];
dist2 = (xs_all - center(1)).^2 + (ys_all - center(2)).^2;
[~, idx] = sort(dist2, 'ascend');

xs = xs_all(idx(1:numMbs));
ys = ys_all(idx(1:numMbs));

end
