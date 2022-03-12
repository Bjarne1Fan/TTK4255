function [E,inliers] = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)

    % Tip: The following snippet extracts a random subset of 8
    % correspondences (w/o replacement) and estimates E using them.
    %   sample = randperm(size(xy1, 2), 8);
    %   E = estimate_E(xy1(:,sample), xy2(:,sample));

end
