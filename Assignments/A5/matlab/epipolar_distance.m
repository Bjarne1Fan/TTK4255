function e = epipolar_distance(F, uv1, uv2)
    % F should be the fundamental matrix (use F_from_E)
    % uv1, uv2 should be 3 x n homogeneous pixel coordinates

    n = size(uv1, 2);
    e = zeros(1, n); % Placeholder, replace with your implementation
end
