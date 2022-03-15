function X = triangulate_many(xy1, xy2, P1, P2)
    % Arguments
    %     xy: Calibrated image coordinates in image 1 and 2
    %         [shape 3 x n]
    %     P:  Projection matrix for image 1 and 2
    %         [shape 3 x 4]
    % Returns
    %     X:  Dehomogenized 3D points in world frame
    %         [shape 4 x n]

    n = size(xy1, 2);
    X = zeros(4, n); % Placeholder, replace with your implementation
end
