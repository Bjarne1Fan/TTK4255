function draw_point_cloud(X, I1, uv1, my_xlim, my_ylim, my_zlim)
    assert(size(X,2) == size(uv1,2), 'If you get this error message in Task 4, it probably means that you did not extract the inliers of all the arrays (uv1,uv2,xy1,xy2) before calling draw_point_cloud.');

    % We take I1 and uv1 as arguments in order to assign a color to each
    % 3D point, based on its pixel coordinates in one of the images.
    C = impixel(I1, uv1(1,:), uv1(2,:));

    % Matlab doesn't let you easily change the up-axis to match the
    % convention we use in the course (it assumes Z is upward). So
    % this code does a silly rearrangement of the Y and Z arguments.

    fig = figure(2);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.5, 0.04, 0.5, 0.6]);
    clf(fig);
    scatter3(X(1,:), X(3,:), X(2,:), 5, C, 'filled');
    grid on;
    box on;
    axis equal;
    axis vis3d;
    camproj perspective;
    ylim(my_zlim);
    xlim(my_xlim);
    zlim(my_ylim);
    set(gca, 'ZDir', 'reverse');
    xlabel('X');
    ylabel('Z');
    zlabel('Y');
    h = annotation('textbox', [0 0.1 0 0], 'String', '[Hover over the figure with your mouse to access the toolbar, and select the rotator tool to rotate the view.]', 'FitBoxToText', true);
end
