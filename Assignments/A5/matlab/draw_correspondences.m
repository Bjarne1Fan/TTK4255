function draw_correspondences(I1, I2, uv1, uv2, F)
    % Draws a random subset of point correspondences and their epipolar lines.

    fig = figure(1);
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8, 0.6]);
    clf(fig);

    k = 8; % Change this to adjust the number of pairs drawn
    sample = randperm(size(uv1, 2), k);
    uv1 = uv1(:,sample);
    uv2 = uv2(:,sample);

    l2 = F*uv1;
    l1 = F'*uv2;

    colors = lines(k);
    subplot(121);
    imshow(I1);
    hold on;
    for i=1:k
        hline(l1(:,i), colors(i,:));
    end
    scatter(uv1(1,:), uv1(2,:), 100, colors, 'x', 'LineWidth', 2);
    xlabel('Image 1');
    title(sprintf('Point correspondences and associated epipolar lines (showing %d randomly drawn pairs)', k));
    subplot(122);
    imshow(I2);
    hold on;
    for i=1:k
        hline(l2(:,i), colors(i,:));
    end
    scatter(uv2(1,:), uv2(2,:), 100, colors, 'o', 'LineWidth', 2);
    xlabel('Image 2');
end

function hline(l, color)
    % Draws a homogeneous 2D line.
    % You must explicitly set the figure xlim, ylim before or after using this.

    lim = [-1e8, +1e8]; % Surely you don't have a figure bigger than this!
    a = l(1);
    b = l(2);
    c = l(3);
    if abs(a) > abs(b)
        x = -(c + b*lim)/a;
        y = lim;
    else
        x = lim;
        y = -(c + a*lim)/b;
    end
    plot(x, y, 'color', color, 'linewidth', 1.5, 'linestyle', '--');
end
