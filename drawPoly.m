function drawPoly(poly, colorVal, lineStyle)
% drawPoly
% Draw polygon outline

    if nargin < 3
        lineStyle = '--';
    end

    p = [poly; poly(1,:)];
    plot(p(:,1), p(:,2), 'Color', colorVal, 'LineStyle', lineStyle, 'LineWidth', 1.5);
end