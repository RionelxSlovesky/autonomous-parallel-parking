function tf = polygonsOverlapSAT(poly1, poly2)
% polygonsOverlapSAT
% Separating Axis Theorem for convex polygons (here rectangles)

    tf = true;

    axes = [getAxes(poly1); getAxes(poly2)];

    for i = 1:size(axes,1)
        axisVec = axes(i,:);

        p1 = projectPolygon(poly1, axisVec);
        p2 = projectPolygon(poly2, axisVec);

        if max(p1) < min(p2) || max(p2) < min(p1)
            tf = false;
            return;
        end
    end
end

function axes = getAxes(poly)
    n = size(poly,1);
    axes = zeros(n,2);

    for k = 1:n
        p1 = poly(k,:);
        p2 = poly(mod(k,n)+1,:);

        edge = p2 - p1;
        normal = [-edge(2), edge(1)];

        if norm(normal) < 1e-12
            normal = [0, 0];
        else
            normal = normal / norm(normal);
        end

        axes(k,:) = normal;
    end
end