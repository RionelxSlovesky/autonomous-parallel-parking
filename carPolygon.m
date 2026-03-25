function poly = carPolygon(center, yaw, L, W)
% carPolygon
% Returns 4x2 polygon corners of a rotated rectangle (vehicle body)

    c = cos(yaw);
    s = sin(yaw);
    R = [c -s; s c];

    corners = [ L/2,  W/2;
                L/2, -W/2;
               -L/2, -W/2;
               -L/2,  W/2]';

    worldCorners = R * corners + center(:);

    poly = worldCorners';
end