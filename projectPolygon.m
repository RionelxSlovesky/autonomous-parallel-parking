function vals = projectPolygon(poly, axisVec)
% projectPolygon
% Projects polygon points onto an axis

    axisVec = axisVec / norm(axisVec);
    vals = poly * axisVec(:);
end