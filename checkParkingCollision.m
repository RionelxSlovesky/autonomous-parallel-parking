function [collision, hitFront, hitRear, hitCurb, egoPoly, frontPoly, rearPoly] = ...
    checkParkingCollision(egoCenter, egoYaw, EgoLength, EgoWidth, ...
                          frontCenter, rearCenter, parkedLength, parkedWidth, CurbY)
% checkParkingCollision
% Uses rotated rectangles for ego + parked cars

    egoPoly   = carPolygon(egoCenter, egoYaw, EgoLength, EgoWidth);
    frontPoly = carPolygon(frontCenter, 0, parkedLength, parkedWidth);
    rearPoly  = carPolygon(rearCenter,  0, parkedLength, parkedWidth);

    hitFront = polygonsOverlapSAT(egoPoly, frontPoly);
    hitRear  = polygonsOverlapSAT(egoPoly, rearPoly);
    hitCurb  = egoHitsCurb(egoPoly, CurbY);

    collision = hitFront || hitRear || hitCurb;
end