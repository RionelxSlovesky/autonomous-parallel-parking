function tf = egoHitsCurb(egoPoly, CurbY)
% egoHitsCurb
% True if any corner goes below the curb line

    tf = any(egoPoly(:,2) < CurbY);
end