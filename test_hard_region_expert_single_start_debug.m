% test_hard_region_expert_single_start_debug.m
clear; clc; close all;

Ts = 0.02; MaxSteps = 12000;
Wheelbase = 0.256; SteerMin = -0.5; SteerMax = 0.5; TauSteer = 0.16; Vmax = 0.25;
EgoLength = 0.32; EgoWidth = 0.18;
ParkedCarLength = 0.36; ParkedCarWidth = 0.20;
SlotLength = 0.90; SlotDepth = 0.36;
Geofence = [-1.5 1.5 -0.8 1.0]; CurbY = -0.30;
TargetX = 0.0; TargetY = CurbY + SlotDepth/2; TargetYaw = 0.0;
gapHalf = SlotLength/2; carHalf = ParkedCarLength/2;
FrontCarCenter = [gapHalf + carHalf, TargetY];
RearCarCenter = [-gapHalf - carHalf, TargetY];
GoalRadius = 0.10; GoalYawTol = deg2rad(25); GoalSpeedTol = 0.06;
softStopPosTol = 0.2; softStopYawTol = deg2rad(20);

x = 0.04; y = 0.16; psi = deg2rad(-3); v = 0.0; delta = 0.0;
entryX = 0.75; reverseTurnYawTarget = -0.55; headingCorrectionTol = deg2rad(1.5);
steer_approach = 0.00;  vref_approach = 0.12;
steer_turnin = -0.24;   vref_turnin = -0.12;
steer_straight = 0.75;  vref_straight = -0.08;
steer_correct_l = -0.12; steer_correct_r = 0.18; vref_correct = -0.02;
steer_stop = 0.00; vref_stop = 0.00;
phase = "forward_approach";

X = nan(MaxSteps,1); Y = nan(MaxSteps,1); PSI = nan(MaxSteps,1); V = nan(MaxSteps,1);
DELTA = nan(MaxSteps,1); STEER_CMD = nan(MaxSteps,1); VREF = nan(MaxSteps,1);
PHASE_NAME = strings(MaxSteps,1);
hitFrontLog = false(MaxSteps,1); hitRearLog = false(MaxSteps,1); hitCurbLog = false(MaxSteps,1);
success = false; collision = false; outOfBounds = false;
finalEgoPoly = []; finalFrontPoly = []; finalRearPoly = [];

for k = 1:MaxSteps
    posErr = hypot(x-TargetX, y-TargetY);
    yawErrSigned = wrapToPiLocal(TargetYaw - psi);
    yawErr = abs(yawErrSigned);
    if posErr < softStopPosTol && yawErr < softStopYawTol, phase = "stop"; end

    switch phase
        case "forward_approach"
            if x >= entryX, phase = "reverse_turn_in"; end
        case "reverse_turn_in"
            if psi <= reverseTurnYawTarget || y <= (TargetY + 0.10), phase = "reverse_straighten"; end
        case "reverse_straighten"
            if posErr < 0.12 && yawErr < deg2rad(15), phase = "reverse_heading_correction"; end
        case "reverse_heading_correction"
            if posErr < GoalRadius && yawErr < GoalYawTol && abs(v) < 0.08, phase = "stop"; end
    end

    switch phase
        case "forward_approach"
            steerCmd = steer_approach; vRef = vref_approach;
        case "reverse_turn_in"
            steerCmd = steer_turnin; vRef = vref_turnin;
        case "reverse_straighten"
            steerCmd = steer_straight; vRef = vref_straight;
        case "reverse_heading_correction"
            if yawErrSigned > headingCorrectionTol
                steerCmd = steer_correct_l;
            elseif yawErrSigned < -headingCorrectionTol
                steerCmd = steer_correct_r;
            else
                steerCmd = 0.0;
            end
            vRef = vref_correct;
        otherwise
            steerCmd = steer_stop; vRef = vref_stop;
    end

    steerCmd = min(max(steerCmd, SteerMin), SteerMax);
    if phase == "stop", vCmd = 2.5*(vRef-v); else, vCmd = 1.5*(vRef-v); end
    vCmd = min(max(vCmd, -Vmax), Vmax);

    delta = delta + (Ts/TauSteer)*(steerCmd - delta);
    v = 0.85*v + 0.15*vCmd;
    x = x + Ts*v*cos(psi);
    y = y + Ts*v*sin(psi);
    psi = wrapToPiLocal(psi + Ts*(v/Wheelbase)*tan(delta));

    [collision, hitFront, hitRear, hitCurb, egoPoly, frontPoly, rearPoly] = ...
        checkParkingCollision([x,y], psi, EgoLength, EgoWidth, FrontCarCenter, RearCarCenter, ParkedCarLength, ParkedCarWidth, CurbY);

    outOfBounds = (x < Geofence(1)) || (x > Geofence(2)) || (y < Geofence(3)) || (y > Geofence(4));
    posErr = hypot(x-TargetX, y-TargetY);
    yawErr = abs(wrapToPiLocal(TargetYaw - psi));
    success = (posErr < GoalRadius) && (yawErr < GoalYawTol) && (abs(v) < GoalSpeedTol);

    X(k)=x; Y(k)=y; PSI(k)=psi; V(k)=v; DELTA(k)=delta; STEER_CMD(k)=steerCmd; VREF(k)=vRef; PHASE_NAME(k)=phase;
    hitFrontLog(k)=hitFront; hitRearLog(k)=hitRear; hitCurbLog(k)=hitCurb;

    if success || collision || outOfBounds
        X=X(1:k);Y=Y(1:k);PSI=PSI(1:k);V=V(1:k);DELTA=DELTA(1:k);STEER_CMD=STEER_CMD(1:k);VREF=VREF(1:k);PHASE_NAME=PHASE_NAME(1:k);
        hitFrontLog=hitFrontLog(1:k); hitRearLog=hitRearLog(1:k); hitCurbLog=hitCurbLog(1:k);
        finalEgoPoly=egoPoly; finalFrontPoly=frontPoly; finalRearPoly=rearPoly;
        break;
    end
end

fprintf('\n==== Hard-Region Expert Single-Start Debug Test ====\n');
fprintf('Success: %d\nCollision: %d\nOutOfBounds: %d\n', success, collision, outOfBounds);
fprintf('Final x: %.4f\nFinal y: %.4f\nFinal yaw(deg): %.2f\n', X(end), Y(end), rad2deg(PSI(end)));
fprintf('Final pos err: %.4f\nFinal yaw err(deg): %.2f\n', hypot(X(end)-TargetX, Y(end)-TargetY), rad2deg(abs(wrapToPiLocal(TargetYaw-PSI(end)))));
fprintf('Final phase: %s\n', PHASE_NAME(end));
fprintf('hitFront: %d\nhitRear : %d\nhitCurb : %d\n', hitFrontLog(end), hitRearLog(end), hitCurbLog(end));

figure('Name','Hard-Region Expert Single-Start Debug'); hold on; grid on; axis equal;
drawCar(RearCarCenter,0,ParkedCarLength,ParkedCarWidth,[0.7 0.7 0.7]);
drawCar(FrontCarCenter,0,ParkedCarLength,ParkedCarWidth,[0.7 0.7 0.7]);
drawCar([TargetX,TargetY],TargetYaw,EgoLength,EgoWidth,[0.2 0.8 0.2]);
plot(X,Y,'b.-','LineWidth',1.5,'MarkerSize',10); plot(X(1),Y(1),'go'); plot(X(end),Y(end),'ro');
plot([Geofence(1),Geofence(2)],[CurbY,CurbY],'k--','LineWidth',1.5);
rectangle('Position',[-SlotLength/2 CurbY SlotLength SlotDepth],'EdgeColor',[0 0.5 0],'LineStyle','--','LineWidth',1.5);
if ~isempty(finalEgoPoly)
    drawPoly(finalEgoPoly,[1 0 0],'--'); drawPoly(finalFrontPoly,[1 1 0],'--'); drawPoly(finalRearPoly,[1 1 0],'--');
end
xlabel('X (m)'); ylabel('Y (m)'); title('Hard-Region Expert Single-Start Debug');

function drawCar(center, yaw, L, W, colorVal)
    poly = carPolygon(center, yaw, L, W); p = [poly; poly(1,:)];
    patch(p(:,1), p(:,2), colorVal, 'FaceAlpha', 0.4, 'EdgeColor', 'k');
end
function a = wrapToPiLocal(a), a = mod(a + pi, 2*pi) - pi; end
