classdef QCarParallelParkingPathEnv < rl.env.MATLABEnvironment
    properties
        Ts = 0.02;
        MaxSteps = 500;

        % Vehicle model
        Wheelbase = 0.256;
        SteerMin = -0.5;
        SteerMax = 0.5;
        TauSteerNominal = 0.16;
        Vmax = 0.35;

        % Ego vehicle size
        EgoLength = 0.32;
        EgoWidth  = 0.18;

        % Parked vehicle size
        ParkedCarLength = 0.36;
        ParkedCarWidth  = 0.20;

        % Slot geometry
        SlotLength = 0.72;
        SlotDepth  = 0.30;

        % Scene
        Geofence = [-1.5 1.5 -0.8 1.0];
        CurbY = -0.30;

        % Success
        GoalRadius = 0.08;
        GoalYawTol = deg2rad(10);
    end

    properties(Access = private)
        StepCount = 0;

        % State
        x; y; psi;
        v;
        delta;

        prevSteerCmd;
        prevVCmd;
        tauSteer;

        % Path
        PathXY
        PathYaw
        PathSpeed
        PathIndex

        % Memory
        prevPathErr;
        prevHeadErr;

        % Parked cars
        FrontCarCenter;
        RearCarCenter;
        TargetX;
        TargetY;
        TargetYaw;
    end

    methods
        function this = QCarParallelParkingPathEnv()
            obsDim = 6;
            actDim = 2;
            steerMin = -0.5;
            steerMax =  0.5;
            vmax = 0.35;

            obsInfo = rlNumericSpec([obsDim 1], ...
                "LowerLimit", -inf(obsDim,1), ...
                "UpperLimit",  inf(obsDim,1));
            obsInfo.Name = "obs";

            actInfo = rlNumericSpec([actDim 1], ...
                "LowerLimit", [steerMin; -vmax], ...
                "UpperLimit", [steerMax;  vmax]);
            actInfo.Name = "action";

            this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            this.SteerMin = steerMin;
            this.SteerMax = steerMax;
            this.Vmax = vmax;
            this.tauSteer = this.TauSteerNominal;

            this.updateSceneGeometry();
            reset(this);
        end

        function [refX, refY] = getReferencePath(this)
            refX = this.PathXY(:,1);
            refY = this.PathXY(:,2);
        end

        function obs = reset(this)
            this.StepCount = 0;
            this.tauSteer = this.TauSteerNominal;
            this.updateSceneGeometry();
            this.buildReferencePath();

            starts = [
                0.55, 0.18,  0.00;
                0.60, 0.20,  0.02;
                0.65, 0.22, -0.02
            ];
            idx = randi(size(starts,1));
            this.x = starts(idx,1);
            this.y = starts(idx,2);
            this.psi = starts(idx,3);

            this.v = 0;
            this.delta = 0;
            this.prevSteerCmd = 0;
            this.prevVCmd = 0;
            this.PathIndex = 1;

            obs = this.buildObservation();
            this.prevPathErr = abs(obs(1));
            this.prevHeadErr = abs(obs(2));
        end

        function [obs, reward, isDone, info] = step(this, action)
            this.StepCount = this.StepCount + 1;

            steerCmd = min(max(action(1), this.SteerMin), this.SteerMax);
            vCmd = min(max(action(2), -this.Vmax), this.Vmax);

            this.delta = this.delta + (this.Ts/this.tauSteer) * (steerCmd - this.delta);
            this.v = 0.9*this.v + 0.1*vCmd;

            this.x = this.x + this.Ts * this.v * cos(this.psi);
            this.y = this.y + this.Ts * this.v * sin(this.psi);
            this.psi = wrapToPi(this.psi + this.Ts * (this.v/this.Wheelbase) * tan(this.delta));

            this.advancePathIndex();

            obs = this.buildObservation();
            cte = obs(1);
            he = obs(2);
            distToWp = obs(3);
            speedErr = obs(6);

            pathErr = abs(cte);
            headErr = abs(he);

            pathProgress = this.prevPathErr - pathErr;
            headProgress = this.prevHeadErr - headErr;

            reward = 15.0*pathProgress + 6.0*headProgress;
            reward = reward - 0.8*abs(cte) - 0.3*abs(he) - 0.1*abs(speedErr);
            reward = reward - 0.01*(steerCmd - this.prevSteerCmd)^2 ...
                            - 0.01*(vCmd - this.prevVCmd)^2;

            if distToWp < 0.08
                reward = reward + 0.5;
            end

            isCollision = this.checkCollision();
            outOfBounds = this.checkOutOfBounds();
            isSuccess = this.checkSuccess();
            isTimeout = this.StepCount >= this.MaxSteps;

            isDone = isCollision || outOfBounds || isSuccess || isTimeout;

            if isSuccess
                reward = reward + 250;
            elseif isCollision
                reward = reward - 100;
            elseif outOfBounds
                reward = reward - 60;
            elseif isTimeout
                reward = reward - 25;
            end

            this.prevSteerCmd = steerCmd;
            this.prevVCmd = vCmd;
            this.prevPathErr = pathErr;
            this.prevHeadErr = headErr;

            info = struct;
            info.Success = isSuccess;
            info.Collision = isCollision;
            info.OutOfBounds = outOfBounds;
            info.x = this.x;
            info.y = this.y;
            info.psi = this.psi;
            info.v = this.v;
            info.delta = this.delta;
            info.PathIndex = this.PathIndex;
            info.CrossTrackError = cte;
            info.HeadingError = he;
        end
    end

    methods(Access = private)
        function updateSceneGeometry(this)
            this.TargetX = 0.0;
            this.TargetY = this.CurbY + this.SlotDepth/2;
            this.TargetYaw = 0.0;

            gapHalf = this.SlotLength / 2;
            carHalf = this.ParkedCarLength / 2;
            this.FrontCarCenter = [ gapHalf + carHalf, this.TargetY];
            this.RearCarCenter  = [-gapHalf - carHalf, this.TargetY];
        end

        function buildReferencePath(this)
            % Segment 1: forward approach
            x1 = linspace(0.60, 0.30, 25)';
            y1 = linspace(0.20, 0.18, 25)';

            % Segment 2: reverse turning arc into slot
            t = linspace(0,1,40)';
            x2 = 0.30 - 0.35*t;
            y2 = 0.18 - 0.28*(t.^1.2);

            % Segment 3: straighten inside slot
            x3 = linspace(-0.05, 0.00, 20)';
            y3 = linspace(-0.10, this.TargetY, 20)';

            xy = [x1 y1; x2 y2; x3 y3];

            yaw = zeros(size(xy,1),1);
            for i = 1:size(xy,1)-1
                d = xy(i+1,:) - xy(i,:);
                yaw(i) = atan2(d(2), d(1));
            end
            yaw(end) = this.TargetYaw;

            % Desired signed speed profile
            spd1 = 0.15*ones(length(x1),1);     % forward
            spd2 = -0.18*ones(length(x2),1);    % reverse
            spd3 = -0.10*ones(length(x3),1);    % slow reverse settle

            this.PathXY = xy;
            this.PathYaw = yaw;
            this.PathSpeed = [spd1; spd2; spd3];
        end

        function advancePathIndex(this)
            while this.PathIndex < size(this.PathXY,1)
                p = this.PathXY(this.PathIndex,:);
                if norm([this.x this.y] - p) < 0.10
                    this.PathIndex = this.PathIndex + 1;
                else
                    break;
                end
            end
        end

        function obs = buildObservation(this)
            idx = min(this.PathIndex, size(this.PathXY,1));
            wp = this.PathXY(idx,:);
            wpYaw = this.PathYaw(idx);
            wpSpeed = this.PathSpeed(idx);

            dx = wp(1) - this.x;
            dy = wp(2) - this.y;

            c = cos(wpYaw);
            s = sin(wpYaw);

            alongErr =  c*dx + s*dy; %#ok<NASGU>
            crossErr = -s*dx + c*dy;
            headErr = wrapToPi(wpYaw - this.psi);
            distToWp = hypot(dx,dy);
            speedErr = wpSpeed - this.v;

            obs = [crossErr; headErr; distToWp; this.v; this.delta; speedErr];
        end

        function tf = checkSuccess(this)
            posErr = norm([this.x - this.TargetX, this.y - this.TargetY]);
            yawErr = abs(wrapToPi(this.TargetYaw - this.psi));
            tf = posErr < this.GoalRadius && yawErr < this.GoalYawTol && abs(this.v) < 0.08;
        end

        function tf = checkOutOfBounds(this)
            tf = (this.x < this.Geofence(1)) || (this.x > this.Geofence(2)) || ...
                 (this.y < this.Geofence(3)) || (this.y > this.Geofence(4));
        end

        function tf = checkCollision(this)
            tf = false;

            ego = this.getEgoAABB();
            front = this.getRectAABB(this.FrontCarCenter, this.ParkedCarLength, this.ParkedCarWidth);
            rear  = this.getRectAABB(this.RearCarCenter,  this.ParkedCarLength, this.ParkedCarWidth);

            if this.aabbOverlap(ego, front) || this.aabbOverlap(ego, rear)
                tf = true;
                return;
            end

            if ego(3) < this.CurbY
                tf = true;
            end
        end

        function box = getEgoAABB(this)
            halfL = this.EgoLength / 2;
            halfW = this.EgoWidth  / 2;
            box = [this.x-halfL, this.x+halfL, this.y-halfW, this.y+halfW];
        end

        function box = getRectAABB(~, center, L, W)
            halfL = L / 2;
            halfW = W / 2;
            box = [center(1)-halfL, center(1)+halfL, center(2)-halfW, center(2)+halfW];
        end

        function tf = aabbOverlap(~, a, b)
            tf = (a(1) <= b(2)) && (a(2) >= b(1)) && ...
                 (a(3) <= b(4)) && (a(4) >= b(3));
        end
    end
end