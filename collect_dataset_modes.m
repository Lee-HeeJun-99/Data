%% collect_dataset_modes.m
% MODE 0~3 자동 시뮬레이션 + simOut에서 직접 신호 수집 + 저장
% 현재 모델은 simOut에: HF_sig, I_sig, T_sig, V_sig, tout 로 저장

clear; clc;

%% ===== 사용자 설정 =====
modelName   = "final_base_model";
stopTime    = 1000;                   % 각 MODE 시뮬레이션 시간(초)
saveFolder  = fullfile(pwd, "dataset_out");
saveMatName = fullfile(saveFolder, "dataset_modes.mat");

% 라벨
modeLabel = containers.Map('KeyType','double','ValueType','char');
modeLabel(0) = "normal";
modeLabel(1) = "loose";
modeLabel(2) = "arc";
modeLabel(3) = "overcurrent";

baseSeed = 1234;

%% ===== 준비 =====
if ~exist(saveFolder, "dir")
    mkdir(saveFolder);
end

load_system(modelName);

% MODE Constant 블록 찾기
modeBlk = findModeBlock(modelName);
fprintf("[INFO] MODE block: %s\n", modeBlk);

% 반복 실행 가속(가능하면)
try
    set_param(modelName, "FastRestart", "on");
catch
end

%% ===== 데이터 구조 =====
dataset = struct();
dataset.meta.modelName  = modelName;
dataset.meta.stopTime   = stopTime;
dataset.meta.createdAt  = char(datetime("now"));
dataset.runs = [];

%% ===== MODE 0~3 실행 =====
numRunsPerMode = 10;
modeOrder = repelem([0 1 2 3], numRunsPerMode);

for k = 1:numel(modeOrder)
    m = modeOrder(k);

    fprintf("\n=============================\n");
    fprintf("[RUN] MODE = %d (%s)\n", m, modeLabel(m));
    fprintf("=============================\n");

    % MODE 세팅
    set_param(modeBlk, "Value", num2str(m));

    % 랜덤 시드 고정(재현성)
    rng(baseSeed + 100*m + k, "twister");

    % 시뮬레이션 실행
    simOut = sim(modelName, ...
        "StopTime", num2str(stopTime), ...
        "ReturnWorkspaceOutputs", "on");

    % ---- simOut에서 직접 꺼내기 (out 없음) ----
    mustHave = {"V_sig","I_sig","HF_sig","T_sig","tout"};
    names = simOut.who;
    for i = 1:numel(mustHave)
        if ~any(strcmp(names, mustHave{i}))
            error("simOut에 '%s'가 없습니다. 현재 포함 변수: %s", ...
                mustHave{i}, strjoin(names, ", "));
        end
    end

    V  = simOut.get("V_sig");
    I  = simOut.get("I_sig");
    HF = simOut.get("HF_sig");
    T  = simOut.get("T_sig");
    t  = simOut.get("tout");
    % ------------------------------------------

    % timeseries가 아니라면 timeseries로 변환(안전)
    V  = toTS(V,  t);
    I  = toTS(I,  t);
    HF = toTS(HF, t);
    T  = toTS(T,  t);

    % run 저장
    run = struct();
    run.mode     = m;
    run.label    = modeLabel(m);
    run.seed     = baseSeed + m;
    run.stopTime = stopTime;

    run.signals = struct();
    run.signals.t  = t;
    run.signals.V  = V;
    run.signals.I  = I;
    run.signals.HF = HF;
    run.signals.T  = T;

    dataset.runs = [dataset.runs; run];
end

%% ===== 저장 =====
save(saveMatName, "dataset", "-v7.3");
fprintf("\n[SAVED] %s\n", saveMatName);

%% ===== 간단 확인 플롯 =====
quickPlotAllModes(dataset);

%% ===== 정리 =====
try
    set_param(modelName, "FastRestart", "off");
catch
end

disp("완료되었습니다.");

%% ===== 로컬 함수 =====
function modeBlk = findModeBlock(modelName)
    blks = find_system(modelName, "SearchDepth", 3, "BlockType", "Constant", "Name", "MODE");
    if ~isempty(blks)
        modeBlk = blks{1};
        return;
    end
    cblks = find_system(modelName, "BlockType", "Constant");
    for i = 1:numel(cblks)
        nm = lower(string(get_param(cblks{i}, "Name")));
        if contains(nm, "mode")
            modeBlk = cblks{i};
            return;
        end
    end
    error("MODE Constant 블록을 찾지 못했습니다. 블록 이름을 MODE로 해주세요.");
end

function ts = toTS(x, t)
    if isa(x, "timeseries")
        ts = x;
        return;
    end
    if isnumeric(x)
        ts = timeseries(x, t);
        return;
    end
    % Simulink signal object 형태일 수 있음
    try
        if isprop(x, "Values") && isa(x.Values, "timeseries")
            ts = x.Values;
            return;
        end
    catch
    end
    error("신호를 timeseries로 변환할 수 없습니다. 타입: %s", class(x));
end

function quickPlotAllModes(dataset)
    figure("Name", "All MODE Quick Check");
    tiledlayout(4,1);

    for k = 1:numel(dataset.runs)
        r = dataset.runs(k);
        nexttile;
        plot(r.signals.I.Time, r.signals.I.Data); hold on;
        plot(r.signals.HF.Time, r.signals.HF.Data);
        plot(r.signals.T.Time, r.signals.T.Data);
        hold off;
        title(sprintf("MODE=%d (%s): I, HF, T", r.mode, r.label));
        xlabel("Time (s)");
        legend("I","HF","T");
    end
end
