%% collect_csv_modes.m
clear; clc;

%% ===== 사용자 설정 =====
modelName = "final_base_model";
stopTime = 1000;
numRunsPerMode = 10;
saveFolder = fullfile(pwd, "dataset_out");

modeBlockNameHint = "mode";

%% ===== 저장 폴더 =====
if ~exist(saveFolder, "dir")
    mkdir(saveFolder);
end

%% ===== 모델 로드 =====
load_system(modelName);

%% ===== MODE 블록 찾기 =====
modeBlk = findModeBlock(modelName, modeBlockNameHint);
fprintf("[INFO] MODE block found: %s\n", modeBlk);

%% ===== Fast Restart =====
try
    set_param(modelName, "FastRestart", "on");
catch
end

%% ===== 실행 =====
modeList = [0 1 2 3];

for m = modeList
    for runIdx = 1:numRunsPerMode
        fprintf("\n=============================\n");
        fprintf("[RUN] mode=%d, run=%d/%d\n", m, runIdx, numRunsPerMode);
        fprintf("=============================\n");

        % mode 설정
        set_param(modeBlk, "Value", num2str(m));

        % 재현성용 시드
        rng(1000*m + runIdx, "twister");

        % 시뮬레이션
        simOut = sim(modelName, ...
            "StopTime", num2str(stopTime), ...
            "ReturnWorkspaceOutputs", "on");

        % 디버그
        disp("[DEBUG] simOut contains:");
        disp(simOut.who)

        % ===== 신호 읽기 =====
        mustHave = {"I_sig","HF_sig","T_sig","tout"};
        names = simOut.who;
        for i = 1:numel(mustHave)
            if ~any(strcmp(names, mustHave{i}))
                error("simOut에 '%s'가 없습니다. 현재 포함 변수: %s", ...
                    mustHave{i}, strjoin(names, ", "));
            end
        end

        I  = simOut.get("I_sig");
        HF = simOut.get("HF_sig");
        T  = simOut.get("T_sig");
        t  = simOut.get("tout");

        % ===== timeseries / numeric 처리 =====
        I_data  = extractSignalData(I);
        HF_data = extractSignalData(HF);
        T_data  = extractSignalData(T);
        t_data  = extractTimeData(t, I_data);

        % 길이 맞추기
        N = min([numel(I_data), numel(HF_data), numel(T_data), numel(t_data)]);
        I_data  = I_data(1:N);
        HF_data = HF_data(1:N);
        T_data  = T_data(1:N);
        t_data  = t_data(1:N);

        mode_col = m * ones(N,1);

        % 최종 저장 데이터: [I, HF, T, mode]
        data = [I_data, HF_data, T_data, mode_col];

        % CSV 저장
        csvName = fullfile(saveFolder, sprintf("mode%d_run%02d.csv", m, runIdx));

        header = {'I_meas','HF_energy','T_meas','mode'};
        writecell(header, csvName);
        writematrix(data, csvName, "WriteMode", "append");

        fprintf("[SAVED] %s | size = [%d x %d]\n", csvName, size(data,1), size(data,2));
    end
end

%% ===== 종료 =====
try
    set_param(modelName, "FastRestart", "off");
catch
end

disp("모든 CSV 저장이 완료되었습니다.");

%% ===== 로컬 함수 =====
function blk = findModeBlock(modelName, hint)
    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "MODE");

    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "mode_cmd");

    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    cblks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant");

    for i = 1:numel(cblks)
        nm = lower(string(get_param(cblks{i}, "Name")));
        if contains(nm, lower(hint))
            blk = string(cblks{i});
            return;
        end
    end

    error("mode Constant 블록을 찾지 못했습니다. 이름을 확인하세요.");
end

function x = extractSignalData(sig)
    if isa(sig, "timeseries")
        x = sig.Data;
    elseif isnumeric(sig)
        x = sig;
    elseif isstruct(sig) && isfield(sig, "signals")
        x = sig.signals.values;
    else
        try
            x = sig.Data;
        catch
            error("신호 데이터를 추출할 수 없습니다. 타입: %s", class(sig));
        end
    end

    x = squeeze(x);
    x = x(:);
end

function t_out = extractTimeData(t, ref)
    if isa(t, "timeseries")
        t_out = t.Time;
    elseif isnumeric(t)
        t_out = t(:);
    else
        try
            t_out = t.Time;
        catch
            % tout이 없거나 이상하면 ref 길이 기준 인덱스 생성
            t_out = (0:numel(ref)-1).';
        end
    end

    t_out = t_out(:);
end
