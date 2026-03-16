%% collect_csv_modes.m
clear; clc;

%% ===== 사용자 설정 =====
modelName = "main_model";
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

%% ===== Fast Restart ====
try
    set_param(modelName, "FastRestart", "on");
catch
    warning("[WARN] FastRestart를 켤 수 없습니다.");
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
        [I_data, HF_energy_data, HF_rms_data, T_data, t_data] = readSignals(simOut);

        % 길이 맞추기
        N = min([numel(I_data), numel(HF_energy_data), numel(HF_rms_data), numel(T_data), numel(t_data)]);
        I_data         = I_data(1:N);
        HF_energy_data = HF_energy_data(1:N);
        HF_rms_data    = HF_rms_data(1:N);
        T_data         = T_data(1:N);
        t_data         = t_data(1:N);

        % mode 열 (루프 변수 기준)
        mode_col = m * ones(N,1);

        % 최종 저장 데이터
        % [I_meas, HF_energy, HF_rms, T_meas, mode]
        data = [I_data, HF_energy_data, HF_rms_data, T_data, mode_col];

        % CSV 저장
        csvName = fullfile(saveFolder, sprintf("mode%d_run%02d.csv", m, runIdx));

        header = {'I_meas','HF_energy','HF_rms','T_meas','mode'};
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
    % 1순위: 이름이 MODE인 Constant 블록
    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "MODE");

    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    % 2순위: 이름이 mode_cmd인 Constant 블록
    blks = find_system(modelName, ...
        "SearchDepth", 3, ...
        "BlockType", "Constant", ...
        "Name", "mode_cmd");

    if ~isempty(blks)
        blk = string(blks{1});
        return;
    end

    % 3순위: 이름에 hint(mode)가 들어가는 Constant 블록
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

function [I_data, HF_energy_data, HF_rms_data, T_data, t_data] = readSignals(simOut)
    names = simOut.who;
    ds = [];

    % -------------------------------------------------
    % 1) simOut 안에 dataset_out이 직접 있는 경우
    % -------------------------------------------------
    if any(strcmp(names, "dataset_out"))
        ds = simOut.get("dataset_out");
        fprintf("[INFO] dataset_out 사용\n");
    end

    % -------------------------------------------------
    % 2) simOut 안에 out이 있고, out.dataset이 있는 경우
    % -------------------------------------------------
    if isempty(ds) && any(strcmp(names, "out"))
        outVar = simOut.get("out");

        % 객체/struct 둘 다 대응
        if isstruct(outVar) && isfield(outVar, "dataset")
            ds = outVar.dataset;
            fprintf("[INFO] simOut의 out.dataset 사용 (struct)\n");
        else
            try
                ds = outVar.dataset;
                fprintf("[INFO] simOut의 out.dataset 사용 (object)\n");
            catch
            end
        end
    end

    % -------------------------------------------------
    % 3) base workspace의 out.dataset fallback
    % -------------------------------------------------
    if isempty(ds) && evalin("base", "exist('out','var')")
        outVar = evalin("base", "out");

        if isstruct(outVar) && isfield(outVar, "dataset")
            ds = outVar.dataset;
            fprintf("[INFO] base workspace의 out.dataset 사용 (struct)\n");
        else
            try
                ds = outVar.dataset;
                fprintf("[INFO] base workspace의 out.dataset 사용 (object)\n");
            catch
            end
        end
    end

    if isempty(ds)
        error("simOut 안에도 dataset_out이 없고, out.dataset도 찾지 못했습니다.");
    end

    % -------------------------------------------------
    % Dataset 안에서 신호 읽기
    % -------------------------------------------------
    I_data         = getDatasetSignal(ds, ["I_meas","I_sig","I_true"]);
    HF_energy_data = getDatasetSignal(ds, ["HF_energy","HF_sig"]);
    HF_rms_data    = getDatasetSignal(ds, ["HF_rms"]);
    T_data         = getDatasetSignal(ds, ["T_meas","T_sig","T_body"]);

    t_data = getDatasetTime(ds, I_data);
end

function x = getDatasetSignal(ds, candNames)
    % 1) 이름으로 직접 접근 시도
    for k = 1:numel(candNames)
        name = char(candNames(k));
        try
            elem = ds.get(name);
            vals = elem.Values;
            x = extractSignalData(vals);
            fprintf("[INFO] Dataset에서 '%s' 읽음\n", name);
            return;
        catch
        end
    end

    % 2) element 순회하면서 이름 비교
    try
        for i = 1:ds.numElements
            elem = ds{i};
            if isprop(elem, "Name")
                nm = string(elem.Name);
                if any(strcmpi(nm, candNames))
                    x = extractSignalData(elem.Values);
                    fprintf("[INFO] Dataset에서 '%s' 읽음\n", nm);
                    return;
                end
            end
        end
    catch
    end

    error("Dataset에서 후보 신호를 찾지 못했습니다: %s", strjoin(cellstr(candNames), ", "));
end

function t = getDatasetTime(ds, ref)
    % Dataset 안 첫 번째 timeseries의 Time 사용
    try
        for i = 1:ds.numElements
            elem = ds{i};
            vals = elem.Values;
            if isa(vals, "timeseries")
                t = vals.Time(:);
                return;
            end
        end
    catch
    end

    % fallback
    t = (0:numel(ref)-1).';
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
